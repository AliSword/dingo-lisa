import math
import numpy as np
import torch
import pandas as pd
from bilby.gw.detector.interferometer import Interferometer
from lal import GreenwichMeanSiderealTime
from typing import Union
from bilby.gw.detector import calibration
from bilby.gw.prior import CalibrationPriorDict
from bilby.gw.detector import InterferometerList
from dingo.gw.lisa import LISAInterferometerList
from dingo.gw.lisa import LISALowFrequencyInterferometer
from dingo.gw.gwutils import get_optimal_snr, get_inner_product
#import h5py

CC = 299792458.0


def time_delay_from_geocenter(
    ifo: Interferometer,
    ra: Union[float, np.ndarray, torch.Tensor],
    dec: Union[float, np.ndarray, torch.Tensor],
    time: float,
):
    """
    Calculate time delay between ifo and geocenter. Identical to method
    ifo.time_delay_from_geocenter(ra, dec, time), but the present implementation allows
    for batched computation, i.e., it also accepts arrays and tensors for ra and dec.

    Implementation analogous to bilby-cython implementation
    https://git.ligo.org/colm.talbot/bilby-cython/-/blob/main/bilby_cython/geometry.pyx,
    which is in turn based on XLALArrivaTimeDiff in TimeDelay.c.

    Parameters
    ----------
    ifo: bilby.gw.detector.interferometer.Interferometer
        bilby interferometer object.
    ra: Union[float, np.array, torch.Tensor]
        Right ascension of the source in radians. Either float, or float array/tensor.
    dec: Union[float, np.array, torch.Tensor]
        Declination of the source in radians. Either float, or float array/tensor.
    time: float
        GPS time in the geocentric frame.

    Returns
    -------
    float: Time delay between the two detectors in the geocentric frame
    """

    if isinstance(ra, np.floating):
        ra = float(ra)
    if isinstance(dec, np.floating):
        dec = float(dec)
        
    # check that ra and dec are of same type and length
    if isinstance(ra, (np.ndarray, torch.Tensor)):
        if not isinstance(dec, type(ra)):
            raise ValueError(
                f"ra and dec must be of the same array type. Got {type(ra)} and {type(dec)}."
            )
        if len(ra.shape) != 1:
            raise ValueError(f"Only one axis expected for ra and dec, got shape {ra.shape}.")
        if ra.shape != dec.shape:
            raise ValueError(
                f"Shapes of ra ({ra.shape}) and dec ({dec.shape}) don't match."
            )
        
    if isinstance(ra, (float, np.float32, np.float64)):
        return ifo.time_delay_from_geocenter(ra, dec, time)

    elif isinstance(ra, (np.ndarray, torch.Tensor)) and len(ra) == 1:
        return ifo.time_delay_from_geocenter(ra[0], dec[0], time)

    else:
        if isinstance(ra, np.ndarray):
            sin = np.sin
            cos = np.cos
        elif isinstance(ra, torch.Tensor):
            sin = torch.sin
            cos = torch.cos
        else:
            raise NotImplementedError(
                "ra, dec must be either float, np.ndarray, or torch.Tensor."
            )

        gmst = math.fmod(GreenwichMeanSiderealTime(float(time)), 2 * np.pi)
        phi = ra - gmst
        theta = np.pi / 2 - dec
        sintheta = sin(theta)
        costheta = cos(theta)
        sinphi = sin(phi)
        cosphi = cos(phi)
        detector_1 = ifo.vertex
        detector_2 = np.zeros(3)
        return (
            (detector_2[0] - detector_1[0]) * sintheta * cosphi
            + (detector_2[1] - detector_1[1]) * sintheta * sinphi
            + (detector_2[2] - detector_1[2]) * costheta
        ) / CC


class GetDetectorTimes(object):
    """
    Compute the time shifts in the individual detectors based on the sky
    position (ra, dec), the geocent_time and the ref_time.
    """

    def __init__(self, ifo_list, ref_time):
        self.ifo_list = ifo_list
        self.ref_time = ref_time

    def __call__(self, input_sample):
        sample = input_sample.copy()
        # the line below is required as sample is a shallow copy of
        # input_sample, and we don't want to modify input_sample
        extrinsic_parameters = sample["extrinsic_parameters"].copy()
        ra = extrinsic_parameters["ra"]
        dec = extrinsic_parameters["dec"]
        geocent_time = extrinsic_parameters["geocent_time"]
        for ifo in self.ifo_list:
            if type(ra) == torch.Tensor:
                # computation does not work on gpu, so do it on cpu
                ra = ra.cpu()
                dec = dec.cpu()
            dt = time_delay_from_geocenter(ifo, ra, dec, self.ref_time)
            if type(dt) == torch.Tensor:
                dt = dt.to(geocent_time.device)
            ifo_time = geocent_time + dt
            extrinsic_parameters[f"{ifo.name}_time"] = ifo_time
        sample["extrinsic_parameters"] = extrinsic_parameters
        return sample


class ProjectOntoDetectors(object):
    """
    Project the GW polarizations onto the detectors in ifo_list. This does
    not sample any new parameters, but relies on the parameters provided in
    sample['extrinsic_parameters']. Specifically, this transform applies the
    following operations:

    (1) Rescale polarizations to account for sampled luminosity distance
    (2) Project polarizations onto the antenna patterns using the ref_time and
        the extrinsic parameters (ra, dec, psi)
    (3) Time shift the strains in the individual detectors according to the
        times <ifo.name>_time provided in the extrinsic parameters.
    """

    def __init__(self, ifo_list, domain, ref_time):
        self.ifo_list = ifo_list
        self.domain = domain
        self.ref_time = ref_time

    def __call__(self, input_sample):
        sample = input_sample.copy()
        # the line below is required as sample is a shallow copy of
        # input_sample, and we don't want to modify input_sample
        parameters = sample["parameters"].copy()
        extrinsic_parameters = sample["extrinsic_parameters"].copy()

        if isinstance(self.ifo_list, InterferometerList):
            try:
                d_ref = parameters["luminosity_distance"]
                d_new = extrinsic_parameters.pop("luminosity_distance")
                ra = extrinsic_parameters.pop("ra")
                dec = extrinsic_parameters.pop("dec")
                psi = extrinsic_parameters.pop("psi")
                tc_ref = parameters["geocent_time"]
                assert tc_ref == 0, (
                    "This should always be 0. If for some reason "
                    "you want to save time shifted polarizations,"
                    " then remove this assert statement."
                    )
                tc_new = extrinsic_parameters.pop("geocent_time")
                response_vars = [ra, dec, self.ref_time, psi]
            except KeyError as e:
                raise ValueError(f"Missing parameters: {e}")
        elif isinstance(self.ifo_list, LISAInterferometerList):
            try:
                d_ref = parameters["luminosity_distance"]
                d_new = extrinsic_parameters.pop("luminosity_distance")
                theta_s = extrinsic_parameters.pop("theta_s")
                phi_s = extrinsic_parameters.pop("phi_s")
                #theta_l = extrinsic_parameters.pop("theta_l")
                #phi_l = extrinsic_parameters.pop("phi_l")
                psi = extrinsic_parameters.pop("psi")
                tc_ref = parameters["geocent_time"]
                theta_jn = parameters["theta_jn"]
                theta_l, phi_l = LISALowFrequencyInterferometer.GetEclipticAngularMomentum(theta_jn, theta_s, phi_s, psi)
                assert tc_ref == 0
                tc_new = extrinsic_parameters.pop("geocent_time")
                response_vars = [theta_s, phi_s, theta_l, phi_l, self.ref_time] 
            except KeyError as e:
                raise ValueError(f"Missing parameter: {e}")

        # (1) rescale polarizations and set distance parameter to sampled value
        hc = sample["waveform"]["h_cross"] * d_ref / d_new
        hp = sample["waveform"]["h_plus"] * d_ref / d_new
        parameters["luminosity_distance"] = d_new

        strains = {}
        for ifo in self.ifo_list:
            # (2) project strains onto the different detectors
            fp = ifo.antenna_response(*response_vars, mode="plus")
            fc = ifo.antenna_response(*response_vars, mode="cross")
            strain = fp * hp + fc * hc

            if isinstance(self.ifo_list, InterferometerList):
                try:
                    # (3) time shift the strain. If polarizations are timeshifted by
                    #     tc_ref != 0, undo this here by subtracting it from dt.
                    dt = extrinsic_parameters[f"{ifo.name}_time"] - tc_ref
                    strains[ifo.name] = self.domain.time_translate_data(strain, dt)
                    

                    # Add extrinsic parameters corresponding to the transformations
                    # applied in the loop above to parameters. These have all been popped off of
                    # extrinsic_parameters, so they only live one place.
                    parameters["ra"] = ra
                    parameters["dec"] = dec
                    parameters["psi"] = psi
                    parameters["geocent_time"] = tc_new
                
                    # Add ifo time and popped off it of extrinsic parameters
                    param_name = f"{ifo.name}_time"
                    parameters[param_name] = extrinsic_parameters.pop(param_name)
                except KeyError:
                    print(f"Parameter {param_name} not found in extrinsic_parameters for {ifo.name}")
            elif isinstance(self.ifo_list, LISAInterferometerList):
                try:
                    dt = tc_new - tc_ref
                    strains[ifo.name] = self.domain.time_translate_data(strain, dt)
                    
                    # Add extrinsic parameters corresponding to the transformations
                    # applied in the loop above to parameters. These have all been popped off of
                    # extrinsic_parameters, so they only live one place. 
                    parameters["theta_s"] = theta_s
                    parameters["phi_s"] = phi_s
                    parameters["psi"] = psi
                    parameters["geocent_time"] = tc_new

                    # Add ecliptic angular momentum
                    parameters["theta_l"] = theta_l
                except KeyError as e:
                    print(f"Parameter {e.args[0]} not found for {ifo.name}")
                try:
                    parameters["phi_l"] = phi_l
                except KeyError as e:
                    print(f"Parameter {e.args[0]} not found for {ifo.name}")

        sample["waveform"] = strains
        sample["parameters"] = parameters
        sample["extrinsic_parameters"] = extrinsic_parameters

        return sample


class SampleSNRTargetLuminosityDistance(object):
    """
    SNR-conditioned reweighting of the training set (see project notes /
    Labrador paper Sec. IV.B for the derivation).

    Overrides the luminosity_distance sampled by SampleExtrinsicParameters so
    that the network *optimal* SNR of each example is drawn from a chosen
    target distribution (log-uniform between snr_min and snr_max), instead of
    whatever SNR distribution is induced by sampling distance directly from
    a (possibly physically-motivated) prior.

    This exploits the fact that, in ProjectOntoDetectors, the amplitude
    rescaling for luminosity distance is an EXACT, frequency-independent
    scalar factor (h *= d_ref / d_new) applied before antenna-pattern
    projection. Since SNR^2 is quadratic in the strain, this means exactly:

        SNR(D) = SNR_ref(theta_intrinsic, sky, psi) * (d_ref / D)

    where SNR_ref is the optimal SNR of the reference waveform (as stored,
    at distance d_ref) projected with this sample's sky location /
    polarization, evaluated against a FIXED reference ASD sampled once at
    construction time (NOT the per-sample ASD, which is (re-)sampled later
    in the pipeline by SampleNoiseASD -- using a fixed reference ASD here
    only affects how "SNR" is defined for the purpose of balancing the
    training set, not the actual noise added during training).

    Must be placed AFTER SampleExtrinsicParameters (needs sky/psi/time
    already sampled) and BEFORE ProjectOntoDetectors (which applies the real
    rescale + projection using the luminosity_distance value written here).

    The target SNR is drawn log-uniformly within a PER-SOURCE window that
    is the intersection of the globally requested [snr_min, snr_max] with
    the SNR range actually achievable while keeping D inside the physical
    distance prior's support [d_phys_min, d_phys_max] (different sources --
    different intrinsic parameters / sky / polarization -- reach different
    SNR at any fixed distance, so a fixed global window would often force D
    outside the physical prior for quieter or louder-than-typical sources).
    This guarantees every sample lands inside the physical prior's support
    (no example is ever discarded / gets zero importance weight), without
    narrowing the global ambition of the reweighting.

    Also computes and stores, under
    sample["extrinsic_parameters"]["snr_reweight_log_weight"], the log of
    the importance weight

        log[ p_physical(D) / p_proposal(D) ]

    needed to correct for this deliberate mismatch when computing the loss
    (the per-source truncated log-uniform proposal for the target SNR
    induces, for the resulting D, p_proposal(D) = 1 / (D * log_range), where
    log_range is the per-sample log-width of that source's SNR window --
    see project notes for the derivation). This is picked up downstream by
    AttachImportanceWeight.

    Parameters
    ----------
    ifo_list : InterferometerList or LISAInterferometerList
    domain : Domain
    ref_time : float
    asd_dataset : ASDDataset
        Used only to fix ONE reference ASD per detector (sampled once, here,
        at construction time) for defining the SNR target.
    snr_min, snr_max : float
        Support of the target (log-uniform) SNR distribution.
    physical_distance_prior_dict : bilby PriorDict-like object
        Must expose ln_prob({"luminosity_distance": D}). This is the
        distance prior the trained posterior should actually target (i.e.
        whatever was previously used for luminosity_distance in
        extrinsic_prior_dict).
    """

    def __init__(
        self,
        ifo_list,
        domain,
        ref_time,
        asd_dataset,
        snr_min,
        snr_max,
        physical_distance_prior_dict,
    ):
        self.ifo_list = ifo_list
        self.domain = domain
        self.ref_time = ref_time
        self.snr_min = float(snr_min)
        self.snr_max = float(snr_max)
        self.log_snr_min = np.log(self.snr_min)
        self.log_snr_max = np.log(self.snr_max)
        self.log_snr_range = self.log_snr_max - self.log_snr_min
        self.physical_distance_prior_dict = physical_distance_prior_dict
        dist_prior = physical_distance_prior_dict["luminosity_distance"]
        self.d_phys_min = float(dist_prior.minimum)
        self.d_phys_max = float(dist_prior.maximum)
        if not (np.isfinite(self.d_phys_min) and np.isfinite(self.d_phys_max)):
            raise ValueError(
                "SampleSNRTargetLuminosityDistance requires a physical "
                "luminosity_distance prior with finite minimum/maximum "
                f"(got minimum={self.d_phys_min}, maximum={self.d_phys_max})."
            )
        # Fix ONE reference ASD per detector, sampled once here, used
        # throughout for defining the SNR target (see class docstring).
        # Cast to float64: typical LISA ASD values (~1e-20) squared (~1e-40)
        # underflow in float32 (min positive normal ~1.18e-38), which
        # otherwise silently poisons the SNR computation below with NaN/inf.
        self.ref_asds = {
            k: v.astype(np.float64)
            for k, v in asd_dataset.sample_random_asds().items()
        }

    def _restrict_to_valid_band(self, arr):
        """
        Restrict the last axis of arr to the valid frequency band
        [domain.min_idx : domain.max_idx + 1], handling both the full-length
        (len(domain)) and pre-truncated (len(domain) - domain.min_idx)
        storage conventions used elsewhere in dingo (see e.g.
        Domain.get_sample_frequencies_astype).
        """
        n = arr.shape[-1]
        n_full = len(self.domain)
        if n == n_full:
            return arr[..., self.domain.min_idx : self.domain.max_idx + 1]
        elif n == n_full - self.domain.min_idx:
            return arr[..., : self.domain.max_idx + 1 - self.domain.min_idx]
        else:
            raise ValueError(
                f"Array with last dimension {n} is incompatible with domain "
                f"of length {n_full} (min_idx={self.domain.min_idx})."
            )

    def __call__(self, input_sample):
        sample = input_sample.copy()
        parameters = sample["parameters"]
        extrinsic_parameters = sample["extrinsic_parameters"].copy()

        d_ref = parameters["luminosity_distance"]
        # Cast to complex128: see note above on ref_asds re: float32 underflow
        # (the squared magnitude of the strain enters get_inner_product too).
        hp = sample["waveform"]["h_plus"].astype(np.complex128)
        hc = sample["waveform"]["h_cross"].astype(np.complex128)

        if isinstance(self.ifo_list, InterferometerList):
            ra = extrinsic_parameters["ra"]
            dec = extrinsic_parameters["dec"]
            psi = extrinsic_parameters["psi"]
            response_vars = [ra, dec, self.ref_time, psi]
        elif isinstance(self.ifo_list, LISAInterferometerList):
            theta_s = extrinsic_parameters["theta_s"]
            phi_s = extrinsic_parameters["phi_s"]
            psi = extrinsic_parameters["psi"]
            theta_jn = parameters["theta_jn"]
            theta_l, phi_l = LISALowFrequencyInterferometer.GetEclipticAngularMomentum(
                theta_jn, theta_s, phi_s, psi
            )
            response_vars = [theta_s, phi_s, theta_l, phi_l, self.ref_time]
        else:
            raise TypeError(f"Unsupported ifo_list type: {type(self.ifo_list)}")

        # SNR^2 of the reference waveform (at d_ref, no rescale) against the
        # FIXED reference ASD, summed over detectors. Restricted to the valid
        # frequency band [domain.min_idx : domain.max_idx + 1]: outside this
        # range the ASD (and possibly the signal) can take placeholder/extreme
        # values (e.g. near f=0, below f_min) that otherwise cause spurious
        # overflow/NaN in the division inside get_inner_product.
        snr_ref_squared = 0.0
        for ifo in self.ifo_list:
            fp = ifo.antenna_response(*response_vars, mode="plus")
            fc = ifo.antenna_response(*response_vars, mode="cross")
            strain_ref = self._restrict_to_valid_band(fp * hp + fc * hc)
            psd_f = self._restrict_to_valid_band(self.ref_asds[ifo.name] ** 2)
            snr_ref_squared += get_inner_product(
                strain_ref, strain_ref, psd_f, self.domain.delta_f
            )
        snr_ref = np.sqrt(snr_ref_squared)

        # Achievable target-SNR window for THIS source, given the physical
        # distance prior's support [d_phys_min, d_phys_max]. D =
        # d_ref*snr_ref/target_snr is monotonically decreasing in
        # target_snr, so the nearest allowed distance (d_phys_min) sets the
        # highest achievable SNR for this source, and the farthest allowed
        # distance (d_phys_max) sets the lowest. Sampling target_snr
        # log-uniformly WITHIN this per-source window (instead of the fixed
        # global [snr_min, snr_max]) guarantees d_new always lands inside
        # the physical prior's support -- no example is ever discarded /
        # gets zero weight, without narrowing the global ambition of the
        # reweighting (see project notes for the derivation and rationale).
        snr_achievable_max = d_ref * snr_ref / self.d_phys_min
        snr_achievable_min = d_ref * snr_ref / self.d_phys_max

        # Intersect with the globally desired [snr_min, snr_max] window.
        lo = max(self.snr_min, snr_achievable_min)
        hi = min(self.snr_max, snr_achievable_max)
        if lo > hi:
            # This source's achievable range doesn't overlap the globally
            # desired window at all (e.g. too quiet to ever reach snr_min
            # even at d_phys_min, or too loud to stay under snr_max even at
            # d_phys_max). Fall back to its own full achievable range, so
            # d_new still lands inside the physical prior support.
            lo, hi = snr_achievable_min, snr_achievable_max

        log_lo = np.log(lo)
        log_hi = np.log(hi)
        log_range = log_hi - log_lo

        target_snr = float(
            np.exp(np.random.uniform(log_lo, log_hi)) if log_range > 0 else lo
        )

        d_new = d_ref * snr_ref / target_snr

        if not np.isfinite(d_new) or d_new <= 0:
            raise ValueError(
                f"SampleSNRTargetLuminosityDistance produced a non-finite or "
                f"non-positive d_new={d_new} (snr_ref={snr_ref}, "
                f"target_snr={target_snr}, d_ref={d_ref}). This most likely "
                f"means snr_ref itself is non-finite -- check the reference "
                f"ASD for zero/extreme values within "
                f"[domain.min_idx, domain.max_idx]."
            )

        # Importance weight for this reparametrized distance draw. The
        # per-source truncated log-uniform proposal on target_snr induces,
        # for D, a log-uniform proposal p_proposal(D) = 1 / (D * log_range)
        # over D in [d_phys_min, d_phys_max] -- same closed form as the
        # untruncated case, but with a per-sample window width log_range
        # instead of a fixed global one (see project notes for derivation).
        if log_range > 0:
            log_p_proposal = -np.log(d_new) - np.log(log_range)
        else:
            # Degenerate window (achievable range collapsed to a point for
            # this source): d_new is forced to a single value with no
            # freedom, so there is no proposal density to divide out.
            log_p_proposal = 0.0
        log_p_physical = self.physical_distance_prior_dict.ln_prob(
            {"luminosity_distance": d_new}
        )
        log_weight = log_p_physical - log_p_proposal

        extrinsic_parameters["luminosity_distance"] = float(d_new)
        extrinsic_parameters["snr_reweight_log_weight"] = float(log_weight)
        extrinsic_parameters["snr_reweight_target_snr"] = float(target_snr)

        sample["extrinsic_parameters"] = extrinsic_parameters
        return sample

    @property
    def reproduction_dict(self):
        return {"snr_min": self.snr_min, "snr_max": self.snr_max}


class TimeShiftStrain(object):
    """
    Time shift the strains in the individual detectors according to the
    times <ifo.name>_time provided in the extrinsic parameters.
    """

    def __init__(self, ifo_list, domain):
        self.ifo_list = ifo_list
        self.domain = domain

    def __call__(self, input_sample):
        sample = input_sample.copy()
        extrinsic_parameters = input_sample["extrinsic_parameters"].copy()

        strains = {}

        if isinstance(input_sample["waveform"], dict):
            for ifo in self.ifo_list:
                # time shift the strain
                strain = input_sample["waveform"][ifo.name]
                dt = extrinsic_parameters.pop(f"{ifo.name}_time")
                strains[ifo.name] = self.domain.time_translate_data(strain, dt)

        elif isinstance(input_sample["waveform"], torch.Tensor):
            strains = input_sample["waveform"]
            dt = [extrinsic_parameters.pop(f"{ifo.name}_time") for ifo in self.ifo_list]
            dt = torch.stack(dt, 1)
            strains = self.domain.time_translate_data(strains, dt)

        else:
            raise NotImplementedError(
                f"Unexpected type {type(input_sample['waveform'])}, expected dict or "
                f"torch.Tensor"
            )

        sample["waveform"] = strains
        sample["extrinsic_parameters"] = extrinsic_parameters

        return sample


class ApplyCalibrationUncertainty(object):
    r"""
    Expand out a waveform using several detector calibration draws. These multiple
    draws are intended to be used for marginalizing over calibration uncertainty.

    Detector calibration uncertainty is modeled as described in
    https://dcc.ligo.org/LIGO-T1400682/public

    Gravitational wave data $d$ is assumed to be of the form

    $$d(f) = h_{obs}(f) + n(f),$$

    where $h_{obs}$ is the observed waveform and $n$ is the noise. Since the detector
    is not perfectly calibrated, the observed waveform is not identical to the true
    waveform $h(f)$. Rather, it is assumed to have corrections of the form

    $$h_{obs}(f) = h(f) * (1 + \delta A(f)) * \exp(i \delta \phi(f)),$$

    where $\delta A(f)$ and $\delta \phi(f)$ are frequency-dependent amplitude and
    phase errors. Under the calibration model, these are parametrized with cubic
    splines, defined in terms of calibration parameters $A_i$ and $\phi_i$, defined
    at log-spaced frequency nodes,

    $$
    \delta A(f) &= \mathrm{spline}(f; {f_i, \delta A_i}), \\
    \delta \phi(f) &= \mathrm{spline}(f; {f_i, \delta \phi_i}).
    $$

    The calibration parameters are not known precisely, rather they are assumed to be
    normally distributed, with mean 0 and standard deviation  determined by the
    "calibration envelope", which varies from event to event.

    For each detector waveform, this transform draws a collection of $N$
    calibration curves $\{(\delta A^n(f), \delta \phi^n(f))\}_{n=1}^N$ according to a
    calibration envelope, and applies them to generate $N$ observed waveforms $\{h^n_{
    obs}(f)\}$. This is intended to be used for marginalizing over the calibration
    uncertainty when evaluating the likelihood for importance sampling.
    """

    def __init__(
        self,
        ifo_list,
        data_domain,
        calibration_envelope,
        num_calibration_curves,
        num_calibration_nodes,
    ):
        r"""
        Parameters
        ---------

        ifo_list : InterferometerList
            List of Interferometers present in the analysis.
        data_domain : Domain
            Domain on which data is defined.
        calibration_envelope : dict
            Dictionary of the form ``{"H1": filepath, "L1": filepath}``,
            where the filepaths are strings pointing to ".txt" files containing
            calibration envelopes. The calibration envelope depends on the event analyzed,
            and therefore  remains fixed for all applications of the transform. The
            calibration envelope is used to define the variances $(\sigma_{\delta A_i},
            \sigma_{\delta \phi_i})$ of the calibration paramters.
        num_calibration_curves : int
            Number of calibration curves $N$ to produce and apply to the
            waveform. Ultimately, this will translate to the number of samples in the
            Monte Carlo estimate of the marginalized likelihood integral.
        num_calibration_nodes : int
            Number of log-spaced frequency nodes $f_i$ to use in defining the spline.
        """

        self.ifo_list = ifo_list
        self.num_calibration_curves = num_calibration_curves

        self.data_domain = data_domain
        self.calibration_prior = {}
        if all([s.endswith(".txt") for s in calibration_envelope.values()]):
            # Generating .h5 lookup table from priors in .txt file
            self.calibration_envelope = calibration_envelope
            for ifo in self.ifo_list:
                # Setting calibration model to cubic spline
                ifo.calibration_model = calibration.CubicSpline(
                    f"recalib_{ifo.name}_",
                    minimum_frequency=data_domain.f_min,
                    maximum_frequency=data_domain.f_max,
                    n_points=num_calibration_nodes,
                )

                # Setting priors
                # What this will do is take the the calibration envelope and set
                # a spline on the median and sigma of the amplitude and phase.
                # Then in log frequency it will setup node points say at
                # frequency points, $f_i$.  Then for each node point f_i, it
                # will create a gaussian prior according to the spline of the
                # median and sigma found earlier
                self.calibration_prior[
                    ifo.name
                ] = CalibrationPriorDict.from_envelope_file(
                    self.calibration_envelope[ifo.name],
                    self.data_domain.f_min,
                    self.data_domain.f_max,
                    num_calibration_nodes,
                    ifo.name,
                )

        else:
            raise Exception("Calibration envelope must be specified in a .txt file!")

    def __call__(self, input_sample):
        sample = input_sample.copy()
        for ifo in self.ifo_list:
            calibration_parameter_draws, calibration_draws = {}, {}
            # Sampling from prior
            calibration_parameter_draws[ifo.name] = pd.DataFrame(
                self.calibration_prior[ifo.name].sample(self.num_calibration_curves)
            )
            calibration_draws[ifo.name] = np.zeros(
                (
                    self.num_calibration_curves,
                    len(self.data_domain.sample_frequencies),
                ),
                dtype=complex,
            )

            for i in range(self.num_calibration_curves):
                calibration_draws[ifo.name][
                    i, self.data_domain.frequency_mask
                ] = ifo.calibration_model.get_calibration_factor(
                    self.data_domain.sample_frequencies[
                        self.data_domain.frequency_mask
                    ],
                    prefix="recalib_{}_".format(ifo.name),
                    **calibration_parameter_draws[ifo.name].iloc[i],
                )

            # Multiplying the sample waveform in the interferometer according to
            # the calibration curve.  This is done by following the perscription
            # here:
            #
            # https://dcc.ligo.org/LIGO-T1400682 Eq 3 and 4
            #
            # We take the waveform h(f) and multiply it by C = (1 + \delta A(f))
            # \exp(i \delta \psi) i.e. h_obs(f) = C * h(f)
            # Here C is "calibration_draws"

            # Padding 0's to everything in the calibration array which is below f_min

            sample["waveform"][ifo.name] = (
                sample["waveform"][ifo.name] * calibration_draws[ifo.name]
            )

        return sample

'''class ComputeSNR:
    def __init__(self):
        self.snr_storage = {}

    def __call__(self, sample, idx=None):
        waveforms = sample["waveform"]
        snr_dict = {}
        for channel, strain in waveforms.items():
            if channel not in self.snr_storage:
                self.snr_storage[channel] = []

        )
            snr = np.sqrt(np.sum(np.abs(strain)**2))
            snr_dict[channel] = snr

            self.snr_storage[channel].append(snr)


        sample["snr"] = snr_dict
        return sample


def save_snr_to_hdf5(snr_obj, hdf5_path):
    import h5py
    with h5py.File(hdf5_path, "w") as f:
        grp = f.create_group("snr")
        for channel, snrs in snr_obj.snr_storage.items():
            grp.create_dataset(channel, data=np.array(snrs))'''

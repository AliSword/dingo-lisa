"""
Calibration pass for REJECTION-SAMPLING SNR balancing (see project notes /
misc_scripts/snr_vs_params_marginals.py + chat discussion on 2026-09-15).

Unlike calibrate_snr_bins.py (which records each source's *achievable*
SNR window, for the "smart bins" importance-weighted mechanism), this script
measures the NATURAL (unweighted) histogram of realized network-optimal SNR
when luminosity_distance is drawn directly from the TRUE physical prior --
exactly what misc_scripts/diagnose_snr_distribution.py shows when SNR
reweighting is disabled. From this histogram we derive, per log-SNR bin, an
acceptance probability

    accept(bin) = min(1, p_hat_min / p_hat(bin))

where p_hat(bin) is the natural (normalized) probability of landing in that
bin and p_hat_min is the smallest nonzero p_hat over all bins. This is the
standard "undersample to match the rarest bin" recipe: bins that are
naturally common get thinned down to the rate of the rarest bin, so the
realized histogram of ACCEPTED samples becomes close to flat.

IMPORTANT CAVEAT (see chat): using this acceptance table at training time
(SampleSNRTargetLuminosityDistance with bin_accept_prob=..., no calibration
table for smart bins) draws D directly from the physical prior and keeps
weight=1 for every accepted sample -- i.e. NO importance weight is attached,
unlike the smart-bins mechanism. This means the network is trained on a
DELIBERATELY non-physical (flattened-in-SNR) prior for luminosity_distance,
and the RAW/uncorrected dingo posterior for distance will be biased relative
to the true physical prior. This is fine ONLY if you always apply the
existing importance-sampling correction (dingo/gw/importance_sampling,
result.importance_sample()) against the TRUE physical prior before trusting
or reporting distance results -- do not skip that step. This script and the
resulting acceptance table do not know or care about that; it is purely a
statistical calibration of the natural population histogram, exactly as
recommended for a first quick experiment ("proviamo e poi pensiamo alla
correzione se il training va bene").

Usage:
    python misc_scripts/calibrate_snr_rejection.py \
        --train_settings /sps/lisaf/aspadaro/snr_test/train_settings_baseline_100ep.yaml \
        --num_samples 20000 \
        --out calibration_snr_rejection.json

train_settings should point to a file where snr_reweighting is DISABLED (or
absent) and luminosity_distance uses the true physical prior -- e.g. the same
baseline file used for the Step-1 diagnostic -- so that the measured
histogram reflects the natural, undistorted population.
"""
import argparse
import json

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dingo.gw.training.train_builders import build_dataset, set_train_transforms
from dingo.gw.transforms import (
    AddWhiteNoiseComplex,
    RepackageStrainsAndASDS,
    SelectStandardizeRepackageParameters,
    UnpackDict,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_settings", required=True)
    parser.add_argument("--num_samples", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--snr_min", type=float, default=50.0)
    parser.add_argument("--snr_max", type=float, default=1500.0)
    parser.add_argument("--n_bins", type=int, default=16)
    parser.add_argument("--out", default="calibration_snr_rejection.json")
    args = parser.parse_args()

    with open(args.train_settings, "r") as f:
        train_settings = yaml.safe_load(f)

    data_settings = train_settings["data"]
    asd_dataset_path = train_settings["training"]["stage_0"]["asd_dataset_path"]

    if data_settings.get("snr_reweighting", {}).get("enabled", False):
        print(
            "[warn] snr_reweighting.enabled=true in this train_settings file -- "
            "the measured histogram will reflect the ALREADY-reweighted "
            "distribution, not the natural physical-prior one. Point this "
            "script at your baseline (non-reweighted) settings file instead."
        )

    print("Building waveform dataset...")
    wfd = build_dataset(data_settings)

    print("Setting training transforms (SNR reweighting/noise/repackaging omitted "
          "-- we want the RAW, physical-prior-driven distance draw)...")
    set_train_transforms(
        wfd,
        data_settings,
        asd_dataset_path,
        omit_transforms=[
            AddWhiteNoiseComplex,
            RepackageStrainsAndASDS,
            SelectStandardizeRepackageParameters,
            UnpackDict,
        ],
    )

    loader = DataLoader(wfd, batch_size=args.batch_size, num_workers=args.num_workers)

    snrs = []
    n_collected = 0
    for batch in loader:
        waveform = batch["waveform"]
        snr2 = None
        for ifo, strain in waveform.items():
            contrib = (strain.abs() ** 2).sum(dim=-1)
            snr2 = contrib if snr2 is None else snr2 + contrib
        snrs.append(torch.sqrt(snr2).cpu().numpy())
        n_collected += len(snrs[-1])
        if n_collected >= args.num_samples:
            break

    snrs = np.concatenate(snrs)[: args.num_samples]

    bin_edges = np.geomspace(args.snr_min, args.snr_max, args.n_bins + 1)
    in_range = (snrs >= args.snr_min) & (snrs < args.snr_max)
    frac_below = float(np.mean(snrs < args.snr_min))
    frac_above = float(np.mean(snrs >= args.snr_max))
    print(f"\nFraction of natural samples below snr_min={args.snr_min}: {frac_below:.4f}")
    print(f"Fraction of natural samples at/above snr_max={args.snr_max}: {frac_above:.4f}")
    print("(these tails are always accepted as-is at training time -- not thinned)")

    counts, _ = np.histogram(snrs[in_range], bins=bin_edges)
    n_in_range = counts.sum()
    if n_in_range == 0:
        raise RuntimeError(
            "No calibration samples fell within [snr_min, snr_max) -- check "
            "the range against the natural SNR distribution (see Step 1 "
            "diagnostic) before calibrating."
        )
    bin_prob = counts / n_in_range

    zero_bins = np.flatnonzero(counts == 0)
    if len(zero_bins) > 0:
        print(
            f"\n[warn] {len(zero_bins)} bin(s) had ZERO calibration samples "
            f"(indices {zero_bins.tolist()}) -- their statistics are "
            f"unreliable with only {args.num_samples} samples. Consider a "
            f"larger --num_samples or fewer --n_bins. These bins are given "
            f"accept_prob=1.0 (nothing to reject if we never saw one)."
        )

    p_min_nonzero = bin_prob[bin_prob > 0].min()
    accept_prob = np.where(
        bin_prob > 0,
        np.clip(p_min_nonzero / np.maximum(bin_prob, 1e-12), 1e-3, 1.0),
        1.0,
    )

    print("\nCalibrated natural bin probabilities p_hat(bin):")
    print(np.round(bin_prob, 4))
    print("\nDerived acceptance probabilities accept(bin) = min(1, p_min/p_hat(bin)):")
    print(np.round(accept_prob, 4))
    print(
        f"\nExpected retention rate under this scheme (fraction of in-range "
        f"draws accepted): {float(np.sum(bin_prob * accept_prob)):.4f}"
    )

    out = {
        "snr_min": args.snr_min,
        "snr_max": args.snr_max,
        "n_bins": args.n_bins,
        "bin_edges": bin_edges.tolist(),
        "bin_counts": counts.tolist(),
        "bin_prob": bin_prob.tolist(),
        "accept_prob": accept_prob.tolist(),
        "num_calibration_samples": int(len(snrs)),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved rejection-sampling calibration table to {args.out}")


if __name__ == "__main__":
    main()

import numpy as np


class UnpackDict(object):
    """
    Unpacks the dictionary to prepare it for final output of the dataloader.
    Only returns elements specified in selected_keys.
    """
    def __init__(self, selected_keys):
        self.selected_keys = selected_keys

    def __call__(self, input_sample):
        return [input_sample[k] for k in self.selected_keys]


class AttachImportanceWeight(object):
    """
    Reads a per-sample importance-sampling log-weight, if present (attached
    earlier in the transform chain, e.g. by SampleSNRTargetLuminosityDistance,
    under sample["extrinsic_parameters"]["snr_reweight_log_weight"]), and
    exposes it as sample["weight"] (a plain float, np.exp of the log-weight).

    If no such log-weight is present, weight = 1.0 (no reweighting) -- this
    makes the transform safe to add unconditionally, though in practice it
    is only appended to the training pipeline when SNR reweighting is
    enabled (see set_train_transforms).
    """

    def __call__(self, input_sample):
        sample = input_sample.copy()
        log_weight = sample.get("extrinsic_parameters", {}).get(
            "snr_reweight_log_weight", 0.0
        )
        sample["weight"] = float(np.exp(log_weight))
        return sample

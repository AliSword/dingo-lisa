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


class RejectionRetryTransform(object):
    """
    Dataset-level rejection sampling for SNR reweighting (see project notes /
    misc_scripts/calibrate_snr_rejection.py, and chat 2026-09-15).

    Wraps a two-phase transform chain around a WaveformDataset:

      - early_transform: everything up to and including
        SampleSNRTargetLuminosityDistance (cheap -- needs the raw
        polarizations and antenna response, but no noise/whitening). Marks
        extrinsic_parameters["_snr_reject"] = True/False.
      - late_transform: everything from ProjectOntoDetectors onward
        (the expensive part: real detector projection, noise, whitening).

    Must be installed directly as `wfd.transform` (not composed inside a
    plain torchvision.transforms.Compose), because it needs a reference to
    the dataset itself: when a candidate is rejected, retrying with a NEW
    distance for the SAME source cannot work (a source whose entire
    achievable SNR range falls in one bin can never land in another bin, no
    matter how many distances are tried for it). Instead, on rejection this
    fetches a genuinely DIFFERENT random source from the dataset (via
    wfd.get_raw, which bypasses wfd.transform to avoid infinite recursion)
    and retries the cheap early_transform on that one, up to max_tries
    times. Only once a candidate is accepted (or max_tries is exhausted, as
    a safety fallback) does it run the expensive late_transform, once.
    """

    # Lightweight class-level counters for diagnosing retry cost (printed
    # periodically, not per-call, so this stays cheap). See chat 2026-09-15
    # OOM investigation: we want to know if a small number of pathological
    # samples need close to max_tries, which would explain memory/CPU blowup
    # even though avg_tries is modest.
    _dbg_n_calls = 0
    _dbg_total_tries = 0
    _dbg_max_tries_seen = 0
    _dbg_hit_max_count = 0

    def __init__(self, wfd, early_transform, late_transform, max_tries=50):
        self.wfd = wfd
        self.early_transform = early_transform
        self.late_transform = late_transform
        self.max_tries = max_tries

    def __call__(self, data):
        candidate = self.early_transform(data)
        tries = 1
        while (
            candidate["extrinsic_parameters"].get("_snr_reject", False)
            and tries < self.max_tries
        ):
            new_idx = np.random.randint(len(self.wfd))
            raw = self.wfd.get_raw(new_idx)
            candidate = self.early_transform(raw)
            tries += 1

        cls = RejectionRetryTransform
        cls._dbg_n_calls += 1
        cls._dbg_total_tries += tries
        cls._dbg_max_tries_seen = max(cls._dbg_max_tries_seen, tries)
        if tries >= self.max_tries:
            cls._dbg_hit_max_count += 1
        if cls._dbg_n_calls % 500 == 0:
            print(
                f"[REJECTION RETRY DEBUG] n={cls._dbg_n_calls} "
                f"avg_tries={cls._dbg_total_tries / cls._dbg_n_calls:.1f} "
                f"max_tries_seen={cls._dbg_max_tries_seen} "
                f"hit_max_count={cls._dbg_hit_max_count} "
                f"(current max_tries={self.max_tries})",
                flush=True,
            )

        return self.late_transform(candidate)


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

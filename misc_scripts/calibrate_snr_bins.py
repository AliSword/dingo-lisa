"""
Calibration pass for the "smart" SNR-bin reweighting mode of
SampleSNRTargetLuminosityDistance (see project notes / conversation).

Context: sampling a per-source target SNR log-uniformly WITHIN each
source's own physically-achievable window (the fix that eliminates zero
importance weights) does NOT give a flat SNR histogram when aggregated
over the whole population -- most sources can only reach a "typical",
central range of SNR, so that region gets over-represented and the rare
very-quiet / very-loud sources' extreme ranges get under-represented.

This script measures, for a set of log-spaced SNR bins between snr_min and
snr_max, what fraction of sources in the population can actually reach
each bin while keeping D inside the physical distance prior's support
(the coverage r(bin)). At training time, SampleSNRTargetLuminosityDistance
uses 1/r(bin) to bias which bin each source is pushed into (among the ones
it can physically reach), so that sources capable of reaching a RARE bin
are sent there more often -- compensating for the imbalance -- while every
sample still stays inside the physical prior's support (importance weight
never exactly zero).

Usage:
    python calibrate_snr_bins.py \
        --train_settings /path/to/train_settings.yaml \
        --num_samples 20000 \
        --n_bins 16 \
        --out calibration_snr_bins.json

train_settings.yaml must have data.snr_reweighting.enabled: true with
snr_min/snr_max already set (this script calibrates against that exact
window). Point data.snr_reweighting.calibration_table at the resulting
JSON file to activate the smart-bin mode for actual training.
"""
import argparse
import json

import numpy as np
import yaml
from torch.utils.data import DataLoader

from dingo.gw.training.train_builders import build_dataset, set_train_transforms
from dingo.gw.transforms import (
    AddWhiteNoiseComplex,
    RepackageStrainsAndASDS,
    SampleSNRTargetLuminosityDistance,
    SelectStandardizeRepackageParameters,
    UnpackDict,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_settings", required=True, help="Path to train_settings.yaml")
    parser.add_argument("--num_samples", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--n_bins", type=int, default=16)
    parser.add_argument("--out", default="calibration_snr_bins.json")
    args = parser.parse_args()

    with open(args.train_settings, "r") as f:
        train_settings = yaml.safe_load(f)

    data_settings = train_settings["data"]
    asd_dataset_path = train_settings["training"]["stage_0"]["asd_dataset_path"]
    snr_reweight_settings = data_settings.get("snr_reweighting", {})
    if not snr_reweight_settings.get("enabled", False):
        raise ValueError(
            "train_settings must have data.snr_reweighting.enabled=true "
            "(with snr_min/snr_max already set) to calibrate against."
        )
    snr_min = float(snr_reweight_settings["snr_min"])
    snr_max = float(snr_reweight_settings["snr_max"])

    print("Building waveform dataset (intrinsic, from disk)...")
    wfd = build_dataset(data_settings)

    print("Setting training transforms (noise/repackaging omitted, as in "
          "diagnose_snr_distribution.py)...")
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

    # Switch the SampleSNRTargetLuminosityDistance transform already
    # inserted by set_train_transforms into calibration mode: it computes
    # snr_ref exactly as during real training (same geometry/response code
    # path), but records the source's achievable SNR window instead of
    # actually sampling a target SNR.
    found = False
    for t in wfd.transform.transforms:
        if isinstance(t, SampleSNRTargetLuminosityDistance):
            t.calibration_mode = True
            # calibration_mode needs d_phys_min/d_phys_max, which are only
            # computed in __init__ when smart_bins or calibration_mode was
            # already True at construction time -- since this run's YAML
            # has no calibration_table set yet, resolve them here too.
            if not hasattr(t, "d_phys_min"):
                dist_prior = t.physical_distance_prior_dict["luminosity_distance"]
                t.d_phys_min = float(dist_prior.minimum)
                t.d_phys_max = float(dist_prior.maximum)
            found = True
    if not found:
        raise RuntimeError(
            "No SampleSNRTargetLuminosityDistance transform found in the "
            "pipeline -- check that data.snr_reweighting.enabled=true."
        )

    loader = DataLoader(wfd, batch_size=args.batch_size, num_workers=args.num_workers)

    achievable_min = []
    achievable_max = []
    n_collected = 0
    print(f"Collecting achievable SNR windows for {args.num_samples} sources...")
    for batch in loader:
        ep = batch["extrinsic_parameters"]
        achievable_min.append(ep["calib_snr_achievable_min"].numpy())
        achievable_max.append(ep["calib_snr_achievable_max"].numpy())
        n_collected += len(ep["calib_snr_achievable_min"])
        if n_collected >= args.num_samples:
            break

    achievable_min = np.concatenate(achievable_min)[: args.num_samples]
    achievable_max = np.concatenate(achievable_max)[: args.num_samples]

    bin_edges = np.geomspace(snr_min, snr_max, args.n_bins + 1)
    lo_edges = bin_edges[:-1]
    hi_edges = bin_edges[1:]

    coverage = np.zeros(args.n_bins)
    for i in range(args.n_bins):
        overlap = (achievable_max > lo_edges[i]) & (achievable_min < hi_edges[i])
        coverage[i] = overlap.mean()

    print("\nCoverage r(bin) -- fraction of sources that can reach each bin:")
    for i in range(args.n_bins):
        print(f"  [{lo_edges[i]:8.2f}, {hi_edges[i]:8.2f}): r = {coverage[i]:.4f}")

    if np.any(coverage == 0):
        print(
            "\n[warn] Some bins have r=0 -- NO source in this calibration "
            "sample can reach them, so they will never be selectable at "
            "training time either. Consider a larger --num_samples, or "
            "narrowing snr_min/snr_max to the range that is actually "
            "achievable for this population."
        )

    out = {
        "snr_min": snr_min,
        "snr_max": snr_max,
        "n_bins": args.n_bins,
        "bin_edges": bin_edges.tolist(),
        "coverage": coverage.tolist(),
        "num_calibration_samples": int(len(achievable_min)),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved calibration table to {args.out}")


if __name__ == "__main__":
    main()

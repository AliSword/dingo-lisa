"""
Diagnostic script: measure the distribution of network optimal SNR in the
CURRENT training set (intrinsic dataset + on-the-fly extrinsic sampling, as
actually seen by the network during training), before deciding how to
rebalance it.

No changes to Dingo are needed to run this: it reuses build_dataset() and
set_train_transforms() exactly as train_pipeline.py does, just omitting the
noise-addition and final repackaging transforms so that sample["waveform"]
stays as a dict of per-detector, noiseless, WHITENED complex arrays. After
whitening (see WhitenAndScaleStrain docstring), noise would be unit-variance
white, so the network optimal SNR is simply:

    SNR = sqrt( sum_over_detectors_and_frequencies |whitened_strain|^2 )

Usage:
    python diagnose_snr_distribution.py \
        --train_settings /path/to/train_settings.yaml \
        --num_samples 5000

train_settings.yaml is the same file you already use for training (e.g.
examples/toy_npe_model/train_settings.yaml or your own). It must contain
train_settings["data"] and train_settings["training"]["stage_0"]["asd_dataset_path"].
"""
import argparse

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
    parser.add_argument("--train_settings", required=True, help="Path to train_settings.yaml")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_prefix", default="snr_diagnostic")
    args = parser.parse_args()

    with open(args.train_settings, "r") as f:
        train_settings = yaml.safe_load(f)

    data_settings = train_settings["data"]
    asd_dataset_path = train_settings["training"]["stage_0"]["asd_dataset_path"]

    print("Building waveform dataset (intrinsic, from disk)...")
    wfd = build_dataset(data_settings)

    print("Setting training transforms (extrinsic sampling on the fly, "
          "noise and final repackaging OMITTED for this diagnostic)...")
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
    print(f"Drawing samples exactly as training does (fresh extrinsic draw per sample)...")
    for batch in loader:
        waveform = batch["waveform"]  # dict: ifo -> complex tensor (batch, freq), whitened, noiseless
        snr2 = None
        for ifo, strain in waveform.items():
            contrib = (strain.abs() ** 2).sum(dim=-1)
            snr2 = contrib if snr2 is None else snr2 + contrib
        snr = torch.sqrt(snr2).cpu().numpy()
        snrs.append(snr)
        n_collected += len(snr)
        if n_collected >= args.num_samples:
            break

    snrs = np.concatenate(snrs)[: args.num_samples]
    # np.save(f"{args.out_prefix}_snrs.npy", snrs)  # disabled: not saving the raw array

    print(f"\nCollected {len(snrs)} samples of network optimal SNR.")
    print("Percentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{p:2d}: SNR = {np.percentile(snrs, p):8.2f}")
    print(f"\nFraction with SNR >  20: {np.mean(snrs > 20):.4f}")
    print(f"Fraction with SNR >  50: {np.mean(snrs > 50):.4f}")
    print(f"Fraction with SNR > 100: {np.mean(snrs > 100):.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(snrs, bins=60)
        axes[0].set_xlabel("network optimal SNR")
        axes[0].set_ylabel("count")
        axes[0].set_title("linear scale")

        pos = snrs[snrs > 0]
        bins = np.logspace(np.log10(max(pos.min(), 1e-3)), np.log10(pos.max()), 60)
        axes[1].hist(pos, bins=bins)
        axes[1].set_xscale("log")
        axes[1].set_xlabel("network optimal SNR (log scale)")
        axes[1].set_title("log-x scale")

        plt.tight_layout()
        out_path = f"{args.out_prefix}_histogram.pdf"
        plt.savefig(out_path)
        print(f"Saved histogram plot to {out_path}")
    except ImportError:
        print("matplotlib not available; skipping plot.")


if __name__ == "__main__":
    main()

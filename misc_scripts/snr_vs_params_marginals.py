"""
Simple diagnostic: how does the network optimal SNR relate to chirp mass and
to luminosity distance, in the CURRENT training set (no reweighting)?

This does not touch the reweighting mechanism at all -- it just samples from
the pipeline exactly as diagnose_snr_distribution.py does, and additionally
pulls out chirp_mass (or computes it from mass_1/mass_2 if chirp_mass is not
directly a stored parameter) and luminosity_distance for each sample, so we
can see which one is actually driving the shape of the SNR distribution.

Usage:
    python misc_scripts/snr_vs_params_marginals.py \
        --train_settings /sps/lisaf/aspadaro/snr_test/train_settings_baseline_100ep.yaml \
        --num_samples 5000
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


def to_numpy(x):
    return x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_settings", required=True)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_prefix", default="snr_vs_params")
    args = parser.parse_args()

    with open(args.train_settings, "r") as f:
        train_settings = yaml.safe_load(f)

    data_settings = train_settings["data"]
    asd_dataset_path = train_settings["training"]["stage_0"]["asd_dataset_path"]

    print("Building waveform dataset...")
    wfd = build_dataset(data_settings)

    print("Setting training transforms (noise / final repackaging omitted)...")
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

    snrs, chirp_masses, distances = [], [], []
    n_collected = 0
    printed_keys = False

    for batch in loader:
        if not printed_keys:
            print("\n[debug] top-level batch keys:", list(batch.keys()))
            if "parameters" in batch:
                print("[debug] batch['parameters'] keys:", list(batch["parameters"].keys()))
            if "extrinsic_parameters" in batch:
                print("[debug] batch['extrinsic_parameters'] keys:", list(batch["extrinsic_parameters"].keys()))
            printed_keys = True

        waveform = batch["waveform"]
        snr2 = None
        for ifo, strain in waveform.items():
            contrib = (strain.abs() ** 2).sum(dim=-1)
            snr2 = contrib if snr2 is None else snr2 + contrib
        snrs.append(torch.sqrt(snr2).cpu().numpy())

        params = batch.get("parameters", {})
        extrinsic = batch.get("extrinsic_parameters", {})

        if "chirp_mass" in params:
            chirp_masses.append(to_numpy(params["chirp_mass"]))
        elif "mass_1" in params and "mass_2" in params:
            m1 = to_numpy(params["mass_1"])
            m2 = to_numpy(params["mass_2"])
            chirp_masses.append((m1 * m2) ** 0.6 / (m1 + m2) ** 0.2)
        else:
            chirp_masses.append(np.full(len(snrs[-1]), np.nan))

        if "luminosity_distance" in extrinsic:
            distances.append(to_numpy(extrinsic["luminosity_distance"]))
        elif "luminosity_distance" in params:
            distances.append(to_numpy(params["luminosity_distance"]))
        else:
            distances.append(np.full(len(snrs[-1]), np.nan))

        n_collected += len(snrs[-1])
        if n_collected >= args.num_samples:
            break

    snrs = np.concatenate(snrs)[: args.num_samples]
    chirp_masses = np.concatenate(chirp_masses)[: args.num_samples]
    distances = np.concatenate(distances)[: args.num_samples]

    if np.all(np.isnan(chirp_masses)):
        print("\n[warn] could not find chirp_mass (or mass_1/mass_2) in the batch -- "
              "check the [debug] keys printed above and tell me the right key name.")
    if np.all(np.isnan(distances)):
        print("\n[warn] could not find luminosity_distance in the batch -- "
              "check the [debug] keys printed above.")

    valid_m = ~np.isnan(chirp_masses)
    valid_d = ~np.isnan(distances)

    if valid_m.any():
        r_m = np.corrcoef(np.log(snrs[valid_m]), np.log(chirp_masses[valid_m]))[0, 1]
        print(f"\nPearson corr(log SNR, log chirp_mass) = {r_m:.3f}")
    if valid_d.any():
        r_d = np.corrcoef(np.log(snrs[valid_d]), np.log(distances[valid_d]))[0, 1]
        print(f"Pearson corr(log SNR, log distance)    = {r_d:.3f}")
    print("(closer to -1 or +1 = that parameter strongly drives SNR; "
          "closer to 0 = weak relationship)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        pos = snrs[snrs > 0]
        bins = np.logspace(np.log10(max(pos.min(), 1e-3)), np.log10(pos.max()), 60)
        axes[0].hist(pos, bins=bins)
        axes[0].set_xscale("log")
        axes[0].set_xlabel("network optimal SNR (log scale)")
        axes[0].set_title("SNR distribution")

        if valid_m.any():
            axes[1].hexbin(chirp_masses[valid_m], snrs[valid_m], xscale="log", yscale="log", gridsize=40, mincnt=1)
            axes[1].set_xlabel("chirp mass (log scale)")
            axes[1].set_ylabel("SNR (log scale)")
            axes[1].set_title("SNR vs chirp mass")

        if valid_d.any():
            axes[2].hexbin(distances[valid_d], snrs[valid_d], xscale="log", yscale="log", gridsize=40, mincnt=1)
            axes[2].set_xlabel("luminosity distance (log scale)")
            axes[2].set_ylabel("SNR (log scale)")
            axes[2].set_title("SNR vs distance")

        plt.tight_layout()
        out_path = f"{args.out_prefix}_histogram.pdf"
        plt.savefig(out_path)
        print(f"\nSaved plot to {out_path}")
    except ImportError:
        print("matplotlib not available; skipping plot.")


if __name__ == "__main__":
    main()

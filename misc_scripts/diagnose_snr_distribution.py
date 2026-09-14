"""
Diagnostic script: measure the distribution of network optimal SNR in the
CURRENT training set (intrinsic dataset + on-the-fly extrinsic sampling, as
actually seen by the network during training), before deciding how to
rebalance it.

If SNR reweighting is enabled in train_settings (data.snr_reweighting.enabled:
true), this ALSO reports importance-weight diagnostics: effective sample size
(n_eff, same definition as Labrador Eq. 32-33) and basic weight statistics,
so you can check -- BEFORE launching any training -- whether the raw
importance weights produced by SampleSNRTargetLuminosityDistance are well
behaved, or whether a few extreme-weight examples would dominate the loss.
This does NOT require XGBoost: our weight is known exactly in closed form
(physical_prior(D) / proposal(D)), so there is nothing to regress -- see
project notes. If n_eff/N turns out too low, the script also scans simple
power-scaling (tempering) of the weight, w -> w**alpha, as a cheap
alternative to the paper's XGBoost/lambda(d) smoothing.

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
To compare, e.g., a flat-in-distance vs. a UniformComovingVolume physical
prior, just point --train_settings at two YAMLs that differ only in
data.extrinsic_prior.luminosity_distance and run this script on each.
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
    snr_reweighting_enabled = data_settings.get("snr_reweighting", {}).get("enabled", False)

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
    weights_list = []
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

        if "weight" in batch:
            weights_list.append(batch["weight"].cpu().numpy())

        n_collected += len(snr)
        if n_collected >= args.num_samples:
            break

    snrs = np.concatenate(snrs)[: args.num_samples]
    weights = (
        np.concatenate(weights_list)[: args.num_samples] if weights_list else None
    )
    # np.save(f"{args.out_prefix}_snrs.npy", snrs)  # disabled: not saving the raw array

    print(f"\nCollected {len(snrs)} samples of network optimal SNR.")
    print("Percentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{p:2d}: SNR = {np.percentile(snrs, p):8.2f}")
    print(f"\nFraction with SNR >  20: {np.mean(snrs > 20):.4f}")
    print(f"Fraction with SNR >  50: {np.mean(snrs > 50):.4f}")
    print(f"Fraction with SNR > 100: {np.mean(snrs > 100):.4f}")

    if weights is None:
        if snr_reweighting_enabled:
            print(
                "\n[warn] snr_reweighting.enabled=true in the settings file, but no "
                "'weight' key was found in the batch (AttachImportanceWeight did not "
                "run). Skipping importance-weight diagnostics."
            )
        else:
            print(
                "\nsnr_reweighting is disabled in this settings file -- no importance "
                "weight to analyze (every example counts equally in the loss)."
            )
    else:
        print("\n" + "=" * 70)
        print("IMPORTANCE WEIGHT DIAGNOSTICS: physical_prior(D) / proposal(D)")
        print("=" * 70)
        print(
            "No XGBoost involved: this weight is known exactly in closed form for "
            "every sample (see SampleSNRTargetLuminosityDistance), so there is "
            "nothing to regress. What we check here is only whether the weights "
            "are well-behaved enough to use directly in the loss."
        )

        # n_eff is scale-invariant, but normalize to mean=1 anyway to make the
        # percentiles below directly readable ("this example counts as if it
        # were X typical examples").
        w = weights / weights.mean()
        n_eff = (w.sum() ** 2) / (w ** 2).sum()
        print(
            f"\nn_eff / N = {n_eff / len(w):.5f}  "
            f"(n_eff = {n_eff:.1f} out of N = {len(w)} samples)"
        )
        print(
            "  Same effective-sample-size definition as Labrador Eq. 32-33.\n"
            "  1.0 = every example equally informative (no penalty).\n"
            "  Close to 0 = a handful of examples dominate the loss; the rest are"
            " nearly wasted."
        )

        print("\nWeight percentiles (normalized to mean = 1):")
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            print(f"  p{p:2d}: weight = {np.percentile(w, p):10.4g}")
        print(f"\nmax / min weight ratio: {w.max() / w.min():.4g}")

        order = np.argsort(w)[::-1]
        w_sorted = w[order]
        cumfrac = np.cumsum(w_sorted) / w_sorted.sum()
        print("\nHow concentrated is the total weight mass:")
        for frac in [0.01, 0.05, 0.10]:
            k = max(1, int(round(frac * len(w))))
            print(
                f"  top {frac * 100:4.0f}% of examples (by weight) carry "
                f"{cumfrac[k - 1] * 100:5.1f}% of the total weight mass"
            )

        print(
            "\nEffect of power-scaling (tempering) the weight, w -> w**alpha:\n"
            "  (cheap alternative to the paper's XGBoost/lambda(d) smoothing --\n"
            "   shrinks extreme weights directly, no regression needed. alpha=1.0\n"
            "   is the raw/unbiased weight; smaller alpha trades some bias for a\n"
            "   flatter, more stable weight distribution -- pick the smallest\n"
            "   alpha reduction that gets n_eff/N to an acceptable level.)"
        )
        for alpha in [1.0, 0.75, 0.5, 0.35, 0.25, 0.1]:
            w_temp = w ** alpha
            n_eff_temp = (w_temp.sum() ** 2) / (w_temp ** 2).sum()
            print(f"  alpha = {alpha:.2f}: n_eff/N = {n_eff_temp / len(w):.5f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_panels = 4 if weights is not None else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))

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

        if weights is not None:
            w = weights / weights.mean()
            wpos = w[w > 0]
            wbins = np.logspace(
                np.log10(max(wpos.min(), 1e-6)), np.log10(wpos.max()), 60
            )
            axes[2].hist(wpos, bins=wbins)
            axes[2].set_xscale("log")
            axes[2].set_yscale("log")
            axes[2].set_xlabel("importance weight (normalized to mean=1, log scale)")
            axes[2].set_title("weight distribution")

            alphas = np.linspace(0.05, 1.0, 40)
            n_eff_curve = []
            for a in alphas:
                wa = w ** a
                n_eff_curve.append((wa.sum() ** 2) / (wa ** 2).sum() / len(w))
            axes[3].plot(alphas, n_eff_curve)
            axes[3].set_xlabel("power-scaling alpha")
            axes[3].set_ylabel("n_eff / N")
            axes[3].set_title("effect of tempering w -> w^alpha")
            axes[3].axhline(1.0, ls="--", lw=0.5, color="gray")

        plt.tight_layout()
        out_path = f"{args.out_prefix}_histogram.pdf"
        plt.savefig(out_path)
        print(f"\nSaved histogram plot to {out_path}")
    except ImportError:
        print("matplotlib not available; skipping plot.")


if __name__ == "__main__":
    main()

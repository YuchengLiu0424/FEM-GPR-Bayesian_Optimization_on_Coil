"""Five-split ARD and predictive-calibration diagnostics for the 200-point GP."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split

import py1_generate_parameter_table as py1
from py3_train_gp import fit_gp_from_observations, predict_gp_physical


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT = SCRIPT_DIR / "py2_output" / "output.csv"
OUTPUT = SCRIPT_DIR / "py12_output"
SEEDS = (42, 43, 44, 45, 46)
TRAIN_SIZE = 200
VALIDATION_SIZE = 50


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def fit_split(data: pd.DataFrame, seed: int) -> tuple[dict, pd.DataFrame, list[str]]:
    feature_columns = list(data.columns[1:-1])
    target_column = data.columns[-1]
    train_pool, validation = train_test_split(
        data,
        test_size=VALIDATION_SIZE,
        random_state=seed,
        shuffle=True,
    )
    order = np.random.RandomState(seed).permutation(train_pool.index)
    train = (
        train_pool.loc[order[:TRAIN_SIZE]].sort_index().reset_index(drop=True)
    )
    validation = validation.sort_index().reset_index(drop=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fitted = fit_gp_from_observations(
            train[feature_columns].to_numpy(float),
            train[target_column].to_numpy(float),
            feature_columns=feature_columns,
            random_state=seed,
        )
    mean, std = predict_gp_physical(
        fitted, validation[feature_columns].to_numpy(float)
    )
    std = np.maximum(std, np.finfo(float).eps)
    true = validation[target_column].to_numpy(float)
    result = validation[[data.columns[0], *feature_columns]].copy()
    result["seed"] = seed
    result["target_true"] = true
    result["target_pred"] = mean
    result["target_std"] = std
    result["standardized_residual"] = (true - mean) / std
    result["absolute_error"] = np.abs(true - mean)
    result["nlpd"] = (
        0.5 * np.log(2.0 * np.pi * std**2)
        + 0.5 * result["standardized_residual"].to_numpy(float) ** 2
    )
    messages = [str(item.message) for item in caught]
    return fitted, result, messages


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configure_style()
    data = pd.read_csv(INPUT)
    py1.validate_fem_dataset_contract(data, INPUT.parent / "metadata.json")
    feature_columns = list(data.columns[1:-1])
    length_scale_rows = []
    calibration_rows = []
    predictions = []
    warning_rows = []

    for seed in SEEDS:
        fitted, result, messages = fit_split(data, seed)
        model = fitted["model"]
        matern = model.kernel_.k1.k2
        white = model.kernel_.k2
        row = {"seed": seed}
        row.update(
            {
                feature: float(scale)
                for feature, scale in zip(feature_columns, matern.length_scale)
            }
        )
        row["signal_variance"] = float(model.kernel_.k1.k1.constant_value)
        row["white_noise_level"] = float(white.noise_level)
        row["log_marginal_likelihood"] = float(
            model.log_marginal_likelihood_value_
        )
        row["target_scale_T"] = float(fitted["y_scaler"].scale_[0])
        length_scale_rows.append(row)
        z = result["standardized_residual"].to_numpy(float)
        target_scale = float(fitted["y_scaler"].scale_[0])
        calibration_rows.append(
            {
                "seed": seed,
                "standardized_residual_mean": float(np.mean(z)),
                "standardized_residual_sd": float(np.std(z, ddof=1)),
                "mean_nlpd_T": float(result["nlpd"].mean()),
                "mean_nlpd_standardized_target": float(
                    result["nlpd"].mean() - np.log(target_scale)
                ),
                "convergence_warning_count": len(messages),
            }
        )
        warning_rows.extend({"seed": seed, "message": message} for message in messages)
        predictions.append(result)

    scales = pd.DataFrame(length_scale_rows)
    calibration = pd.DataFrame(calibration_rows)
    all_predictions = pd.concat(predictions, ignore_index=True)
    scale_summary = pd.DataFrame(
        {
            "variable": feature_columns,
            "length_scale_mean": [scales[column].mean() for column in feature_columns],
            "length_scale_sd": [scales[column].std(ddof=1) for column in feature_columns],
        }
    ).sort_values("length_scale_mean")
    scale_summary["surrogate_sensitivity_rank"] = np.arange(
        1, len(scale_summary) + 1
    )

    scales.to_csv(OUTPUT / "ard_length_scales_by_split.csv", index=False)
    scale_summary.to_csv(OUTPUT / "ard_length_scale_summary.csv", index=False)
    calibration.to_csv(OUTPUT / "calibration_metrics_by_split.csv", index=False)
    all_predictions.to_csv(OUTPUT / "validation_predictions_all_splits.csv", index=False)
    pd.DataFrame(warning_rows).to_csv(OUTPUT / "convergence_warnings.csv", index=False)

    nominal = np.linspace(0.1, 0.99, 19)
    reliability_rows = []
    for seed, group in all_predictions.groupby("seed"):
        absolute_z = np.abs(group["standardized_residual"].to_numpy(float))
        for probability in nominal:
            threshold = stats.norm.ppf((1.0 + probability) / 2.0)
            reliability_rows.append(
                {
                    "seed": int(seed),
                    "nominal_coverage": probability,
                    "empirical_coverage": float(np.mean(absolute_z <= threshold)),
                }
            )
    reliability = pd.DataFrame(reliability_rows)
    reliability.to_csv(OUTPUT / "reliability_curve.csv", index=False)

    aggregate = reliability.groupby("nominal_coverage")["empirical_coverage"].agg(
        ["mean", "std"]
    )
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.plot([0, 1], [0, 1], "--", color="0.35", label="Ideal")
    ax.plot(aggregate.index, aggregate["mean"], marker="o", label="Five-split mean")
    ax.fill_between(
        aggregate.index,
        aggregate["mean"] - aggregate["std"],
        aggregate["mean"] + aggregate["std"],
        alpha=0.18,
        label=r"$\pm$ sample SD",
    )
    ax.set(xlabel="Nominal central coverage", ylabel="Empirical coverage", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "reliability_curve.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    for seed, group in all_predictions.groupby("seed"):
        ax.scatter(
            group["target_std"],
            group["absolute_error"],
            s=18,
            alpha=0.72,
            label=f"Seed {seed}",
        )
    limit = float(
        max(all_predictions["target_std"].max(), all_predictions["absolute_error"].max())
    )
    ax.plot([0, limit], [0, limit], "--", color="0.35", label=r"$|e|=\sigma$")
    ax.set(xlabel="Predictive standard deviation (T)", ylabel="Absolute error (T)")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "absolute_error_vs_predictive_std.png", dpi=300)
    plt.close(fig)

    z = np.sort(all_predictions["standardized_residual"].to_numpy(float))
    theoretical = stats.norm.ppf((np.arange(1, len(z) + 1) - 0.5) / len(z))
    fig, ax = plt.subplots(figsize=(4.3, 4.0))
    ax.scatter(theoretical, z, s=16, alpha=0.75)
    lo = float(min(theoretical.min(), z.min()))
    hi = float(max(theoretical.max(), z.max()))
    ax.plot([lo, hi], [lo, hi], "--", color="0.35")
    ax.set(xlabel="Standard-normal quantile", ylabel="Empirical residual quantile")
    fig.tight_layout()
    fig.savefig(OUTPUT / "standardized_residual_qq.png", dpi=300)
    plt.close(fig)

    metadata = {
        "input": str(INPUT),
        "seeds": list(SEEDS),
        "train_size": TRAIN_SIZE,
        "validation_size": VALIDATION_SIZE,
        "pooled_standardized_residual_mean": float(
            all_predictions["standardized_residual"].mean()
        ),
        "pooled_standardized_residual_sd": float(
            all_predictions["standardized_residual"].std(ddof=1)
        ),
        "mean_split_nlpd_T": float(calibration["mean_nlpd_T"].mean()),
        "mean_split_nlpd_standardized_target": float(
            calibration["mean_nlpd_standardized_target"].mean()
        ),
        "total_convergence_warnings": int(
            calibration["convergence_warning_count"].sum()
        ),
        "design_space_definition": py1.design_space_definition(),
        "design_space_fingerprint": py1.design_space_fingerprint(),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()

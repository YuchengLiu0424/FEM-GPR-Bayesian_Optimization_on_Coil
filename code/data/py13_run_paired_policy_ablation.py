"""Offline kernel screening and paired retrospective BO ablation.

This script never calls COMSOL.  Every acquisition is restricted to designs
whose FEM response already exists in the archived dataset.  It therefore
answers a retrospective question ("would the policy have selected better
known designs?"), not the prospective performance of continuous-domain BO.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf_discrete
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from scipy.stats import ttest_rel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from py6_optimization_common import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
    expected_improvement,
    fit_surrogate,
    load_fem_dataset,
    predict_surrogate,
)
from py7_run_bayesian_optimization import initial_and_validation


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "py13_output"
SEEDS = (42, 43, 44, 45, 46)
INITIAL_SIZE = 25
TOTAL_BUDGET = 75
CONTINUATION_BUDGET = TOTAL_BUDGET - INITIAL_SIZE
KERNELS = (
    "matern12_ls0p5",
    "matern32_ls0p5",
    "matern52_ls0p5",
    "matern52_ls2",
    "matern52_ls5",
    "matern52_free",
    "rbf_ls0p5",
    "rbf_ls2",
    "rbf_ls5",
    "rbf_free",
)
METHODS = (
    "random",
    "sklearn_mean",
    "sklearn_ei",
    "botorch_mean",
    "botorch_logei",
)
DTYPE = torch.double
EPS = 1e-15
FEM_NOISE_VARIANCE_T2 = 1e-14  # 0.1 µT numerical-noise standard deviation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel-maxiter",
        type=int,
        default=75,
        help="Maximum scipy iterations for each kernel-screen GP fit.",
    )
    parser.add_argument(
        "--policy-maxiter",
        type=int,
        default=50,
        help="Maximum scipy iterations for each sequential BoTorch GP fit.",
    )
    parser.add_argument(
        "--kernel-only",
        action="store_true",
        help="Run kernel screening but skip the paired policy ablation.",
    )
    parser.add_argument(
        "--policy-only",
        action="store_true",
        help="Reuse selected_kernel.json and skip kernel screening.",
    )
    return parser.parse_args()


def make_covar(kernel_name: str, dim: int) -> ScaleKernel:
    # Inputs are normalized to [0, 1].  Screen several prior scales because
    # the archived response is unusually smooth and a generic O(0.5) prior
    # can over-regularize it.
    family, scale_label = kernel_name.rsplit("_", maxsplit=1)
    if scale_label == "free":
        lengthscale_prior = None
    else:
        prior_mean = float(scale_label.removeprefix("ls").replace("p", "."))
        lengthscale_prior = GammaPrior(
            concentration=3.0,
            rate=3.0 / prior_mean,
        )
    if family == "rbf":
        base = RBFKernel(ard_num_dims=dim, lengthscale_prior=lengthscale_prior)
    else:
        nu = {
            "matern12": 0.5,
            "matern32": 1.5,
            "matern52": 2.5,
        }[family]
        base = MaternKernel(
            nu=nu,
            ard_num_dims=dim,
            lengthscale_prior=lengthscale_prior,
        )
    return ScaleKernel(base)


def fit_botorch_gp(
    x: np.ndarray,
    y: np.ndarray,
    kernel_name: str,
    maxiter: int,
    seed: int,
) -> SingleTaskGP:
    torch.manual_seed(seed)
    train_x = torch.tensor(np.array(x, dtype=float, copy=True), dtype=DTYPE)
    train_y = torch.tensor(
        np.array(y, dtype=float, copy=True),
        dtype=DTYPE,
    ).reshape(-1, 1)
    train_yvar = torch.full_like(train_y, FEM_NOISE_VARIANCE_T2)
    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        train_Yvar=train_yvar,
        covar_module=make_covar(kernel_name, train_x.shape[-1]),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(
        mll,
        optimizer_kwargs={"options": {"maxiter": maxiter}},
    )
    model.eval()
    model.likelihood.eval()
    return model


def botorch_predict(
    model: SingleTaskGP,
    x: np.ndarray,
    observation_noise: bool,
) -> tuple[np.ndarray, np.ndarray]:
    test_x = torch.as_tensor(x, dtype=DTYPE)
    with torch.no_grad():
        posterior = model.posterior(test_x, observation_noise=observation_noise)
        mean = posterior.mean.squeeze(-1).cpu().numpy()
        std = posterior.variance.clamp_min(EPS).sqrt().squeeze(-1).cpu().numpy()
    return mean, std


def prediction_metrics(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, float]:
    std = np.maximum(np.asarray(std, dtype=float), math.sqrt(EPS))
    residual = np.asarray(y_true, dtype=float) - np.asarray(mean, dtype=float)
    z = residual / std
    nlpd = 0.5 * np.log(2.0 * np.pi * std**2) + 0.5 * z**2
    return {
        "rmse_T": float(math.sqrt(mean_squared_error(y_true, mean))),
        "mae_T": float(mean_absolute_error(y_true, mean)),
        "r2": float(r2_score(y_true, mean)),
        "nlpd": float(np.mean(nlpd)),
        "z_mean": float(np.mean(z)),
        "z_sd": float(np.std(z, ddof=1)),
        "coverage_95": float(np.mean(np.abs(z) <= 1.959963984540054)),
    }


def run_kernel_screen(
    data: pd.DataFrame,
    maxiter: int,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    for split_seed in SEEDS:
        train, test = train_test_split(
            data,
            test_size=50,
            random_state=split_seed,
            shuffle=True,
        )
        x_train = train.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
        y_train = train.loc[:, TARGET_COLUMN].to_numpy(dtype=float)
        x_test = test.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
        y_test = test.loc[:, TARGET_COLUMN].to_numpy(dtype=float)

        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sklearn_model = fit_surrogate(train, seed=split_seed)
        sk_mean, sk_std = predict_surrogate(sklearn_model, x_test)
        row: dict[str, object] = {
            "split_seed": split_seed,
            "model": "sklearn_matern52_current",
            "fit_seconds": time.perf_counter() - start,
            "warning_count": len(caught),
        }
        row.update(prediction_metrics(y_test, sk_mean, sk_std))
        records.append(row)

        for kernel_name in KERNELS:
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = fit_botorch_gp(
                    x_train,
                    y_train,
                    kernel_name,
                    maxiter=maxiter,
                    seed=10_000 + split_seed,
                )
            # Match the existing sklearn diagnostic: the FEM response is
            # treated as a deterministic latent function, so inferred
            # observation noise is not added to held-out uncertainty.
            mean, std = botorch_predict(model, x_test, observation_noise=False)
            row = {
                "split_seed": split_seed,
                "model": f"botorch_{kernel_name}",
                "fit_seconds": time.perf_counter() - start,
                "warning_count": len(caught),
            }
            row.update(prediction_metrics(y_test, mean, std))
            records.append(row)

    by_split = pd.DataFrame(records)
    metric_columns = [
        "rmse_T",
        "mae_T",
        "r2",
        "nlpd",
        "z_mean",
        "z_sd",
        "coverage_95",
        "fit_seconds",
        "warning_count",
    ]
    summary = (
        by_split.groupby("model", sort=False)[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in col if part).rstrip("_")
        if isinstance(col, tuple)
        else col
        for col in summary.columns
    ]

    botorch_rows = summary[summary["model"].str.startswith("botorch_")].copy()
    botorch_rows = botorch_rows.sort_values(
        ["nlpd_mean", "rmse_T_mean"],
        ascending=[True, True],
    )
    selected_model = str(botorch_rows.iloc[0]["model"])
    selected_kernel = selected_model.removeprefix("botorch_")
    return selected_kernel, by_split, summary


def select_candidate(
    method: str,
    observed: pd.DataFrame,
    available: pd.DataFrame,
    selected_kernel: str,
    policy_maxiter: int,
    seed: int,
    iteration: int,
    rng: np.random.Generator,
) -> tuple[int, float, float, float]:
    if method == "random":
        position = int(rng.integers(0, len(available)))
        return position, float("nan"), float("nan"), float("nan")

    x_obs = observed.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    y_obs = observed.loc[:, TARGET_COLUMN].to_numpy(dtype=float)
    x_available = available.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)

    if method.startswith("sklearn_"):
        model = fit_surrogate(
            observed,
            seed=seed * 1_000 + iteration,
        )
        mean, std = predict_surrogate(model, x_available)
        if method == "sklearn_mean":
            acquisition = mean
        else:
            acquisition = expected_improvement(mean, std, float(np.max(y_obs)))
        position = int(np.argmax(acquisition))
        return (
            position,
            float(mean[position]),
            float(std[position]),
            float(acquisition[position]),
        )

    model = fit_botorch_gp(
        x_obs,
        y_obs,
        selected_kernel,
        maxiter=policy_maxiter,
        seed=seed * 1_000 + iteration,
    )
    if method == "botorch_mean":
        mean, std = botorch_predict(model, x_available, observation_noise=False)
        position = int(np.argmax(mean))
        return position, float(mean[position]), float(std[position]), float(mean[position])

    choices = torch.as_tensor(x_available, dtype=DTYPE)
    best_f = torch.tensor(float(np.max(y_obs)), dtype=DTYPE)
    acquisition_function = LogExpectedImprovement(model=model, best_f=best_f)
    candidate, acquisition_value = optimize_acqf_discrete(
        acq_function=acquisition_function,
        q=1,
        choices=choices,
        max_batch_size=2048,
        unique=True,
    )
    distance = np.max(np.abs(x_available - candidate.cpu().numpy()[0]), axis=1)
    position = int(np.argmin(distance))
    mean, std = botorch_predict(
        model,
        x_available[position : position + 1],
        observation_noise=False,
    )
    return (
        position,
        float(mean[0]),
        float(std[0]),
        float(acquisition_value.item()),
    )


def run_method(
    method: str,
    initial: pd.DataFrame,
    available: pd.DataFrame,
    selected_kernel: str,
    policy_maxiter: int,
    seed: int,
) -> pd.DataFrame:
    observed = initial.copy().reset_index(drop=True)
    remaining = available.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    best = -np.inf
    records: list[dict[str, object]] = []

    for call_index, row in observed.iterrows():
        response = float(row[TARGET_COLUMN])
        best = max(best, response)
        records.append(
            {
                "seed": seed,
                "method": method,
                "call": call_index + 1,
                "phase": "shared_initial",
                ID_COLUMN: str(row[ID_COLUMN]),
                TARGET_COLUMN: response,
                "best_so_far_T": best,
                "predicted_mean_T": float("nan"),
                "predicted_std_T": float("nan"),
                "acquisition_value": float("nan"),
            }
        )

    for iteration in range(CONTINUATION_BUDGET):
        position, pred_mean, pred_std, acquisition = select_candidate(
            method=method,
            observed=observed,
            available=remaining,
            selected_kernel=selected_kernel,
            policy_maxiter=policy_maxiter,
            seed=seed,
            iteration=iteration,
            rng=rng,
        )
        selected = remaining.iloc[position]
        response = float(selected[TARGET_COLUMN])
        best = max(best, response)
        records.append(
            {
                "seed": seed,
                "method": method,
                "call": INITIAL_SIZE + iteration + 1,
                "phase": "retrospective_selection",
                ID_COLUMN: str(selected[ID_COLUMN]),
                TARGET_COLUMN: response,
                "best_so_far_T": best,
                "predicted_mean_T": pred_mean,
                "predicted_std_T": pred_std,
                "acquisition_value": acquisition,
            }
        )
        observed = pd.concat(
            [observed, remaining.iloc[[position]]],
            ignore_index=True,
        )
        remaining = remaining.drop(index=position).reset_index(drop=True)
    return pd.DataFrame(records)


def run_policy_ablation(
    data: pd.DataFrame,
    selected_kernel: str,
    policy_maxiter: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    histories: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for seed in SEEDS:
        initial, _, available = initial_and_validation(
            data,
            seed=seed,
            initial_size=INITIAL_SIZE,
        )
        shared_ids = initial[ID_COLUMN].astype(str).tolist()
        for method in METHODS:
            start = time.perf_counter()
            history = run_method(
                method=method,
                initial=initial,
                available=available,
                selected_kernel=selected_kernel,
                policy_maxiter=policy_maxiter,
                seed=seed,
            )
            elapsed = time.perf_counter() - start
            continuation_ids = history.loc[
                history["phase"] == "retrospective_selection", ID_COLUMN
            ].astype(str)
            curve = history["best_so_far_T"].to_numpy(dtype=float)
            summaries.append(
                {
                    "seed": seed,
                    "method": method,
                    "initial_ids": json.dumps(shared_ids),
                    "initial_best_T": float(curve[INITIAL_SIZE - 1]),
                    "final_best_T": float(curve[-1]),
                    "improvement_T": float(curve[-1] - curve[INITIAL_SIZE - 1]),
                    "auc_best_T_calls": float(
                        np.trapezoid(curve, history["call"].to_numpy(dtype=float))
                    ),
                    "unique_continuation_ids": int(continuation_ids.nunique()),
                    "runtime_seconds": elapsed,
                }
            )
            histories.append(history)
            print(
                f"seed={seed} method={method} "
                f"final={curve[-1]:.9f} T runtime={elapsed:.1f}s",
                flush=True,
            )

    history_all = pd.concat(histories, ignore_index=True)
    by_seed = pd.DataFrame(summaries)
    numeric = [
        "initial_best_T",
        "final_best_T",
        "improvement_T",
        "auc_best_T_calls",
        "runtime_seconds",
    ]
    aggregate = by_seed.groupby("method", sort=False)[numeric].agg(["mean", "std"])
    aggregate.columns = ["_".join(col) for col in aggregate.columns]
    aggregate = aggregate.reset_index()

    pivot_endpoint = by_seed.pivot(index="seed", columns="method", values="final_best_T")
    pivot_auc = by_seed.pivot(
        index="seed",
        columns="method",
        values="auc_best_T_calls",
    )
    comparisons: list[dict[str, object]] = []
    reference = "sklearn_ei"
    for method in METHODS:
        if method == reference:
            continue
        endpoint_delta = pivot_endpoint[method] - pivot_endpoint[reference]
        auc_delta = pivot_auc[method] - pivot_auc[reference]
        endpoint_test = ttest_rel(pivot_endpoint[method], pivot_endpoint[reference])
        auc_test = ttest_rel(pivot_auc[method], pivot_auc[reference])
        comparisons.append(
            {
                "method": method,
                "reference": reference,
                "mean_final_delta_T": float(endpoint_delta.mean()),
                "wins_final": int((endpoint_delta > 0).sum()),
                "ties_final": int(np.isclose(endpoint_delta, 0.0, atol=1e-15).sum()),
                "paired_p_final": float(endpoint_test.pvalue),
                "mean_auc_delta_T_calls": float(auc_delta.mean()),
                "wins_auc": int((auc_delta > 0).sum()),
                "paired_p_auc": float(auc_test.pvalue),
            }
        )
    return history_all, by_seed, aggregate, pd.DataFrame(comparisons)


def save_plots(
    kernel_summary: pd.DataFrame | None,
    history: pd.DataFrame | None,
) -> None:
    if kernel_summary is not None:
        ordered = kernel_summary.sort_values("nlpd_mean")
        labels = ordered["model"].str.replace("botorch_", "BT ", regex=False)
        labels = labels.str.replace("sklearn_matern52_current", "sklearn M52", regex=False)
        positions = np.arange(len(ordered))
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 6.2), sharey=True)
        axes[0].barh(
            positions,
            ordered["rmse_T_mean"] * 1e6,
            xerr=ordered["rmse_T_std"] * 1e6,
            capsize=3,
        )
        axes[0].set_xlabel("Held-out RMSE (µT; lower is better)")
        axes[0].set_yticks(positions, labels)
        axes[1].barh(
            positions,
            ordered["nlpd_mean"],
            xerr=ordered["nlpd_std"],
            capsize=3,
        )
        axes[1].set_xlabel("Held-out NLPD (lower is better)")
        for axis in axes:
            axis.grid(axis="x", alpha=0.25)
        axes[0].invert_yaxis()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "kernel_screening.png", dpi=220)
        plt.close(fig)

    if history is None:
        return

    fig, axis = plt.subplots(figsize=(8.8, 5.2))
    for method in METHODS:
        block = history[history["method"] == method]
        curves = block.pivot(index="call", columns="seed", values="best_so_far_T")
        mean = curves.mean(axis=1).to_numpy(dtype=float) * 1e3
        std = curves.std(axis=1, ddof=1).to_numpy(dtype=float) * 1e3
        calls = curves.index.to_numpy(dtype=int)
        axis.plot(calls, mean, linewidth=1.7, label=method)
        axis.fill_between(calls, mean - std, mean + std, alpha=0.12)
    axis.axvline(INITIAL_SIZE, color="0.35", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Cumulative known FEM evaluations")
    axis.set_ylabel("Best known response (mT)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "paired_policy_convergence.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.kernel_only and args.policy_only:
        raise ValueError("--kernel-only and --policy-only are mutually exclusive")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_fem_dataset()

    kernel_by_split: pd.DataFrame | None = None
    kernel_summary: pd.DataFrame | None = None
    selection_path = OUTPUT_DIR / "selected_kernel.json"
    if args.policy_only:
        selected_kernel = json.loads(selection_path.read_text(encoding="utf-8"))[
            "selected_kernel"
        ]
    else:
        selected_kernel, kernel_by_split, kernel_summary = run_kernel_screen(
            data,
            maxiter=args.kernel_maxiter,
        )
        kernel_by_split.to_csv(
            OUTPUT_DIR / "kernel_screening_by_split.csv",
            index=False,
        )
        kernel_summary.to_csv(
            OUTPUT_DIR / "kernel_screening_summary.csv",
            index=False,
        )
        selection_path.write_text(
            json.dumps(
                {
                    "selected_kernel": selected_kernel,
                    "selection_rule": (
                        "Lowest mean held-out NLPD across five fixed 200/50 "
                        "splits; RMSE is the tie-breaker."
                    ),
                    "candidate_kernels": list(KERNELS),
                    "input_scaling": "[0, 1] bounds from the archived design space",
                    "outcome_transform": "BoTorch Standardize(m=1)",
                    "lengthscale_prior_screen": (
                        "Gamma(concentration=3, mean in {0.5, 2, 5}) "
                        "plus an unregularized MLE candidate"
                    ),
                    "fixed_fem_noise_variance_T2": FEM_NOISE_VARIANCE_T2,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"selected kernel: {selected_kernel}", flush=True)

    if args.kernel_only:
        save_plots(kernel_summary, None)
        return

    history, by_seed, aggregate, comparisons = run_policy_ablation(
        data,
        selected_kernel=selected_kernel,
        policy_maxiter=args.policy_maxiter,
    )
    history.to_csv(OUTPUT_DIR / "paired_history.csv", index=False)
    by_seed.to_csv(OUTPUT_DIR / "paired_summary_by_seed.csv", index=False)
    aggregate.to_csv(OUTPUT_DIR / "paired_summary_aggregate.csv", index=False)
    comparisons.to_csv(OUTPUT_DIR / "paired_vs_sklearn_ei.csv", index=False)
    metadata = {
        "offline_only": True,
        "calls_comsol": False,
        "finite_candidate_pool": True,
        "interpretation_limit": (
            "Results compare policies only on the archived labeled FEM pool "
            "and do not estimate continuous-domain prospective performance."
        ),
        "seeds": list(SEEDS),
        "initial_size": INITIAL_SIZE,
        "continuation_budget": CONTINUATION_BUDGET,
        "total_budget": TOTAL_BUDGET,
        "selected_kernel": selected_kernel,
        "methods": list(METHODS),
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    save_plots(kernel_summary, history)
    print(aggregate.to_string(index=False), flush=True)
    print(comparisons.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

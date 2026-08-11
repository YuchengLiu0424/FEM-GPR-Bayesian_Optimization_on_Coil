from __future__ import annotations

import json
import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import py1_generate_parameter_table as py1

SCRIPT_DIR = Path(__file__).resolve().parent
PY3_DIR = SCRIPT_DIR / 'py3_output'
PY4_DIR = SCRIPT_DIR / 'py4_output'
PY4_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CANDIDATES = [
    PY3_DIR / 'gp_model.joblib',
]
VALIDATION_CANDIDATES = [
    PY3_DIR / 'validation_data.csv',
    PY3_DIR / 'val_data.csv',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Validate the GP model and py3 validation split.')
    parser.add_argument('--skip-plots', action='store_true',
                        help='Skip image files while retaining output.csv, metrics.json, and report.')
    return parser.parse_args()


def find_existing(paths: list[Path], kind: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        f'No {kind} file found. Expected one of: ' + ', '.join(p.name for p in paths)
    )


def save_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    # Scatter with ±1σ uncertainty
    ax.errorbar(
        y_true,
        y_pred,
        yerr=y_std,
        fmt='o',
        markersize=5,
        capsize=2,
        elinewidth=1,
        alpha=0.8,
        label=r'Prediction ($\pm1\sigma$)',
    )

    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))

    # Ideal parity line
    ax.plot(
        [lo, hi],
        [lo, hi],
        '--',
        linewidth=2,
        label='Ideal prediction',
    )

    ax.set_xlabel("True target")
    ax.set_ylabel("Predicted target")
    ax.set_title(title)

    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def save_coverage_plot(
    within_1sigma: float,
    within_2sigma: float,
    path: Path,
) -> None:

    expected = [0.6827, 0.9545]
    observed = [within_1sigma, within_2sigma]

    labels = [r'$\pm1\sigma$', r'$\pm2\sigma$']

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5,4))

    ax.bar(
        x - width/2,
        expected,
        width,
        label='Expected'
    )

    ax.bar(
        x + width/2,
        observed,
        width,
        label='Observed'
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylim(0,1.05)

    ax.set_ylabel("Coverage Probability")
    ax.set_title("Uncertainty Calibration")

    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def save_standardized_residual_plot(
    z: np.ndarray,
    path: Path,
) -> None:

    fig, ax = plt.subplots(figsize=(5,4))

    ax.hist(
        z,
        bins=15,
        density=True,
        alpha=0.75,
    )

    ax.axvline(
        0,
        linestyle='--',
        linewidth=2,
    )

    ax.set_xlabel("Standardized residual")
    ax.set_ylabel("Density")
    ax.set_title("Standardized Residual Distribution")

    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def save_calibration_report(
    rmse,
    mae,
    r2,
    within_1sigma,
    within_2sigma,
    standardized_residual,
    path,
):

    with open(path, "w", encoding="utf-8") as f:

        f.write("Gaussian Process Validation Report\n")
        f.write("="*40 + "\n\n")

        f.write(f"RMSE : {rmse:.6f}\n")
        f.write(f"MAE  : {mae:.6f}\n")
        f.write(f"R²   : {r2:.6f}\n\n")

        f.write("Coverage\n")
        f.write("----------------------\n")

        f.write(
            f"Within ±1σ : {within_1sigma:.2%}\n"
        )

        f.write(
            f"Within ±2σ : {within_2sigma:.2%}\n\n"
        )

        f.write("Standardized Residual\n")
        f.write("----------------------\n")

        f.write(
            f"Mean : {np.mean(standardized_residual):.4f}\n"
        )

        f.write(
            f"Std  : {np.std(standardized_residual):.4f}\n"
        )

def main() -> None:
    args = parse_args()
    model_path = find_existing(MODEL_CANDIDATES, 'model')
    validation_path = find_existing(VALIDATION_CANDIDATES, 'validation dataset')

    bundle = joblib.load(model_path)
    if bundle.get('design_space_fingerprint') != py1.design_space_fingerprint():
        raise ValueError(
            'Refusing a GP model fitted under an obsolete design-space contract. '
            'Regenerate py1 through py3 first.'
        )
    model = bundle['model']
    x_scaler = bundle['x_scaler']
    y_scaler = bundle['y_scaler']
    feature_cols = bundle['feature_cols']
    target_col = bundle['target_col']
    id_col = bundle.get('id_col', 'sample_id')

    df = pd.read_csv(validation_path)
    missing = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing:
        raise ValueError(f'Validation file is missing columns: {missing}')

    X = df[feature_cols].to_numpy(dtype=float)
    y_true = df[target_col].to_numpy(dtype=float)
    Xs = x_scaler.transform(X)
    pred_s, std_s = model.predict(Xs, return_std=True)
    y_pred = y_scaler.inverse_transform(pred_s.reshape(-1, 1)).ravel()
    y_std = std_s * y_scaler.scale_[0]

    residual = y_true - y_pred

    # Standardized residual
    eps = 1e-12
    standardized_residual = residual / np.maximum(y_std, eps)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    within_1sigma = np.mean(np.abs(y_true - y_pred) <= y_std)
    within_2sigma = np.mean(np.abs(y_true - y_pred) <= 2 * y_std)

    out_df = df[[id_col] + feature_cols].copy() if id_col in df.columns else df[feature_cols].copy()
    out_df['target_true'] = y_true
    out_df['target_pred'] = y_pred
    out_df['target_std'] = y_std
    out_df['residual'] = residual
    out_df['abs_residual'] = np.abs(residual)
    out_df['standardized_residual'] = standardized_residual
    out_df.to_csv(PY4_DIR / 'output.csv', index=False)

    metrics = {
    'rmse': rmse,
    'mae': mae,
    'r2': r2,
    'within_1sigma': within_1sigma,
    'within_2sigma': within_2sigma,
    'standardized_residual_mean': float(np.mean(standardized_residual)),
    'standardized_residual_std': float(np.std(standardized_residual)),
    'model_path': str(model_path),
    'validation_path': str(validation_path),
    'n_samples': int(len(df)),
    'feature_columns': feature_cols,
    'target_column': target_col,
    'design_space_definition': py1.design_space_definition(),
    'design_space_fingerprint': py1.design_space_fingerprint(),
    }
    with open(PY4_DIR / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    if not args.skip_plots:
        save_plot(y_true, y_pred, y_std, 'Parity plot (held-out validation)', PY4_DIR / 'parity_plot.png')
        save_coverage_plot(within_1sigma, within_2sigma, PY4_DIR / "coverage_plot.png")
        save_standardized_residual_plot(standardized_residual, PY4_DIR / "standardized_residual.png")

    save_calibration_report(rmse, mae, r2, within_1sigma, within_2sigma, standardized_residual, PY4_DIR / "calibration_report.txt")

    print(f'Model: {model_path}')
    print(f'Validation data: {validation_path}')
    print(f'Saved validation output to: {PY4_DIR / "output.csv"}')
    print(f'R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}')


if __name__ == '__main__':
    main()

from __future__ import annotations

import json
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import py1_generate_parameter_table as py1


SCRIPT_DIR = Path(__file__).resolve().parent
PY2_DIR = SCRIPT_DIR / 'py2_output'
PY3_DIR = SCRIPT_DIR / 'py3_output'
PY3_DIR.mkdir(parents=True, exist_ok=True)

FEM_INPUT_FILE = PY2_DIR / 'output.csv'

TEST_SIZE = 0.2
VALIDATION_SIZE = 50
RANDOM_STATE = 42
N_RESTARTS = 8


@dataclass
class SplitInfo:
    input_file: str
    n_samples: int
    n_features: int
    feature_columns: list[str]
    target_column: str
    train_size: int
    validation_size: int
    test_size: float
    random_state: int
    kernel_init: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train the unchanged GP surrogate with a chosen training-set size.'
    )
    parser.add_argument('--train-size', type=int, default=None,
                        help='Training samples drawn from the fixed 200-sample training pool.')
    parser.add_argument('--seed', type=int, default=RANDOM_STATE,
                        help=f'Random seed for the fixed validation split and nested training subsets (default: {RANDOM_STATE}).')
    parser.add_argument('--skip-plots', action='store_true',
                        help='Skip parity plots while retaining model, data, and metrics outputs.')
    return parser.parse_args()


def find_input_file() -> Path:
    if FEM_INPUT_FILE.exists():
        return FEM_INPUT_FILE
    raise FileNotFoundError(
        f'Missing finalized FEM dataset: {FEM_INPUT_FILE}. '
        'Smoke-test and placeholder outputs are not valid training inputs.'
    )


def build_model(random_state: int = RANDOM_STATE) -> GaussianProcessRegressor:
    # Good default for smooth-but-not-too-smooth physical responses.
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(7), length_scale_bounds=(1e-3, 1e3), nu=2.5)
        + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-1))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=False,
        n_restarts_optimizer=N_RESTARTS,
        random_state=random_state,
        alpha=0.0,
    )


def fit_gp_from_observations(
    X_observed: np.ndarray,
    y_observed: np.ndarray,
    *,
    feature_columns: list[str] | None = None,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Fit the unchanged py3 GP to arbitrary observations entirely in memory."""
    X = np.asarray(X_observed, dtype=float)
    y = np.asarray(y_observed, dtype=float).reshape(-1, 1)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError('X_observed must be a non-empty two-dimensional array.')
    if y.shape[0] != X.shape[0]:
        raise ValueError('X_observed and y_observed must contain the same number of rows.')
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError('Observed inputs and targets must be finite.')
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y).ravel()
    model = build_model(random_state=random_state)
    model.fit(X_scaled, y_scaled)
    return {
        'model': model,
        'x_scaler': x_scaler,
        'y_scaler': y_scaler,
        'feature_cols': feature_columns or [f'u_{index}' for index in range(X.shape[1])],
    }


def predict_gp_physical(
    bundle: dict,
    X_candidates: np.ndarray,
    *,
    uncertainty: str = 'observation',
) -> tuple[np.ndarray, np.ndarray]:
    """Predict in physical units with observation or latent uncertainty.

    ``observation`` reproduces scikit-learn's full fitted-kernel variance and
    therefore includes the test-point contribution of ``WhiteKernel``.
    ``latent`` retains the fitted nugget in the training covariance but removes
    its test-point contribution, as required by the deterministic-EI workflow.
    """
    X = np.asarray(X_candidates, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError('X_candidates must be a finite one- or two-dimensional array.')
    X_scaled = bundle['x_scaler'].transform(X)
    mean_scaled, std_scaled = bundle['model'].predict(X_scaled, return_std=True)
    if uncertainty == 'latent':
        model = bundle['model']
        if not isinstance(model.kernel_.k2, WhiteKernel):
            raise TypeError('Latent-variance extraction expects signal_kernel + WhiteKernel.')
        signal_kernel = model.kernel_.k1
        cross_covariance = signal_kernel(X_scaled, model.X_train_)
        projected = np.linalg.solve(model.L_, cross_covariance.T)
        latent_variance = signal_kernel.diag(X_scaled) - np.einsum(
            'ij,ij->j', projected, projected
        )
        std_scaled = np.sqrt(np.maximum(latent_variance, 0.0))
    elif uncertainty != 'observation':
        raise ValueError("uncertainty must be 'observation' or 'latent'.")
    mean = bundle['y_scaler'].inverse_transform(mean_scaled.reshape(-1, 1)).ravel()
    std = np.asarray(std_scaled, dtype=float) * float(bundle['y_scaler'].scale_[0])
    return mean, std


def read_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[1] < 3:
        raise ValueError('Need at least one id column, several features, and one target column.')
    if path.resolve() != FEM_INPUT_FILE.resolve():
        raise ValueError(f'Refusing noncanonical GP training input: {path}')
    py1.validate_fem_dataset_contract(df, PY2_DIR / 'metadata.json')
    return df


def save_plot(y_true: np.ndarray, y_pred: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=26, alpha=0.85)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], linestyle='--')
    ax.set_xlabel('True target')
    ax.set_ylabel('Predicted target')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = find_input_file()
    df = read_data(input_path)

    id_col = df.columns[0]
    feature_cols = list(df.columns[1:-1])
    target_col = df.columns[-1]

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)

    if len(df) <= VALIDATION_SIZE:
        raise ValueError(
            f'Need more than {VALIDATION_SIZE} samples to keep the validation set fixed; got {len(df)}.'
        )

    # Hold out the same 50 samples at every learning-curve point. This is the
    # experimental validation set and never contributes to fitting or scaling.
    train_pool, val_df = train_test_split(
        df,
        test_size=VALIDATION_SIZE,
        random_state=args.seed,
        shuffle=True,
    )
    train_size = len(train_pool) if args.train_size is None else args.train_size
    if not 1 <= train_size <= len(train_pool):
        raise ValueError(
            f'--train-size must be between 1 and {len(train_pool)}; got {train_size}.'
        )

    # A fixed permutation makes the training sets nested: the 25-point set is
    # contained in the 50-point set, and so on through the 200-point set.
    rng = np.random.RandomState(args.seed)
    train_indices = rng.permutation(train_pool.index)[:train_size]
    train_df = train_pool.loc[train_indices].sort_index().reset_index(drop=True)
    val_df = val_df.sort_index().reset_index(drop=True)

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[target_col].to_numpy(dtype=float).reshape(-1, 1)
    X_val = val_df[feature_cols].to_numpy(dtype=float)
    y_val = val_df[target_col].to_numpy(dtype=float).reshape(-1, 1)

    fitted = fit_gp_from_observations(X_train, y_train.ravel(), feature_columns=feature_cols, random_state=args.seed)
    model = fitted['model']
    x_scaler = fitted['x_scaler']
    y_scaler = fitted['y_scaler']
    train_pred, train_std = predict_gp_physical(fitted, X_train)
    val_pred, val_std = predict_gp_physical(fitted, X_val)

    train_true = y_train.ravel()
    val_true = y_val.ravel()

    train_metrics = {
        'rmse': float(np.sqrt(mean_squared_error(train_true, train_pred))),
        'mae': float(mean_absolute_error(train_true, train_pred)),
        'r2': float(r2_score(train_true, train_pred)),
    }
    val_metrics = {
        'rmse': float(np.sqrt(mean_squared_error(val_true, val_pred))),
        'mae': float(mean_absolute_error(val_true, val_pred)),
        'r2': float(r2_score(val_true, val_pred)),
    }

    train_out = train_df[[id_col] + feature_cols].copy()
    train_out['target_true'] = train_true
    train_out['target_pred'] = train_pred
    train_out['target_std'] = train_std
    train_out['residual'] = train_true - train_pred
    train_out['abs_residual'] = np.abs(train_true - train_pred)
    train_out['split'] = 'train'

    val_out = val_df[[id_col] + feature_cols].copy()
    val_out['target_true'] = val_true
    val_out['target_pred'] = val_pred
    val_out['target_std'] = val_std
    val_out['residual'] = val_true - val_pred
    val_out['abs_residual'] = np.abs(val_true - val_pred)
    val_out['split'] = 'validation'

    all_predictions = pd.concat([train_out, val_out], ignore_index=True)
    all_predictions.to_csv(PY3_DIR / 'predictions.csv', index=False)
    train_out.to_csv(PY3_DIR / 'train_predictions.csv', index=False)
    val_out.to_csv(PY3_DIR / 'validation_predictions.csv', index=False)

    # Save split data for py4.
    train_df.to_csv(PY3_DIR / 'train_data.csv', index=False)
    val_df.to_csv(PY3_DIR / 'validation_data.csv', index=False)

    bundle = {
        'model': model,
        'x_scaler': x_scaler,
        'y_scaler': y_scaler,
        'feature_cols': feature_cols,
        'target_col': target_col,
        'id_col': id_col,
        'input_file': str(input_path),
        'design_space_definition': py1.design_space_definition(),
        'design_space_fingerprint': py1.design_space_fingerprint(),
        'split_info': {
            'train_rows': int(train_df.shape[0]),
            'validation_rows': int(val_df.shape[0]),
        },
    }
    joblib.dump(bundle, PY3_DIR / 'gp_model.joblib')

    run_info = SplitInfo(
        input_file=str(input_path),
        n_samples=int(df.shape[0]),
        n_features=int(len(feature_cols)),
        feature_columns=feature_cols,
        target_column=str(target_col),
        train_size=int(train_df.shape[0]),
        validation_size=int(val_df.shape[0]),
        test_size=float(len(val_df) / len(df)),
        random_state=args.seed,
        kernel_init='ConstantKernel * Matern(nu=2.5) + WhiteKernel',
    )
    metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'kernel_optimized': str(model.kernel_),
        'run_info': asdict(run_info),
        'design_space_definition': py1.design_space_definition(),
        'design_space_fingerprint': py1.design_space_fingerprint(),
    }
    with open(PY3_DIR / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    if not args.skip_plots:
        save_plot(train_true, train_pred, 'Parity plot (train)', PY3_DIR / 'parity_train.png')
        save_plot(val_true, val_pred, 'Parity plot (validation)', PY3_DIR / 'parity_validation.png')

    print(f'Input file: {input_path}')
    print(f'Saved model to: {PY3_DIR / "gp_model.joblib"}')
    print(f'Saved predictions to: {PY3_DIR / "predictions.csv"}')
    print(f'Train size={len(train_df)}, Validation size={len(val_df)}')
    print(f'Train R2={train_metrics["r2"]:.4f}, Val R2={val_metrics["r2"]:.4f}')
    print(f'Train RMSE={train_metrics["rmse"]:.4f}, Val RMSE={val_metrics["rmse"]:.4f}')


if __name__ == '__main__':
    main()

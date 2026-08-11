"""Run the GP sample-complexity study without modifying the fixed FEM dataset."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import py1_generate_parameter_table as py1
SCRIPT_DIR = Path(__file__).resolve().parent
PY2_DIR = SCRIPT_DIR / 'py2_output'
PY3_DIR = SCRIPT_DIR / 'py3_output'
PY4_DIR = SCRIPT_DIR / 'py4_output'
PY5_DIR = SCRIPT_DIR / 'py5_output'
PY3_SCRIPT = SCRIPT_DIR / 'py3_train_gp.py'
PY4_SCRIPT = SCRIPT_DIR / 'py4_validate_gp.py'
TRAIN_SIZES = (25, 50, 75, 100, 125, 150, 175, 200)
PARITY_SIZES = {50, 100, 200}
REPEATS = (("5.1", 42), ("5.2", 43), ("5.3", 44), ("5.4", 45), ("5.5", 46))

def read_json(path: Path) -> dict:
    with path.open(encoding='utf-8') as file:
        return json.load(file)

def run_script(script: Path, arguments: list[str]) -> float:
    start = time.perf_counter()
    subprocess.run([sys.executable, str(script), *arguments], check=True, cwd=SCRIPT_DIR)
    return time.perf_counter() - start

def configure_plot_style() -> None:
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
                         'legend.fontsize': 8, 'lines.linewidth': 1.6, 'lines.markersize': 5,
                         'axes.grid': True, 'grid.alpha': 0.28})

def save_learning_curve(results: pd.DataFrame, metric: str, ylabel: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(results['train_size'], results[metric], marker='o', color='#1f4e79')
    ax.set_xlabel('Training sample size')
    ax.set_ylabel(ylabel)
    ax.set_xticks(results['train_size'])
    ax.margins(x=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def run_repeat(label: str, seed: int, generation_seconds: float) -> None:
    output_dir = PY5_DIR / label
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int]] = []
    for train_size in TRAIN_SIZES:
        save_plots = train_size in PARITY_SIZES
        print(f'\n=== Repeat {label}, seed {seed}, training size {train_size} ===')
        train_args = ['--train-size', str(train_size), '--seed', str(seed)]
        validate_args: list[str] = []
        if not save_plots:
            train_args.append('--skip-plots')
            validate_args.append('--skip-plots')
        training_seconds = run_script(PY3_SCRIPT, train_args)
        validation_seconds = run_script(PY4_SCRIPT, validate_args)
        train_metrics = read_json(PY3_DIR / 'metrics.json')
        validation_metrics = read_json(PY4_DIR / 'metrics.json')
        current_fingerprint = py1.design_space_fingerprint()
        if train_metrics.get('design_space_fingerprint') != current_fingerprint:
            raise ValueError('py3 metrics use an obsolete design-space contract.')
        if validation_metrics.get('design_space_fingerprint') != current_fingerprint:
            raise ValueError('py4 metrics use an obsolete design-space contract.')
        run_info = train_metrics['run_info']
        rows.append({'train_size': int(run_info['train_size']), 'validation_size': int(run_info['validation_size']),
                     'train_r2': float(train_metrics['train']['r2']), 'validation_r2': float(validation_metrics['r2']),
                     'rmse': float(validation_metrics['rmse']), 'mae': float(validation_metrics['mae']),
                     'coverage_1sigma': float(validation_metrics['within_1sigma']),
                     'coverage_2sigma': float(validation_metrics['within_2sigma']),
                     'generation_time_seconds': generation_seconds, 'training_time_seconds': training_seconds,
                     'validation_time_seconds': validation_seconds})
        if save_plots:
            shutil.copy2(PY4_DIR / 'parity_plot.png', output_dir / f'parity_train{train_size}.png')
    results = pd.DataFrame(rows).sort_values('train_size').reset_index(drop=True)
    results.to_csv(output_dir / 'learning_curve_results.csv', index=False)
    save_learning_curve(results, 'validation_r2', 'Validation $R^2$', output_dir / 'learning_curve_r2.png')
    save_learning_curve(results, 'rmse', 'Validation RMSE (T)', output_dir / 'learning_curve_rmse.png')
    save_learning_curve(results, 'mae', 'Validation MAE (T)', output_dir / 'learning_curve_mae.png')
    save_learning_curve(results, 'coverage_1sigma', r'Coverage within $\pm 1\sigma$', output_dir / 'learning_curve_cov1.png')
    save_learning_curve(results, 'coverage_2sigma', r'Coverage within $\pm 2\sigma$', output_dir / 'learning_curve_cov2.png')
    with (output_dir / 'run_metadata.json').open('w', encoding='utf-8') as file:
        json.dump({'repeat_label': label, 'seed': seed, 'validation_size': 50,
                   'train_sizes': list(TRAIN_SIZES), 'generation_time_seconds': generation_seconds,
                   'design_space_definition': py1.design_space_definition(),
                   'design_space_fingerprint': py1.design_space_fingerprint()}, file, indent=2)
    print(f'\nSaved repeat {label} to: {output_dir}')

def main() -> None:
    PY5_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    fem_metadata = read_json(PY2_DIR / 'metadata.json')
    if fem_metadata.get('design_space_fingerprint') != py1.design_space_fingerprint():
        raise ValueError('The FEM dataset metadata uses an obsolete design-space contract.')
    generation_seconds = float(fem_metadata['generation_time_seconds'])
    for label, seed in REPEATS:
        run_repeat(label, seed, generation_seconds)

if __name__ == '__main__':
    main()

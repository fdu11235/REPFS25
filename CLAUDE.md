# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Financial market prediction system using PyTorch neural networks. The pipeline: preprocess market data with TA-Lib indicators, train a classifier (BUY/HOLD/SELL), generate predictions, and backtest with vectorbt. Experiment tracking via MLflow.

## Environment Setup

**Docker (recommended):**
```bash
docker-compose build
docker-compose run trading_app
conda activate trading-env
```

**Local:**
```bash
conda env create -f environment.yml
conda activate trading-env
```

Key dependencies: Python 3.10, PyTorch 2.1.0+cu118, vectorbt, TA-Lib (built from source), SHAP, MLflow.

## Running Scripts

All scripts run from the project root. The `src/` directory is added to the Python path via `.pylintrc` init-hook.

```bash
# Full experiment pipeline (preprocess → train → predict → backtest)
python src/run_experiment.py

# Individual steps
python src/run_preprocess_dataset.py
python src/run_torch_training.py
python src/run_predict_asset.py
python src/run_backtest.py
python src/run_shap_explainer.py
```

MLflow requires a running server at `http://localhost:5000` (hardcoded in `src/trainer/trainer.py`).

## Linting

```bash
pylint src/
```

## Architecture

### Configuration (`src/config/config.py`)
Single `RUN` dictionary holds all parameters: labeling thresholds (alpha/beta), backtest date range, balancing algorithm, epochs, PCA settings. Passed through the pipeline as a dict.

### Data Pipeline (`src/libs/`)
- `technical_analysis_lib.py` — ~30 TA-Lib indicators (RSI, MACD, Bollinger, Z-score, momentum, volatility)
- `compute_indicators_labels_lib.py` — Loads CSVs from `market_data/`, computes indicators, generates 3-class labels (0=BUY, 1=HOLD, 2=SELL) based on alpha/beta price change thresholds with backward/forward windows. Outputs to `processed_market_data/` and `asset_training/`
- `imbalanced_lib.py` — Class balancing via SRS or NCR (Neighborhood Cleaning Rule)

### Model (`src/model/`)
- `Pytorch_NNModel.py` — 4-layer FC network: input → 128 → 64 → 32 → 3 classes, LeakyReLU activation, early stopping, LR scheduling
- `CustomDataset.py` — PyTorch Dataset wrapper for pandas DataFrames

### Trainer (`src/trainer/trainer.py`)
Orchestrates the training loop: loads data via pluggable `get_data_fn`, splits 70/30 (train+val/test) then 80/20 (train/val), applies StandardScaler, optional PCA, class-weighted CrossEntropyLoss, Adam optimizer. Logs to MLflow and registers model as "FinancialNNModel" with Production stage.

### Experiment Strategies (`src/run_experiment.py`)
Select strategy by uncommenting in `__main__`:
- `classic_backtest()` — Single model on all assets
- `classic_backtest_per_asset()` — Individual model per asset
- `expanding_window_backtest()` — Monthly expanding training window
- `rolling_window_backtest()` — Monthly rolling window (fixed ~6-year training window)
- Per-asset variants of expanding/rolling also available

### Backtesting (`src/run_backtest.py`)
Uses vectorbt: label 0 → enter long, label 2 → exit, label 1 → hold. Configurable stop-loss, fees, slippage. Outputs HTML plots and CSV stats to `vectorbt_reports/`.

## Data Directories (gitignored)

- `market_data/` — Raw CSV input (OHLCV)
- `processed_market_data/` — Feature-engineered data
- `asset_training/` — Per-asset training data
- `backtest_data/` — Prediction CSVs for backtesting
- `torch_model/` — Saved model weights (.pt)
- `vectorbt_reports/` — Backtest results (HTML plots + CSV)
- `training_report/` — Classification report CSVs
- `shap_outputs/` — SHAP feature importance PDFs

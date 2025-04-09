# 📊 REP_2025: Market Prediction with Deep Learning & VectorBT

This project focuses on predicting financial market movements using deep learning models and advanced backtesting techniques. It uses PyTorch for modeling, vectorbt for portfolio backtesting, TA-Lib for technical indicators, and SHAP for model explainability — all inside a GPU-accelerated Docker + Conda environment.

---

## 🧠 Features

- ✅ Training pipeline for market prediction 
- ✅ Backtesting with vectorbt 
- ✅ Technical indicators via TA-Lib
- ✅ SHAP explainability for model interpretation
- ✅ Preprocessing and statistical analysis utilities
- ✅ Fully reproducible Conda + Docker setup
- ✅ GPU-enabled Dev Container support in VS Code

---

## 📁 Project Structure

```
REP_2025/
├── .devcontainer/             # VS Code Dev Container config
├── src/                       # Source code
│   ├── model/                 # Model definitions
│   ├── config/                # Configuration files
│   ├── libs/                  # Utility libraries
│   ├── run_torch_training.py  # Main training script
│   ├── run_backtest.py        # Backtesting logic
│   └── ...                    # Other tools and scripts
├── environment.yml            # Conda environment
├── Dockerfile                 # Docker setup
├── docker-compose.yml         # GPU-ready service definition
└── README.md                  # Project documentation
```

---

## 🚀 Getting Started

### 🔧 Requirements

- Docker
- Docker Compose
- optional: NVIDIA GPU + CUDA Drivers (for GPU support)
- [VS Code + Dev Containers Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) (optional)

---

### 🐳 Run in a Dev Container

```bash
docker-compose build
docker-compose run trading_app
```

Or, open the project in **VS Code**, press `F1`, then choose:

```
Dev Containers: Reopen in Container
```

---

### ⚡ Workflow

Once docker container is running use following commands:

**Activate conda environment inside container:**

```bash
conda activate trading-env
```

**Preprocess data:**

```bash
python src/run_preprocess_dataset.py
```


**Train your model:**

```bash
python src/run_torch_training.py
```

**Predict on out-of-sample data:**

```bash
python src/run_predict_asset.py
```

**Run backtest:**

```bash
python src/run_backtest.py
```

**Explain model predictions with SHAP:**

```bash
python src/run_shap_explainer.py
```
---

## Local Development

It is also possible to just develop in a local conda environment.
Use following command to create a conda environment and to install dependencies.

```bash
conda env create -f environment.yml
```


---

## 📦 Environment

| Package        | Version        |
|----------------|----------------|
| Python         | 3.10           |
| PyTorch        | 2.1.0+cu118     |
| vectorbt       | 0.26.1         |
| numba          | 0.56.4         |
| ta-lib         | Built from source |
| shap           | Latest         |
| imbalanced-learn | Latest       |
| plotly         | ≥ 5.13         |

---

## 📎 Notes

- Market data should be located in `processed_market_data/`
- Run settings and paths are configured via `src/config/`
- Use the `--gpus all` flag with Docker or use the `nvidia` runtime (already handled in `docker-compose.yml`)

---

## 🤝 Contributions

Contributions and improvements are welcome.
Please open an issue or pull request if you'd like to propose changes or fixxes.

---

## 📝 License

This project is licensed under the MIT License.

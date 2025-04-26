import pandas as pd
import numpy as np

from libs.technical_analysis_lib import BUY, HOLD, SELL
from sklearn.preprocessing import StandardScaler
from model.Pytorch_NNModel import NNModel
import os
import traceback
import sys
from config.config import RUN as run_conf
import torch
from torch.utils.data import DataLoader
from model.CustomDataset import CustomDataset
import libs.compute_indicators_labels_lib as compute_indicators_labels_lib
from sklearn.decomposition import PCA
import random


def predict_asset(RUN, filename, mdl_name="torch_model/best_model.pt"):
    """
    Predict BUY, HOLD and SELL signals on a timeseries
    """
    seed = RUN["seed"]
    torch.manual_seed(seed)
    nr = StandardScaler()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = compute_indicators_labels_lib.get_predictions_dataset(RUN, f"{filename}")
    data["Date"] = pd.to_datetime(data["Date"])
    data = data[data["Date"] >= RUN["back_test_start"]]
    data = data[data["Date"] <= RUN["back_test_end"]]
    if len(data.index) == 0:
        raise ValueError("Void dataframe")

    labels = data["label"].copy()
    ohlc = data[["Date", "Open", "High", "Low", "Close", "label"]].copy()
    data.drop(
        columns=[
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Asset_name",
            "label",
        ],
        inplace=True,
    )
    if len(data.index) == 0:
        raise ValueError("Void dataframe")
    columns = data.columns
    index = data.index
    nr.fit(data)
    X = nr.transform(data)
    # === Add PCA here ===
    if "pca_components" in RUN and RUN["pca_components"] is not None:
        pca = PCA(n_components=RUN["pca_components"])
        X = pca.fit_transform(X)
        print(
            f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}"
        )

    # Update DataFrame after PCA
    if "pca_components" in RUN and RUN["pca_components"] is not None:
        columns = [f"PC{i+1}" for i in range(RUN["pca_components"])]
    else:
        columns = data.columns

    data = pd.DataFrame(X, columns=columns, index=index)
    data["label"] = ohlc["label"]

    print(f"train set shape 1: {data.shape[1]}")
    print(f"train set columns: {data.columns}")
    expected_input_size = data.shape[1] - 1
    print(f"Expected input size: {expected_input_size}")
    train_loader = DataLoader(CustomDataset(data, device=device), batch_size=16)
    model = NNModel(data.shape[1] - 1, 3).to(device)
    model.eval()
    model.load_state_dict(torch.load(mdl_name))
    labels = model.predict(train_loader)

    data["label"] = labels
    data["Open"] = ohlc["Open"]
    data["High"] = ohlc["High"]
    data["Low"] = ohlc["Low"]
    data["Close"] = ohlc["Close"]
    data["Date"] = ohlc["Date"]

    # Create an output folder of the predictions
    output_dir = "backtest_data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{filename}.csv")

    # === Append if file exists ===
    if os.path.exists(output_file):
        existing = pd.read_csv(output_file)
        existing["Date"] = pd.to_datetime(existing["Date"])
        combined = pd.concat([existing, data], ignore_index=True)
        combined.drop_duplicates(
            subset=["Date"], inplace=True
        )  # optional deduplication
        combined.sort_values("Date", inplace=True)
        combined.to_csv(output_file, index=False)
        print(f"Appended predictions to {output_file}")
    else:
        data.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    predict_asset(run_conf, "BTC-USD")

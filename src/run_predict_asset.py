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


def predict_asset(RUN, filename, mdl_name="torch_model/best_model.pt"):
    """
    Predict BUY, HOLD and SELL signals on a timeseries
    """
    try:
        nr = StandardScaler()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = compute_indicators_labels_lib.get_backtest_dataset(RUN, filename)
        data["Date"] = pd.to_datetime(data["Date"])
        data = data[data["Date"] >= RUN["back_test_start"]]
        data = data[data["Date"] <= RUN["back_test_end"]]
        print(data)
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
        train_loader = DataLoader(CustomDataset(data, device=device), batch_size=16)
        model = NNModel(data.shape[1] - 1, 3).to(device)
        model.load_state_dict(torch.load(mdl_name))
        labels = model.predict(train_loader)

        data["label"] = labels
        data["Open"] = ohlc["Open"]
        data["High"] = ohlc["High"]
        data["Low"] = ohlc["Low"]
        data["Close"] = ohlc["Close"]
        data["Date"] = ohlc["Date"]

        # Create an output folder of the predictions
        output_dir = "predictions_data"
        os.makedirs(output_dir, exist_ok=True)

        # Save predictions to CSV
        output_file = os.path.join(output_dir, f"{filename}")
        data.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")

    except Exception:
        print("Exception in user code:")
        print("-" * 60)
        traceback.print_exc(file=sys.stdout)
        print("-" * 60)


if __name__ == "__main__":
    predict_asset(run_conf, "MSFT.csv")

import pandas as pd
import numpy as np

from technical_analysis_lib import BUY, HOLD, SELL
from sklearn.preprocessing import StandardScaler
from Pytorch_NNModel import NNModel
import os
import traceback
import sys
from config import RUN as run_conf
import torch
from torch.utils.data import DataLoader
from CustomDataset import CustomDataset
import compute_indicators_labels_lib


def predict_Asset(RUN, filename, mdl_name="torch_model/best_model.pt"):
    """
    Predict BUY, HOLD and SELL signals on a timeseries
    """
    try:
        nr = StandardScaler()

        data = compute_indicators_labels_lib.get_backtest_dataset(RUN, filename)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data = data.dropna()
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
        data = pd.DataFrame(X, columns=columns, index=index)
        data["label"] = ohlc["label"]

        print(f"train set shape 1: {data.shape[1]}")
        print(f"train set columns: {data.columns}")
        train_loader = DataLoader(CustomDataset(data), batch_size=16)
        model = NNModel(data.shape[1] - 1, 3).to("cuda")
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

    except Exception:
        print("Exception in user code:")
        print("-" * 60)
        traceback.print_exc(file=sys.stdout)
        print("-" * 60)


if __name__ == "__main__":
    predict_Asset(run_conf, "NVDA.csv")

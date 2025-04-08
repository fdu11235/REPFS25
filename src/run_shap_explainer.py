import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import libs.compute_indicators_labels_lib as compute_indicators_labels_lib
from model.Pytorch_NNModel import NNModel
import torch
from sklearn.utils import shuffle
import random

from config.config import RUN as run_conf
from numpy.random import seed
from libs.imbalanced_lib import get_sampler
import shap


def explain_shap(RUN, mdl_name="torch_model/best_model.pt"):
    random.seed(RUN["seed"])
    seed(42)

    if torch.cuda.is_available():
        print("GPU is available!")
    scaler = StandardScaler()
    sampler = get_sampler(run_conf["balance_algo"])
    data = compute_indicators_labels_lib.get_dataset(RUN)

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.dropna()
    print("===============")
    print(data)
    print("===============")
    data["Date"] = pd.to_datetime(data["Date"])
    data = data[
        (data["Date"] < RUN["back_test_start"]) | (data["Date"] > RUN["back_test_end"])
    ]  # exclude backtest data from trainig/test set

    data = data[data["pct_change"] < RUN["beta"]]  # remove outliers

    labels = data["label"].copy()
    labels = labels.astype(int)

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
    columns = data.columns
    index = data.index
    X = scaler.fit_transform(data.values)

    data = pd.DataFrame(X, columns=columns, index=index)
    data["label"] = labels
    data.dropna(inplace=True)
    data = shuffle(data, random_state=RUN["seed"])
    data = sampler(data)
    labels = data["label"]
    data.drop(
        columns=["label"],
        inplace=True,
    )

    # Xs, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    # First, split into train+val and test
    train_set, test_set = train_test_split(
        data, test_size=0.3, random_state=RUN["seed"], stratify=labels
    )

    # print(train_set)
    print(f"train set shape 1: {train_set.shape[1]}")
    print(f"train set columns: {train_set.columns}")

    model = NNModel(train_set.shape[1] - 1, 3).to("cuda")
    model = NNModel(data.shape[1], 3).to("cuda")
    model.load_state_dict(torch.load(mdl_name))

    # Select a background dataset (needed for SHAP to compute baseline expectations)
    background = train_set[:100].to_numpy()  # Convert DataFrame to NumPy array
    background = torch.tensor(background, dtype=torch.float32).to("cuda")

    # Select test samples to explain
    test_samples = test_set[:5].to_numpy()
    test_samples = torch.tensor(test_samples, dtype=torch.float32).to("cuda")

    # Create the explainer
    explainer = shap.GradientExplainer(model, background)

    # Compute SHAP values for test samples
    shap_values = explainer.shap_values(test_samples)
    # Beeswarm plot for class 0
    shap.summary_plot(
        shap_values[:, :, 0],  # class 0 SHAP values: shape (5, 68)
        test_samples.cpu().numpy(),  # same 5x68 shape
        feature_names=train_set.columns.tolist(),
    )


if __name__ == "__main__":
    explain_shap(run_conf)

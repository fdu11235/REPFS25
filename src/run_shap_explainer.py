import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import libs.compute_indicators_labels_lib as compute_indicators_labels_lib
from model.Pytorch_NNModel import NNModel
import torch
from sklearn.utils import shuffle
import random
import matplotlib.pyplot as plt

from config.config import RUN as run_conf
from numpy.random import seed
from libs.imbalanced_lib import get_sampler
import shap
import os


def explain_shap(RUN, mdl_name="torch_model/best_model.pt"):
    random.seed(RUN["seed"])
    seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # First, split into train and test
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
    background = torch.tensor(background, dtype=torch.float32).to(device)

    # Select test samples to explain
    test_samples = test_set[:5].to_numpy()
    test_samples = torch.tensor(test_samples, dtype=torch.float32).to(device)

    # Create the explainer
    explainer = shap.GradientExplainer(model, background)

    # Compute SHAP values for test samples
    shap_values = explainer.shap_values(test_samples)

    # Create output directory
    output_dir = "shap_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Create SHAP plot
    shap.summary_plot(
        shap_values[:, :, 0],
        test_samples.cpu().numpy(),
        feature_names=train_set.columns.tolist(),
        show=True,
    )
    """¨
    shap.force_plot(
        explainer.expected_value[0],
        shap_values[0][0],  # explanation for the first sample
        test_samples.cpu().numpy()[0],
        feature_names=train_set.columns.tolist(),
        matplotlib=True,  # use matplotlib backend
    )
    """

    shap.bar_plot(
        shap_values[:, :, 1].mean(axis=0),
        feature_names=train_set.columns.tolist(),
        max_display=30,
    )

    plt.savefig(os.path.join(output_dir, "bar_plot_class1.png"), bbox_inches="tight")

    shap.bar_plot(
        shap_values[:, :, 0].mean(axis=0),
        feature_names=train_set.columns.tolist(),
        max_display=30,
    )

    plt.savefig(os.path.join(output_dir, "bar_plot_class0.png"), bbox_inches="tight")

    shap.bar_plot(
        shap_values[:, :, 2].mean(axis=0),
        feature_names=train_set.columns.tolist(),
        max_display=30,
    )


if __name__ == "__main__":
    explain_shap(run_conf)

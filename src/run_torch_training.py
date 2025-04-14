import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import libs.compute_indicators_labels_lib as compute_indicators_labels_lib
from model.Pytorch_NNModel import NNModel
import torch
from torch.utils.data import DataLoader
from model.CustomDataset import CustomDataset
from sklearn.utils import shuffle
import random
import torch.nn as nn
import torch.optim as optim

from config.config import RUN as run_conf
from numpy.random import seed
from libs.imbalanced_lib import get_sampler
from sklearn.decomposition import PCA


def train_test(RUN, save_to="torch_model/model_final.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(RUN["seed"])
    seed(42)

    if torch.cuda.is_available():
        print("GPU is available!")
    scaler = StandardScaler()
    sampler = get_sampler(run_conf["balance_algo"])
    data = compute_indicators_labels_lib.get_dataset(RUN)
    data["Date"] = pd.to_datetime(data["Date"])
    """
    data = data[
        (data["Date"] < RUN["back_test_start"]) | (data["Date"] > RUN["back_test_end"])
    ]  # exclude backtest data from trainig/test set
    """
    data = data[data["Date"] < RUN["back_test_start"]]

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
    data["label"] = labels
    data.dropna(inplace=True)
    data = shuffle(data, random_state=RUN["seed"])
    data = sampler(data)
    labels = data["label"]
    print(f"Label value counts: {labels.value_counts()}")
    # Xs, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    # First, split into train+val and test
    train_val_set, test_set = train_test_split(
        data, test_size=0.3, random_state=RUN["seed"], stratify=labels
    )
    # Grab labels again for train_val_set
    train_val_labels = train_val_set["label"]
    # Then split train_val into train and val
    train_set, val_set = train_test_split(
        train_val_set,
        test_size=0.2,
        random_state=RUN["seed"],  # 20% of train_val -> 14% of total
        stratify=train_val_labels,
    )

    # print(train_set)
    print(f"train set shape 1: {train_set.shape[1]}")
    print(f"train set columns: {train_set.columns}")

    model = NNModel(train_set.shape[1] - 1, 3).to(device)
    # Define your loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train_loader = DataLoader(CustomDataset(train_set, device=device), batch_size=32)
    test_loader = DataLoader(CustomDataset(test_set, device=device), batch_size=32)
    val_loader = DataLoader(CustomDataset(val_set, device=device), batch_size=32)
    model.print_num_parameters()
    model.train_model(
        train_loader,
        val_loader,
        criterion,
        optimizer,
        int(RUN["epochs"]),
        "torch_model",
    )
    model.save("torch_model")

    preds_test = model.predict(test_loader)
    preds_train = model.predict(train_loader)

    test_rep = classification_report(
        test_set["label"], preds_test, digits=2, output_dict=True
    )
    print(test_rep)
    train_rep = classification_report(
        train_set["label"], preds_train, digits=2, output_dict=True
    )
    print("================================================")
    print(train_rep)


if __name__ == "__main__":
    train_test(run_conf)

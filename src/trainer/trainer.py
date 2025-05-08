import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import numpy as np
import libs.compute_indicators_labels_lib as compute_indicators_labels_lib
from model.Pytorch_NNModel import NNModel
import torch
from torch.utils.data import DataLoader
from model.CustomDataset import CustomDataset
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight

import random
import torch.nn as nn
import torch.optim as optim

from config.config import RUN as run_conf
from libs.imbalanced_lib import get_sampler
from sklearn.decomposition import PCA


class Trainer:
    def __init__(
        self,
        RUN,
        get_data_fn,
        filename=None,
        save_to="torch_model/model_final.pt",
        report_save_to="training_reports/expanding_model_reports.csv",
    ):
        self.RUN = RUN
        self.get_data_fn = get_data_fn
        self.filename = filename
        self.report_save_to = report_save_to
        self.save_to = save_to
        self.seed = RUN["seed"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)
        self._init()

    def _init(self):
        if torch.cuda.is_available():
            print("GPU is available!")
        self.scaler = StandardScaler()
        self.sampler = get_sampler(run_conf["balance_algo"])

    def load_data(self):
        data = (
            self.get_data_fn(self.RUN)
            if self.filename is None
            else self.get_data_fn(self.RUN, self.filename)
        )
        data["Date"] = pd.to_datetime(data["Date"])
        data = data[data["Date"] < self.RUN["back_test_start"]]
        data = data[data["pct_change"] < self.RUN["beta"]]

        # === Optional: Rolling training window filter ===
        train_start = self.RUN["train_start"]
        train_end = self.RUN["train_end"]

        if train_start is not None:
            data = data[(data["Date"] >= train_start) & (data["Date"] < train_end)]

        labels = data["label"].astype(int)
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

        index = data.index
        X = self.scaler.fit_transform(data.values)

        if self.RUN.get("pca_components"):
            pca = PCA(n_components=self.RUN["pca_components"])
            X = pca.fit_transform(X)
            print(
                f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}"
            )
            columns = [f"PC{i+1}" for i in range(self.RUN["pca_components"])]
        else:
            columns = data.columns

        data = pd.DataFrame(X, columns=columns, index=index)
        data["label"] = labels
        data.dropna(inplace=True)
        data = shuffle(data, random_state=self.seed)
        data = self.sampler(data)
        print(data["label"].value_counts())

        return data

    def split_data(self, data):
        labels = data["label"]
        train_val, test = train_test_split(
            data, test_size=0.3, random_state=self.seed, stratify=labels
        )
        train, val = train_test_split(
            train_val,
            test_size=0.2,
            random_state=self.seed,
            stratify=train_val["label"],
        )
        return train, val, test

    def prepare_model(self, input_dim, train_labels):
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_labels),
            y=train_labels.values,
        )
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(
            self.device
        )
        print(f"Class weights: {class_weights_tensor}")
        model = NNModel(input_dim, 3).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        return model, criterion, optimizer

    def create_loaders(self, train, val, test, batch_size=256):
        return (
            DataLoader(CustomDataset(train, device=self.device), batch_size=batch_size),
            DataLoader(CustomDataset(val, device=self.device), batch_size=batch_size),
            DataLoader(CustomDataset(test, device=self.device), batch_size=batch_size),
        )

    def evaluate(self, model, loader, true_labels, name):
        preds = model.predict(loader)
        report = classification_report(true_labels, preds, digits=2, output_dict=True)

        print(
            f"{'-'*20} {name} Report {'-'*20}\n{classification_report(true_labels, preds, digits=2)}"
        )

        # Convert report dictionary to a flat DataFrame
        report_df = pd.DataFrame(report).transpose().reset_index()
        report_df["run_name"] = name
        report_df["Date"] = (
            f"{self.RUN['back_test_start'].strftime('%Y%m%d')}"
            if self.RUN.get("back_test_start")
            else ""
        )
        report_df["Asset"] = self.filename
        # Optional: reorder
        cols = [
            "Date",
            "run_name",
            "index",
            "precision",
            "recall",
            "f1-score",
            "support",
            "Asset",
        ]
        report_df = report_df[cols]

        # Path to your report CSV
        report_path = self.report_save_to
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        # Append or create
        if os.path.exists(report_path):
            report_df.to_csv(report_path, mode="a", header=False, index=False)
        else:
            report_df.to_csv(report_path, mode="w", header=True, index=False)

        return report

    def run(self):
        data = self.load_data()
        train, val, test = self.split_data(data)
        input_dim = train.shape[1] - 1

        model, criterion, optimizer = self.prepare_model(input_dim, train["label"])
        train_loader, val_loader, test_loader = self.create_loaders(train, val, test)

        model.print_num_parameters()
        model.train_model(
            train_loader,
            val_loader,
            criterion,
            optimizer,
            int(self.RUN["epochs"]),
            "torch_model",
        )

        os.makedirs(os.path.dirname(self.save_to), exist_ok=True)
        model.save(self.save_to)

        self.evaluate(model, test_loader, test["label"], name="Test")
        self.evaluate(model, train_loader, train["label"], name="Train")

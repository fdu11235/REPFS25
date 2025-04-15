import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report


import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
import torch
from torch.nn import CrossEntropyLoss

from lightning.pytorch import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, MAE
from lightning.pytorch.tuner import Tuner


torch.set_float32_matmul_precision("high")

# 1. Load and Clean the Data
df = pd.read_csv(
    "processed_market_data/market_data_training_data.csv", parse_dates=["Date"]
)


# Keep only the target label column
df["label"] = df["lab_2_2"]
df.drop(
    columns=[col for col in df.columns if col.startswith("lab_") and col != "lab_2_2"],
    inplace=True,
)

# Ensure numeric columns are numeric
macro_cols = [
    "CPI_XYOY_INDEX",
    "CPI_YOY_INDEX",
    "PCE_CYOY_INDEX",
    "PCE_YOY_INDEX",
    "PPI_YOY_INDEX",
    "GDP_CQOQ_INDEX",
    "IP_INDEX",
    "NAPMNMI_INDEX",
    "NAPMPMI_INDEX",
    "RSTAMOM_INDEX",
    "CONCCONF_INDEX",
    "CONSSENT_INDEX",
    "SBOITOTL_INDEX",
    "INJCJC_INDEX",
    "NFP TCH INDEX",
    "USURTOT INDEX",
    "FDTR_INDEX",
    "FEDL01_INDEX",
    "USGG10YR_INDEX",
    "USGG2YR_INDEX",
    "USGG5YR_INDEX",
    "USYC2Y10_INDEX",
    "USYC3M10_INDEX",
]

tech_cols = [
    "Low",
    "High",
    "Open",
    "Close",
    "Volume",
    "Z_score",
    "RSI",
    "MACD",
    "MACD_Signal_Line",
    "boll",
    "ULTOSC",
    "pct_change",
    "zsVol",
    "PR_MA_Ratio_short",
    "MA_Ratio_short",
    "MA_Ratio",
    "PR_MA_Ratio",
]

for col in macro_cols + tech_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with NaNs
df.dropna(inplace=True)

# Sort by date and create time index
df["Date"] = pd.to_datetime(df["Date"])
df.sort_values("Date", inplace=True)
df["time_idx"] = ((df["Date"] - df["Date"].min()) / pd.Timedelta(hours=6)).astype(int)

# Define training cutoff
max_encoder_length = 28
max_prediction_length = 10
training_cutoff = df["time_idx"].max() - max_prediction_length
max_time_idx = df.groupby("Asset_name")["time_idx"].transform("max")
df["is_train"] = df["time_idx"] <= (max_time_idx - max_prediction_length)

# Set target column
target = "label"

# Define additional unknown reals
additional_unknown_reals = (
    macro_cols
    + tech_cols
    + [
        "pct_change",
        "zsVol",
        "PR_MA_Ratio_short",
        "MA_Ratio_short",
        "MA_Ratio",
        "PR_MA_Ratio",
        "DayOfWeek",
        "Month",
        "Hourly",
    ]
)

# Define dataset
training = TimeSeriesDataSet(
    df[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target=target,
    group_ids=["Asset_name"],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["Asset_name"],
    static_reals=[],
    time_varying_known_categoricals=[],
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[target] + additional_unknown_reals,
    target_normalizer=None,
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,
)

# Validation dataset
validation = TimeSeriesDataSet.from_dataset(
    training, df, predict=True, stop_randomization=True
)

# Dataloaders
batch_size = 256
train_dataloader = training.to_dataloader(
    train=True, batch_size=batch_size, num_workers=0
)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=batch_size, num_workers=0
)

# Build and train model
pl.seed_everything(41)
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.0006,
    hidden_size=16,
    attention_head_size=2,
    dropout=0.1,
    hidden_continuous_size=8,
    log_interval=10,
    reduce_on_plateau_patience=4,
)

# Trainer
early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
)

lr_monitor = LearningRateMonitor(logging_interval="epoch")
tb_logger = TensorBoardLogger("lightning_logs")

trainer = pl.Trainer(
    max_epochs=10,
    accelerator="gpu",
    enable_model_summary=True,
    gradient_clip_val=0.1,
)

# Train
trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

# calcualte mean absolute error on validation set
predictions = best_tft.predict(
    val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="gpu")
)
print(type(predictions.y))  # probably <class 'tuple'>
print(len(predictions.y))  # see how many items
print(type(predictions.y[0]))  # should be <class 'torch.Tensor'>
# Get predicted class and true class
# Get predicted class and true class
y_true = predictions.y[0].cpu().numpy()
y_pred = predictions.output.argmax(dim=-1).cpu().numpy()

# Inspect shapes
print("y_true shape:", y_true.shape)
print("y_pred shape:", y_pred.shape)

# Handle potential (N, 1) or (N, C) shapes in y_true
if y_true.ndim > 1:
    if y_true.shape[1] == 1:
        y_true = y_true.squeeze()  # from (N,1) to (N,)
    else:
        y_true = y_true.argmax(axis=1)  # from one-hot to class index

# Compute classification metrics
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, digits=4))

# Compute metrics
precision = precision_score(y_true, y_pred, average="weighted")
recall = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")
accuracy = accuracy_score(y_true, y_pred)

# Print
print(f"📊 Validation Metrics:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

mae = MAE()(predictions.output, predictions.y)
print(f"Validation MAE: {mae:.4f}")
# raw predictions are a dictionary from which all kind of information including quantiles can be extracted
raw_predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True)

# print(raw_predictions)

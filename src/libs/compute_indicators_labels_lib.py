from operator import concat
import os
import pandas as pd
from libs.technical_analysis_lib import TechnicalAnalysis
import datetime
import random
from config.config import RUN as run_conf
from multiprocessing import pool
import numpy as np
import glob
from functools import reduce
from run_alpha_beta import get_thresholds


# https://stats.stackexchange.com/questions/312780/why-is-accuracy-not-the-best-measure-for-assessing-classification-models
def clean_df(df, filename=None):
    # Define a mapping from full Yahoo column names to standard names
    # Find the prefix dynamically (e.g., "BTC.F" or "AAPL")
    asset_name = os.path.splitext(os.path.basename(filename))[0]
    print(df)
    print("==========================")
    print(asset_name)
    print("COLUMNS:", df.columns.tolist())
    for col in df.columns:
        if "Open" in col:
            prefix = col.split("Open")[0]
            break
    print("==========================")
    if "prefix" not in locals():
        raise RuntimeError("Prefix was never assigned — check column names!")
    print(prefix)
    rename_map = {
        f"{prefix}Open": "Open",
        f"{prefix}High": "High",
        f"{prefix}Low": "Low",
        # f"{prefix}Close": "Close",
        f"{prefix}Volume": "Volume",
        f"{prefix}Adj Close": "Close",  # sometimes it's named this way
        f"{prefix}Adjusted": "Close",  # or this way
    }
    df["Asset_name"] = asset_name
    # Always rename "date" to "Date"
    df = df.rename(columns={"date": "Date", **rename_map})

    # Select the columns you're interested in
    columns_to_keep = ["Date", "Open", "High", "Low", "Close", "Volume", "Asset_name"]
    df = df[[col for col in columns_to_keep if col in df.columns]]
    return df


def output_to_backtest(RUN, final_df):
    # Ensure 'Date' is datetime
    final_df["Date"] = pd.to_datetime(final_df["Date"])

    # Filter rows for back testing
    filtered_df = final_df[
        (final_df["Date"] > RUN["back_test_start"])
        & (final_df["Date"] < RUN["back_test_end"])
    ]

    # Create an output folder (optional)
    output_dir = "backtest_data"
    os.makedirs(output_dir, exist_ok=True)

    # Split by Asset_name and save to CSV
    for asset in filtered_df["Asset_name"].dropna().unique():
        asset_df = filtered_df[filtered_df["Asset_name"] == asset]
        filename = f"{asset.replace(' ', '_').replace('/', '-')}.csv"
        asset_df.to_csv(os.path.join(output_dir, filename), index=False)
        print(f"Saved: {filename}")


def output_to_predictions(RUN, final_df):
    # Ensure 'Date' is datetime
    final_df["Date"] = pd.to_datetime(final_df["Date"])

    # Create an output folder (optional)
    output_dir = "predictions_data"
    os.makedirs(output_dir, exist_ok=True)

    # Split by Asset_name and save to CSV
    for asset in final_df["Asset_name"].dropna().unique():
        asset_df = final_df[final_df["Asset_name"] == asset]
        filename = f"{asset.replace(' ', '_').replace('/', '-')}.csv"
        asset_df.to_csv(os.path.join(output_dir, filename), index=False)
        print(f"Saved: {filename}")


def output_to_asset_training(final_df):
    # Ensure 'Date' is datetime
    final_df["Date"] = pd.to_datetime(final_df["Date"])

    # Create an output folder (optional)
    output_dir = "asset_training"
    os.makedirs(output_dir, exist_ok=True)

    # Split by Asset_name and save to CSV
    for asset in final_df["Asset_name"].dropna().unique():
        asset_df = final_df[final_df["Asset_name"] == asset]
        filename = f"{asset.replace(' ', '_').replace('/', '-')}.csv"
        asset_df.to_csv(os.path.join(output_dir, filename), index=False)
        print(f"Saved: {filename}")


def preprocess_filename(params):
    filename, RUN = params
    data = pd.read_csv(f"{RUN['folder']}{filename}")
    data = clean_df(data, filename)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.dropna()
    data = TechnicalAnalysis.compute_oscillators(data)
    # data = TechnicalAnalysis.find_patterns(data)
    data = TechnicalAnalysis.add_timely_data(data)
    # alpha, beta = get_thresholds(data)
    labels = pd.DataFrame()
    for bw in range(1, RUN["b_lim_sup_window"]):
        for fw in range(1, RUN["f_lim_sup_window"]):
            labels["lab_%d_%d" % (bw, fw)] = TechnicalAnalysis.assign_labels(
                data, bw, fw, RUN["alpha"], RUN["beta"]
            )

    return data, labels


def preprocess_thresholds(params):
    filename, RUN = params
    data = pd.read_csv(f"{RUN['folder']}{filename}")
    data = clean_df(data, filename)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.dropna()
    data = TechnicalAnalysis.compute_oscillators(data)
    # data = TechnicalAnalysis.find_patterns(data)
    data = TechnicalAnalysis.add_timely_data(data)
    alpha, beta = get_thresholds(data)
    labels = pd.DataFrame()
    for bw in range(1, RUN["b_lim_sup_window"]):
        for fw in range(1, RUN["f_lim_sup_window"]):
            labels["lab_%d_%d" % (bw, fw)] = TechnicalAnalysis.assign_labels(
                data, bw, fw, alpha, beta
            )

    return data, labels


# Helper function to load and format CSVs from a folder
def load_data_from_folder(folder):
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file, parse_dates=["date"])
        df = df.rename(columns={"date": "Date"})
        col_name = os.path.splitext(os.path.basename(file))[0]
        df = df[["Date", "px"]].rename(columns={"px": col_name})
        dfs.append(df)
    return dfs


def preprocess(RUN):
    """
    Parallel preprocessing and labeling of datasets
    Save final dataset to a file in preprocessed_data folder
    :param RUN: configuration dict
    :return:
    """
    jobs = pool.Pool(24)

    # print("Preprocessing with: %s" % RUN)
    filenames = os.listdir(RUN["folder"])
    args = zip(filenames, [RUN] * len(filenames))
    args = [(k, v) for k, v in args]
    print(args)

    # We engineer features of market data
    data_labels = jobs.map(preprocess_filename, args)
    jobs.terminate()
    data_list = [d[0] for d in data_labels]
    labels_list = [d[1] for d in data_labels]

    concat_data = pd.concat(data_list, ignore_index=True)
    concat_data["Date"] = pd.to_datetime(concat_data["Date"])
    concat_labels = pd.concat(labels_list, ignore_index=True)

    # After merging with labels
    market_df = pd.concat(
        [concat_data.reset_index(drop=True), concat_labels.reset_index(drop=True)],
        axis=1,
    )
    market_df = market_df.dropna()

    # We add macrodata from following directories
    folders = ["inflation", "economy", "expectation", "labor_market", "interest_rates"]

    # Load all data from each folder
    all_dfs = []
    for folder in folders:
        all_dfs.extend(load_data_from_folder(folder))

    # Merge all datasets from the above folders on 'Date'
    combined_macro_df = reduce(
        lambda left, right: pd.merge(left, right, on="Date", how="outer"), all_dfs
    )

    # Merge everything on 'Date'
    final_df = pd.merge(combined_macro_df, market_df, on="Date", how="outer")
    final_df = final_df.sort_values("Date").reset_index(drop=True)

    # feature engineering on macro data
    final_df = final_df.dropna()
    final_df = TechnicalAnalysis.compute_macro_features(final_df)
    final_df.to_csv(
        "processed_market_data/%straining_data.csv" % RUN["folder"].replace("/", "_"),
        index=False,
    )
    output_to_predictions(RUN, final_df)


def preprocess_asset(RUN):
    """
    Parallel preprocessing and labeling of datasets
    Save final dataset to a file in preprocessed_data folder
    :param RUN: configuration dict
    :return:
    """
    jobs = pool.Pool(24)

    # print("Preprocessing with: %s" % RUN)
    filenames = os.listdir(RUN["folder"])
    args = zip(filenames, [RUN] * len(filenames))
    args = [(k, v) for k, v in args]
    print(args)

    # We engineer features of market data
    data_labels = jobs.map(preprocess_thresholds, args)
    jobs.terminate()
    data_list = [d[0] for d in data_labels]
    labels_list = [d[1] for d in data_labels]

    concat_data = pd.concat(data_list, ignore_index=True)
    concat_data["Date"] = pd.to_datetime(concat_data["Date"])
    concat_labels = pd.concat(labels_list, ignore_index=True)

    # After merging with labels
    market_df = pd.concat(
        [concat_data.reset_index(drop=True), concat_labels.reset_index(drop=True)],
        axis=1,
    )
    market_df = market_df.dropna()

    # We add macrodata from following directories
    folders = ["inflation", "economy", "expectation", "labor_market", "interest_rates"]

    # Load all data from each folder
    all_dfs = []
    for folder in folders:
        all_dfs.extend(load_data_from_folder(folder))

    # Merge all datasets from the above folders on 'Date'
    combined_macro_df = reduce(
        lambda left, right: pd.merge(left, right, on="Date", how="outer"), all_dfs
    )

    # Merge everything on 'Date'
    final_df = pd.merge(combined_macro_df, market_df, on="Date", how="outer")
    final_df = final_df.sort_values("Date").reset_index(drop=True)

    # feature engineering on macro data
    final_df = final_df.dropna()
    final_df = TechnicalAnalysis.compute_macro_features(final_df)
    output_to_asset_training(final_df)


def get_dataset(RUN):
    """
    returns a dataset labeled with given forward and backward window
    :param RUN: run configuration dictionary
    :return: pandas dataframe wit 'label' column
    """

    ds = pd.read_csv(
        "processed_market_data/%straining_data.csv" % RUN["folder"].replace("/", "_")
    )
    ds.replace([np.inf, -np.inf], np.nan, inplace=True)
    ds = ds.dropna()

    fw = RUN["f_window"]
    bw = RUN["b_window"]
    label_col = "lab_%d_%d" % (bw, fw)

    labels = ds[label_col].copy()

    droped_lab = []
    for bw in range(1, RUN["b_lim_sup_window"]):
        for fw in range(1, RUN["f_lim_sup_window"]):
            label_col = "lab_%d_%d" % (bw, fw)
            droped_lab.append(label_col)

    ds = ds.drop(columns=droped_lab)

    ds["label"] = labels

    return ds


def get_asset_dataset(RUN, file):
    """
    returns a asset datasets labeled with given forward and backward window
    :param RUN: run configuration dictionary
    :return: pandas dataframe wit 'label' column
    """

    ds = pd.read_csv(f"asset_training/{file}.csv")
    ds.replace([np.inf, -np.inf], np.nan, inplace=True)
    ds = ds.dropna()

    fw = RUN["f_window"]
    bw = RUN["b_window"]
    label_col = "lab_%d_%d" % (bw, fw)

    labels = ds[label_col].copy()

    droped_lab = []
    for bw in range(1, RUN["b_lim_sup_window"]):
        for fw in range(1, RUN["f_lim_sup_window"]):
            label_col = "lab_%d_%d" % (bw, fw)
            droped_lab.append(label_col)

    ds = ds.drop(columns=droped_lab)

    ds["label"] = labels

    return ds


def get_backtest_dataset(RUN, filename):
    """
    returns the backtesting dataset labeled with given forward and backward window
    :param RUN: run configuration dictionary
    :return: pandas dataframe wit 'label' column
    """
    filepath = f"backtest_data/{filename}.csv"
    try:
        ds = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")

    fw = RUN["f_window"]
    bw = RUN["b_window"]
    label_col = "lab_%d_%d" % (bw, fw)

    labels = ds[label_col].copy()

    droped_lab = []
    for bw in range(1, RUN["b_lim_sup_window"]):
        for fw in range(1, RUN["f_lim_sup_window"]):
            label_col = "lab_%d_%d" % (bw, fw)
            droped_lab.append(label_col)

    ds = ds.drop(columns=droped_lab)

    ds["label"] = labels

    return ds


def get_predictions_dataset(RUN, filename):
    """
    returns the dataset where we run predictions
    :param RUN: run configuration dictionary
    :return: pandas dataframe wit 'label' column
    """
    filepath = f"predictions_data/{filename}.csv"
    try:
        ds = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")

    fw = RUN["f_window"]
    bw = RUN["b_window"]
    label_col = "lab_%d_%d" % (bw, fw)

    labels = ds[label_col].copy()

    droped_lab = []
    for bw in range(1, RUN["b_lim_sup_window"]):
        for fw in range(1, RUN["f_lim_sup_window"]):
            label_col = "lab_%d_%d" % (bw, fw)
            droped_lab.append(label_col)

    ds = ds.drop(columns=droped_lab)

    ds["label"] = labels

    return ds

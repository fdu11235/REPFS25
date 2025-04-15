from datetime import timedelta
from pandas import Timestamp
from run_torch_training import train_test
from run_predict_asset import predict_asset
from run_backtest import backtest
from libs.compute_indicators_labels_lib import preprocess
from config.config import RUN as run_conf_base
import os


def fixed_backtest():
    data_dir = "market_data/"  # change this to your actual path if needed
    asset_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    assets = [f.replace(".csv", "") for f in asset_files]

    preprocess(run_conf_base)
    train_test(run_conf_base)

    for asset in assets:
        predict_asset(run_conf_base, asset, mdl_name="torch_model/best_model.pt")
        backtest(run_conf_base, "backtest_data", asset)


def expanding_window_backtest():
    data_dir = "market_data"
    asset_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    assets = [f.replace(".csv", "") for f in asset_files]

    current_start = Timestamp("2024-06-01")
    final_end = Timestamp("2025-04-01")
    test_window = timedelta(days=30)

    while current_start < final_end:
        current_end = current_start + test_window

        # Copy base config and set current interval
        run_conf = run_conf_base.copy()
        run_conf["back_test_start"] = current_start
        run_conf["back_test_end"] = current_end

        print(
            f"\n======= Running interval {current_start.date()} to {current_end.date()} =======\n"
        )

        # Train once per interval
        train_test(
            run_conf,
            f"torch_model/expanding_model/model_{current_start.strftime('%Y%m%d')}_{current_end.strftime('%Y%m%d')}.pt",
        )

        # Predict + backtest for each asset
        for asset in assets:
            print(f"--- Processing asset: {asset} ---")
            predict_asset(
                run_conf,
                asset,
                mdl_name=f"torch_model/expanding_model/model_{current_start.strftime('%Y%m%d')}_{current_end.strftime('%Y%m%d')}.pt",
            )
            backtest(
                run_conf,
                "backtest_data",
                asset,
                "vectorbt_reports/expanding_master_backtest_stats.csv",
            )

        # Move window forward
        current_start = current_end


if __name__ == "__main__":
    fixed_backtest()
    # growing_window_backtest()
    # expanding_window_backtest()

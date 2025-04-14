from datetime import timedelta
from pandas import Timestamp
from run_torch_training import train_test
from run_predict_asset import predict_asset
from run_backtest import backtest
from config.config import RUN as run_conf_base
import os


def main():
    data_dir = "market_data/"  # change this to your actual path if needed
    asset_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    assets = [f.replace(".csv", "") for f in asset_files]
    current_start = Timestamp("2024-06-01")
    final_end = Timestamp("2025-04-01")
    test_window = timedelta(days=30)

    for asset in assets:
        current_start = Timestamp("2024-06-01")  # reset for each asset
        print(f"\n====== Running for {asset} ======\n")

        while current_start < final_end:
            current_end = current_start + test_window

            # Copy base config to avoid overwriting
            run_conf = run_conf_base.copy()
            run_conf["back_test_start"] = current_start
            run_conf["back_test_end"] = current_end

            print(
                f"Running experiment from {current_start.date()} to {current_end.date()}"
            )
            train_test(run_conf)
            predict_asset(run_conf, asset, mdl_name="torch_model/best_model.pt")
            backtest(run_conf, "backtest_data", asset)

            # Move the start forward by 1 month
            current_start = current_end


if __name__ == "__main__":
    main()

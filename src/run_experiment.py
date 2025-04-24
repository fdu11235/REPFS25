import os
from datetime import timedelta
from pandas import Timestamp

from trainer import Trainer
from config.config import RUN as run_conf_base
from libs import compute_indicators_labels_lib
from run_predict_asset import predict_asset
from run_backtest import backtest
from run_backtest1 import backtest_all_assets_with_volatility_hedge


def train_main_model():
    compute_indicators_labels_lib.preprocess(run_conf_base)
    trainer = Trainer(
        RUN=run_conf_base,
        get_data_fn=compute_indicators_labels_lib.get_dataset,
        save_to="torch_model/model_final.pt",
    )
    trainer.run()


def fixed_backtest():
    train_main_model()

    data_dir = "market_data/"
    asset_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    assets = [f.replace(".csv", "") for f in asset_files]

    for asset in assets:
        predict_asset(run_conf_base, asset, mdl_name="torch_model/model_final.pt")
        backtest(run_conf_base, "backtest_data", asset)


def fixed_backtest_per_asset():
    compute_indicators_labels_lib.preprocess(run_conf_base)

    data_dir = "asset_training/"
    asset_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    assets = [f.replace(".csv", "") for f in asset_files]

    for asset in assets:
        model_path = f"torch_model/per_asset/model_{asset}.pt"
        trainer = Trainer(
            RUN=run_conf_base,
            get_data_fn=compute_indicators_labels_lib.get_asset_dataset,
            filename=asset,
            save_to=model_path,
        )
        trainer.run()

        predict_asset(run_conf_base, asset, mdl_name=model_path)
        backtest(run_conf_base, "backtest_data", asset)


def expanding_window_backtest():
    data_dir = "market_data/"
    asset_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    assets = [f.replace(".csv", "") for f in asset_files]

    current_start = Timestamp("2024-06-01")
    final_end = Timestamp("2025-04-01")
    test_window = timedelta(days=30)

    while current_start < final_end:
        current_end = current_start + test_window
        run_conf = run_conf_base.copy()
        run_conf["back_test_start"] = current_start
        run_conf["back_test_end"] = current_end
        date_suffix = (
            f"{current_start.strftime('%Y%m%d')}_{current_end.strftime('%Y%m%d')}"
        )

        print(f"\n======= Running interval {date_suffix} =======\n")

        model_path = f"torch_model/expanding_model/model_{date_suffix}.pt"
        trainer = Trainer(
            RUN=run_conf,
            get_data_fn=compute_indicators_labels_lib.get_dataset,
            save_to=model_path,
        )
        trainer.run()

        for asset in assets:
            print(f"--- Processing asset: {asset} ---")
            predict_asset(run_conf, asset, mdl_name=model_path)
            backtest(
                run_conf,
                "backtest_data",
                asset,
                "vectorbt_reports/expanding_master_backtest_stats.csv",
            )

        current_start = current_end


def expanding_window_backtest_per_asset():
    compute_indicators_labels_lib.preprocess(run_conf_base)

    data_dir = "asset_training/"
    asset_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    assets = [f.replace(".csv", "") for f in asset_files]

    current_start = Timestamp("2024-06-01")
    final_end = Timestamp("2025-04-01")
    test_window = timedelta(days=30)

    while current_start < final_end:
        current_end = current_start + test_window
        run_conf = run_conf_base.copy()
        run_conf["back_test_start"] = current_start
        run_conf["back_test_end"] = current_end
        date_suffix = (
            f"{current_start.strftime('%Y%m%d')}_{current_end.strftime('%Y%m%d')}"
        )

        print(f"\n======= Running interval {date_suffix} =======\n")

        for asset in assets:
            print(f"--- Processing asset: {asset} ---")
            model_path = f"torch_model/per_asset/model_{asset}_{date_suffix}.pt"

            trainer = Trainer(
                RUN=run_conf,
                get_data_fn=compute_indicators_labels_lib.get_asset_dataset,
                filename=asset,
                save_to=model_path,
            )
            trainer.run()

            predict_asset(run_conf, asset, mdl_name=model_path)
            backtest(
                run_conf,
                "backtest_data",
                asset,
                "vectorbt_reports/per_asset_expanding_master_backtest_stats.csv",
            )

        current_start = current_end


if __name__ == "__main__":
    # You can switch between runs here
    fixed_backtest()
    # fixed_backtest_per_asset()
    # expanding_window_backtest()
    # expanding_window_backtest_per_asset()

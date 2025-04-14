from run_torch_training import train_test
from run_predict_asset import predict_asset
from run_backtest import backtest
from config.config import RUN as run_conf


def main():
    train_test(run_conf)
    predict_asset(run_conf, "BTC-USD", mdl_name="torch_model/best_model.pt")
    backtest(run_conf, "predictions_data", "BTC-USD")
    return


if __name__ == "__main__":
    main()

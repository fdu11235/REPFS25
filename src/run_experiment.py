from run_torch_training import train_test
from run_predict_asset import predict_asset
from run_backtest import backtest
from config.config import RUN as run_conf


def main():
    train_test(run_conf, save_to="torch_model/model_final.pt")
    predict_asset(run_conf, "BTC-USD.csv", mdl_name="torch_model/best_model.pt")
    backtest("predictions_data", "BTC-USD.csv")
    return


if __name__ == "__main__":
    main()
from pandas import Timestamp

run1 = {
    "folder": "market_data/",
    "processed_folder": "processed_market_data/",
    "reports": "reports/",
    "alpha": 0.0262,  # computed in determine_alpha.py
    "beta": 0.1293,  # 0.1293 ignore sample greater than beta in percent of change
    "seed": 353598215,
    "commission fee": 0.001,  # 0.0004
    "b_window": 5,
    "f_window": 2,
    # used in define the grid for searching backward and forward window
    "b_lim_sup_window": 6,
    "f_lim_sup_window": 6,
    "back_test_start": Timestamp("2024-04-01"),
    "back_test_end": Timestamp("2025-04-01"),
    "suffix": "ncr",
    "stop_loss": 0.2,
    "off_label_set": [],  # ['BTCUSDT', 'ETHUSDT', 'ALGOUSDT']  # list of coin to be excluded from training/test set. Used in backtesting
    "balance_algo": "srs",  # 'ncr', 'srs', None
    "loss_func": None,  # 'focal', 'categorical'
    "epochs": 100,  # how many epochs spent in training neural network
    "pca_components": None,  # Set to None to disable PCA
}

RUN = run1

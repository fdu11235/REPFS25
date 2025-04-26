import pandas as pd
import vectorbt as vbt
import os
from config.config import RUN as run_conf
from libs.compute_indicators_labels_lib import get_backtest_dataset


def backtest(
    RUN,
    read_dir,
    filename,
    master_path="vectorbt_reports/fixed_master_backtest_stats.csv",
    plot_dir_root="vectorbt_reports/expanding_plots",
):
    """
    do backtest on a backtest dataset in specified directory
    """
    # Load CSV
    df = pd.read_csv(f"{read_dir}/{filename}.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    # print(df)
    # 2. Extract price and label columns
    price = df["Close"]
    labels = df["label"]

    # 3. Initialize position state and build signals
    in_position = False
    entries = []
    exits = []

    for label in labels:
        if label == 0 and not in_position:  # Buy
            entries.append(True)
            exits.append(False)
            in_position = True
        elif label == 2 and in_position:  # Sell
            entries.append(False)
            exits.append(True)
            in_position = False
        else:  # Hold
            entries.append(False)
            exits.append(False)

    # Ensure we exit on the last day if still in position
    # if in_position:
    #   entries[-1] = False
    #  exits[-1] = True

    # 4. Convert to Series
    entries = pd.Series(entries, index=labels.index)
    exits = pd.Series(exits, index=labels.index)
    # 5. Run backtest
    portfolio = vbt.Portfolio.from_signals(
        close=price,
        entries=entries,
        exits=exits,
        init_cash=10_000,
        fees=0.001,
        sl_stop=0.05,  # 5% stop-loss
        tp_stop=0.20,  # 10% take-profit
        freq="1D",
    )
    # Create output directory if it doesn't exist
    output_dir = "vectorbt_reports"
    os.makedirs(output_dir, exist_ok=True)

    # portfolio.plot().show()

    # Set up date-based plot directory
    start_date = RUN["back_test_start"].strftime("%Y-%m-%d")
    end_date = RUN["back_test_end"].strftime("%Y-%m-%d")
    plot_dir = os.path.join(plot_dir_root, filename)
    os.makedirs(plot_dir, exist_ok=True)

    # Save plot
    fig = portfolio.plot()
    plot_name = f"{filename}_{start_date}_to_{end_date}.html"
    fig.write_html(os.path.join(plot_dir, plot_name))
    print(f"Plot saved to {os.path.join(plot_dir, plot_name)}")

    # Save stats
    stats = portfolio.stats()
    stats = pd.DataFrame([stats])
    stats["asset"] = filename
    # stats.to_csv(os.path.join(output_dir, "backtest_stats.csv"), index=False)

    # Append or create master file
    if os.path.exists(master_path):
        existing = pd.read_csv(master_path)
        combined = pd.concat([existing, stats], ignore_index=True)
        combined.to_csv(master_path, index=False)
    else:
        stats.to_csv(master_path, index=False)

    print(f"Backtest results saved to {output_dir}")
    print("Master CSV updated.")
    print(stats)


if __name__ == "__main__":
    backtest(run_conf, "backtest_data", "BTC-USD")

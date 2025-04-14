import pandas as pd
import vectorbt as vbt
import os


def backtest(dir, filename):
    """
    do backtest on a backtest dataset in specified directory
    """
    # 1. Load your CSV
    df = pd.read_csv(f"{dir}/{filename}")
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

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
    if in_position:
        entries[-1] = False
        exits[-1] = True

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
    )

    # 6. Show results
    # Create output directory if it doesn't exist
    output_dir = "vectorbt_reports"
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    fig = portfolio.plot()
    output_path = os.path.join(output_dir, "backtest_output.html")
    fig.write_html(output_path)
    print(f"Plot saved to {output_path}")

    print(portfolio.stats(settings=dict(freq="1D")))
    fig = portfolio.plot()
    fig.write_html("vectorbt_reports/backtest.html")
    portfolio.plot().show()

    ########################################################################################

    # 5. Create unique output directory based on date range
    start_date = df.index.min().strftime("%Y-%m-%d")
    end_date = df.index.max().strftime("%Y-%m-%d")
    base_name = f"{filename}_{start_date}_to_{end_date}"
    output_dir = os.path.join("vectorbt_reports", base_name)
    os.makedirs(output_dir, exist_ok=True)

    # 6. Save backtest plot
    fig = portfolio.plot()
    fig.write_html(os.path.join(output_dir, "backtest_plot.html"))

    # 7. Save stats
    stats = portfolio.stats(settings=dict(freq="1D"))
    stats.to_csv(os.path.join(output_dir, "backtest_stats.csv"))
    stats.to_json(os.path.join(output_dir, "backtest_stats.json"))

    # 8. Append to master CSV
    master_path = "vectorbt_reports/master_backtest_stats.csv"

    # Add metadata
    stats["Start_Date"] = start_date
    stats["End_Date"] = end_date
    stats["Filename"] = filename

    # Transpose and convert to row format
    stats_row = stats.T
    stats_row["Start_Date"] = start_date
    stats_row["End_Date"] = end_date
    stats_row["Filename"] = filename

    # Append or create master file
    if os.path.exists(master_path):
        existing = pd.read_csv(master_path)
        combined = pd.concat([existing, stats_row], ignore_index=True)
        combined.to_csv(master_path, index=False)
    else:
        stats_row.to_csv(master_path, index=False)

    print(f"Backtest results saved to {output_dir}")
    print("Master CSV updated.")
    print(stats)


if __name__ == "__main__":
    backtest("predictions_data","MSFT.csv")

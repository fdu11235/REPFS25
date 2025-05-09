import pandas as pd
import vectorbt as vbt
import numpy as np
import os
from config.config import RUN as run_conf
from scipy.stats import ttest_1samp
import matplotlib.pyplot as plt


def plot_tsla_trades(
    price_df, entry_df, exit_df, output_path="vectorbt_reports/tsla_trades.pdf"
):
    tsla_price = price_df["GOOGL"].dropna()
    tsla_entry = entry_df["GOOGL"].reindex(tsla_price.index).fillna(False).astype(bool)
    tsla_exit = exit_df["GOOGL"].reindex(tsla_price.index).fillna(False).astype(bool)

    # Entry/Exit points
    entry_dates = tsla_entry[tsla_entry].index
    entry_prices = tsla_price.loc[entry_dates]

    exit_dates = tsla_exit[tsla_exit].index
    exit_prices = tsla_price.loc[exit_dates]

    # Plotting
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(14, 6), facecolor="none")

    ax.plot(
        tsla_price.index,
        tsla_price.values,
        label="GOOGL Price",
        color="black",
        linewidth=1.5,
    )
    ax.scatter(
        entry_dates,
        entry_prices,
        marker="^",
        color="green",
        s=100,
        label="Entry",
        zorder=3,
    )
    ax.scatter(
        exit_dates, exit_prices, marker="v", color="red", s=100, label="Exit", zorder=3
    )

    ax.set_title("GOOGL Price with Executed Trades", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    # ax.legend()

    # Force visible gridlines
    ax.grid(
        True,
        which="major",
        linestyle="-",
        linewidth=0.1,
        color="black",
        alpha=0.3,
    )
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save with transparent background
    plt.savefig(output_path, format="pdf", transparent=True)
    plt.close()


def backtest_multi_asset(
    RUN,
    read_dir,
    master_path="vectorbt_reports/portfolio_classic_backtest_stats.csv",
    plot_dir_root="vectorbt_reports/portfolio_classic_plots",
):
    """
    Backtest all assets in read_dir as a multi-asset portfolio
    """
    prices = []
    entries_list = []
    exits_list = []
    asset_names = []

    daily_returns_all = []
    label_flags_all = []

    for file in os.listdir(read_dir):
        if file.endswith(".csv"):
            asset_name = file.replace(".csv", "")
            df = pd.read_csv(os.path.join(read_dir, file))
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)

            price = df["Close"]
            labels = df["label"]

            ret = price.pct_change().fillna(0).values

            # Only consider returns when model was "in position"
            # We assume '1' means in position (holding the asset), adjust if needed
            in_position_mask = labels == 1  # or use your entry-exit logic if different
            strategy_returns = ret[in_position_mask]

            daily_returns_all.extend(ret.tolist())
            label_flags_all.extend(in_position_mask.tolist())

            in_position = False
            entries = []
            exits = []

            for label in labels:
                if label == 0 and not in_position:
                    entries.append(True)
                    exits.append(False)
                    in_position = True
                elif label == 2 and in_position:
                    entries.append(False)
                    exits.append(True)
                    in_position = False
                else:
                    entries.append(False)
                    exits.append(False)

            prices.append(price)
            entries_list.append(pd.Series(entries, index=price.index))
            exits_list.append(pd.Series(exits, index=price.index))
            asset_names.append(asset_name)

    # Combine into aligned DataFrames
    price_df = pd.concat(prices, axis=1)
    entry_df = pd.concat(entries_list, axis=1).astype(bool)
    exit_df = pd.concat(exits_list, axis=1).astype(bool)
    price_df.columns = entry_df.columns = exit_df.columns = asset_names

    # Backtest
    portfolio = vbt.Portfolio.from_signals(
        close=price_df,
        entries=entry_df,
        exits=exit_df,
        init_cash=10_000,
        fees=0.00001,
        # tp_stop=0.2,
        sl_stop=0.1,
        freq="1D",
        cash_sharing=False,
    )

    # Save stats
    stats = portfolio.stats()
    print(stats)
    print(portfolio.value().tail())  # see if shape is (dates, assets) or just (dates,)
    returns = portfolio.daily_returns()

    for col in returns.columns:
        r = returns[col].dropna()
        t_stat, p_value = ttest_1samp(r, 0)
        print(f"{col}: t = {t_stat:.3f}, p = {p_value:.4f}")

    # Flatten all returns into one series
    all_returns = returns.stack().dropna().values
    print(len(all_returns))

    # Bootstrap
    n_boot = 10000
    boot_means = np.array(
        [
            np.mean(np.random.choice(all_returns, size=len(all_returns), replace=True))
            for _ in range(n_boot)
        ]
    )

    # Calculate p-value: proportion of bootstrap samples where mean ≤ 0
    p_value = np.mean(boot_means <= 0)

    print(f"Bootstrap mean: {np.mean(boot_means):.6f}")
    print(f"Empirical p-value (H0: mean <= 0): {p_value:.4f}")

    # plot_tsla_trades(
    #    price_df, entry_df, exit_df, output_path="vectorbt_reports/googl_trades.pdf"
    # )


def backtest_multi_asset_p(
    RUN,
    read_dir,
    master_path="vectorbt_reports/portfolio_classic_backtest_stats.csv",
    plot_dir_root="vectorbt_reports/portfolio_classic_plots",
    n_permutations=10000,
):
    """
    Backtest all assets in read_dir as a multi-asset portfolio,
    and perform a Monte Carlo permutation test to assess statistical significance.
    """
    prices = []
    entries_list = []
    exits_list = []
    asset_names = []

    # For permutation test
    daily_returns_all = []
    label_flags_all = []

    for file in os.listdir(read_dir):
        if file.endswith(".csv"):
            asset_name = file.replace(".csv", "")
            df = pd.read_csv(os.path.join(read_dir, file))
            df["Date"] = pd.to_datetime(df["Date"])
            df = df[df["Date"] >= "2024-04-01"]
            df.set_index("Date", inplace=True)

            price = df["Close"]
            labels = df["label"].values
            ret = price.pct_change().fillna(0).values

            in_position = False
            entries = []
            exits = []

            # Generate entries and exits
            for label in labels:
                if label == 0 and not in_position:
                    entries.append(True)
                    exits.append(False)
                    in_position = True
                elif label == 2 and in_position:
                    entries.append(False)
                    exits.append(True)
                    in_position = False
                else:
                    entries.append(False)
                    exits.append(False)

            prices.append(price)
            entries_list.append(pd.Series(entries, index=price.index))
            exits_list.append(pd.Series(exits, index=price.index))
            asset_names.append(asset_name)

            # For permutation test: define "in position" as label == 1
            daily_returns_all.extend(ret.tolist())
            label_flags_all.extend((df["label"].values == 1).tolist())

    # Combine into aligned DataFrames
    price_df = pd.concat(prices, axis=1)
    entry_df = pd.concat(entries_list, axis=1).astype(bool)
    exit_df = pd.concat(exits_list, axis=1).astype(bool)
    price_df.columns = entry_df.columns = exit_df.columns = asset_names

    # Backtest
    portfolio = vbt.Portfolio.from_signals(
        close=price_df,
        entries=entry_df,
        exits=exit_df,
        init_cash=10_000,
        fees=0.00001,
        sl_stop=0.1,
        freq="1D",
        cash_sharing=False,
    )

    stats = portfolio.stats()
    print(stats)
    daily_returns = portfolio.daily_returns().dropna()
    n_boot = 10000

    results = {}

    for asset in daily_returns.columns:
        r = daily_returns[asset].values
        real_mean = np.mean(r)

        # Step 1–3: zero-center the returns
        zero_centered = r - real_mean

        # Step 4–5: resample with replacement and compute mean return each time
        boot_means = np.array(
            [
                np.mean(np.random.choice(zero_centered, size=len(r), replace=True))
                for _ in range(n_boot)
            ]
        )

        # Step 6: compute p-value
        p_value = np.mean(boot_means >= real_mean)

        results[asset] = {"mean_return": real_mean, "p_value": p_value}

        print(
            f"{asset}: mean return = {real_mean:.6f}, bootstrap p-value = {p_value:.4f}"
        )


if __name__ == "__main__":
    backtest_multi_asset_p(run_conf, "backtest_data/rolling_daily_16")

import os
import pandas as pd
import numpy as np
import vectorbt as vbt
from config.config import RUN as run_conf


def backtest_all_assets_with_volatility_hedge(
    RUN,
    read_dir,
    hedge_asset="ES=F.csv",
    master_path="vectorbt_reports/fixed_master_backtest_stats.csv",
    plot_dir_root="vectorbt_reports/expanding_plots",
    init_cash=10_000,
    capital_allocation_ratio=0.5,  # 50% capital per trade (main + hedge)
):
    # Load hedge data
    hedge_df = pd.read_csv(os.path.join(read_dir, hedge_asset))
    hedge_df["Date"] = pd.to_datetime(hedge_df["Date"])
    hedge_df.set_index("Date", inplace=True)
    hedge_price = hedge_df["Close"]

    stats_list = []

    for filename in os.listdir(read_dir):
        if not filename.endswith(".csv") or filename == hedge_asset:
            continue

        asset_name = filename.replace(".csv", "")
        df = pd.read_csv(os.path.join(read_dir, filename))
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

        # Align with hedge
        merged = pd.DataFrame(index=df.index.union(hedge_df.index)).sort_index()
        merged = merged.join(
            df[["Close", "label"]].rename(columns={"Close": asset_name})
        )
        merged = merged.join(hedge_price.rename("HEDGE"))
        merged = merged.ffill().dropna()

        price_main = merged[asset_name]
        price_hedge = merged["HEDGE"]
        labels = merged["label"]

        # === Build signals ===
        in_position = False
        entries, exits = [], []
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

        entries = pd.Series(entries, index=labels.index)
        exits = pd.Series(exits, index=labels.index)

        # === Hedge entry/exit — same timing as main trade ===
        hedge_entries = entries.copy()
        hedge_exits = exits.copy()

        # === Hedge ratio ===
        main_vol = price_main.pct_change().rolling(20).std()
        hedge_vol = price_hedge.pct_change().rolling(20).std()
        hedge_ratio = (main_vol / hedge_vol).replace(0, np.nan).fillna(method="ffill")
        hedge_ratio = hedge_ratio.clip(0.1, 2.0).ewm(span=10).mean()

        # === Capital allocation ===
        dollar_alloc = init_cash * capital_allocation_ratio
        main_size = dollar_alloc / price_main
        hedge_size = (dollar_alloc * hedge_ratio) / price_hedge

        # === Signal DataFrames ===
        price_df = pd.DataFrame({asset_name: price_main, "HEDGE": price_hedge})
        size_df = pd.DataFrame({asset_name: main_size, "HEDGE": hedge_size})

        # Long and short entries/exists
        entries_df = pd.DataFrame({asset_name: entries, "HEDGE": False})
        exits_df = pd.DataFrame({asset_name: exits, "HEDGE": False})
        short_entries_df = pd.DataFrame({asset_name: False, "HEDGE": hedge_entries})
        short_exits_df = pd.DataFrame({asset_name: False, "HEDGE": hedge_exits})

        # === Backtest ===
        portfolio = vbt.Portfolio.from_signals(
            close=price_df,
            entries=entries_df,
            exits=exits_df,
            short_entries=short_entries_df,
            short_exits=short_exits_df,
            size=size_df.abs(),  # must be positive
            init_cash=init_cash,
            fees=0.001,
            sl_stop=0.05,
            tp_stop=0.20,
            freq="1D",
        )

        # === Plot ===
        plot_dir = os.path.join(plot_dir_root, asset_name)
        os.makedirs(plot_dir, exist_ok=True)
        plot_name = f"{asset_name}_{RUN['back_test_start'].strftime('%Y-%m-%d')}_to_{RUN['back_test_end'].strftime('%Y-%m-%d')}.html"
        fig = portfolio.value().vbt.plot(title=f"{asset_name} (with ES=F hedge)")
        fig.write_html(os.path.join(plot_dir, plot_name))

        # === Stats ===
        stats = portfolio.stats()
        stats = pd.DataFrame([stats])
        stats["asset"] = asset_name
        stats_list.append(stats)

        print(f"✅ Backtest complete: {asset_name}")

    # === Save master stats ===
    all_stats = pd.concat(stats_list, ignore_index=True)

    if os.path.exists(master_path):
        existing = pd.read_csv(master_path)
        combined = pd.concat([existing, all_stats], ignore_index=True)
        combined.to_csv(master_path, index=False)
    else:
        all_stats.to_csv(master_path, index=False)

    print("📊 All backtests complete. Master CSV updated.")


if __name__ == "__main__":
    backtest_all_assets_with_volatility_hedge(run_conf, "backtest_data")

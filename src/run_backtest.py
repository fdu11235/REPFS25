import pandas as pd
import vectorbt as vbt

# 1. Load your CSV
df = pd.read_csv("predictions_data/NVDA.csv")
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
    # slippage=0.001,
)

# 6. Show results
print(portfolio.stats(settings=dict(freq="1D")))
portfolio.plot().show()

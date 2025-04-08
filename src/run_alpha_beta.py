import pandas as pd
from config.config import RUN
import matplotlib.pyplot as plt


df = pd.read_csv(
    "processed_market_data/%sfinal_processed_data.csv"
    % RUN["folder"].replace("/", "_"),
    index_col=0,
)

"""
    .
    .   HOLD
    |
----------  beta

   BUY

----------  alpha
    |
    |
  ----- HOLD
    |
    |
----------  -alpha

   SELL

----------  -beta
    |
    .   HOLD
    .

"""

intl_hold = 0.85  # marks the threshold of hold strip
intl_buy_sell = 0.997  # marks the buy/sell upper/lower limits
# Plot histogram
plt.hist(
    df["pct_change"], bins=100, edgecolor="black"
)  # Adjust the number of bins as needed
plt.title("Histogram of DataFrame Column")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show()
print(df["pct_change"].value_counts())
alpha = df["pct_change"].abs().quantile(intl_hold)
beta = (
    df["pct_change"].abs().quantile(intl_buy_sell)
)  # to ignore the outliers in the beta computation

print("alpha: %.4f" % alpha)
print("beta: %.4f" % beta)

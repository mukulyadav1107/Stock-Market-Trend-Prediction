
import pandas as pd
import numpy as np

print("=" * 55)
print("RELIANCE STOCK MARKET - FEATURE ENGINEERING")
print("=" * 55)

# 1. Load cleaned dataset
data = pd.read_csv("reliance_cleaned_data.csv")

print("\nOriginal dataset shape:", data.shape)

# 2. Convert Date column
data["Date"] = pd.to_datetime(data["Date"])

# 3. Sort data chronologically
data = data.sort_values(by="Date").reset_index(drop=True)

# 4. Daily Returns
data["Daily_Return"] = data["Close"].pct_change()

# 5. Moving Averages
data["MA_5"] = data["Close"].rolling(window=5).mean()
data["MA_10"] = data["Close"].rolling(window=10).mean()
data["MA_20"] = data["Close"].rolling(window=20).mean()

# 6. Volatility
data["Volatility"] = (
    data["Daily_Return"].rolling(window=10).std()
)

# 7. RSI (14-day)
delta = data["Close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.ewm(
    alpha=1/14,
    min_periods=14,
    adjust=False
).mean()

avg_loss = loss.ewm(
    alpha=1/14,
    min_periods=14,
    adjust=False
).mean()

rs = avg_gain / avg_loss

data["RSI"] = 100 - (100 / (1 + rs))

# Handle zero-loss cases
data.loc[
    (avg_loss == 0) & (avg_gain > 0),
    "RSI"
] = 100

data.loc[
    (avg_loss == 0) & (avg_gain == 0),
    "RSI"
] = 50

# 8. MACD
ema_12 = data["Close"].ewm(
    span=12,
    adjust=False
).mean()

ema_26 = data["Close"].ewm(
    span=26,
    adjust=False
).mean()

data["MACD"] = ema_12 - ema_26

# 9. MACD Signal Line
data["MACD_Signal"] = data["MACD"].ewm(
    span=9,
    adjust=False
).mean()

# 10. Create next-day UP/DOWN target
next_close = data["Close"].shift(-1)

data["Target"] = np.where(
    next_close.isna(),
    np.nan,
    (next_close > data["Close"]).astype(int)
)

# 11. Remove rows with missing values
data = data.dropna().reset_index(drop=True)

# 12. Convert target into integer
data["Target"] = data["Target"].astype(int)

# 13. Save final feature-engineered dataset
data.to_csv(
    "reliance_featured_data.csv",
    index=False
)

# 14. Display results
print("\nFEATURE ENGINEERING COMPLETED!")

print("\nFirst 5 records:")
print(data.head())

print("\nDataset shape:", data.shape)

print("\nAvailable columns:")
print(list(data.columns))

print("\nTarget distribution:")
print(data["Target"].value_counts())

print("\nTarget meaning:")
print("1 = UP")
print("0 = DOWN")

print("\nSaved file: reliance_featured_data.csv")

print("\n" + "=" * 55)
print("PROCESS COMPLETED SUCCESSFULLY!")
print("=" * 55)

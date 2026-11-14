
import yfinance as yf
import pandas as pd

print("Downloading Reliance Industries data...")

data = yf.download(
    "RELIANCE.NS",
    period="5y",
    interval="1d",
    auto_adjust=True,
    multi_level_index=False,
    progress=False
)

if data.empty:
    print("Download failed. Check your internet connection.")

else:
    # Keep required columns
    data = data[["Open", "High", "Low", "Close", "Volume"]]

    # Move Date from index into a column
    data = data.reset_index()

    # Remove missing records
    data = data.dropna()

    # Save dataset
    data.to_csv("reliance_stock_data.csv", index=False)

    print("\nDOWNLOAD SUCCESSFUL!")
    print("\nFirst 5 records:")
    print(data.head())

    print("\nDataset shape:", data.shape)
    print("\nColumns:", list(data.columns))
    print("\nSaved as reliance_stock_data.csv")

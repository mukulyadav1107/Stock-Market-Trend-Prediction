
import pandas as pd

# Load original dataset
data = pd.read_csv("reliance_stock_data.csv")

print("Original Dataset Shape:", data.shape)

# Convert Date column to datetime
data["Date"] = pd.to_datetime(data["Date"])

# Sort records by date
data = data.sort_values(by="Date")

# Remove duplicate records
data = data.drop_duplicates(subset="Date")

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Remove rows with missing values
data = data.dropna()

# Reset index
data = data.reset_index(drop=True)

# Save cleaned dataset
data.to_csv("reliance_cleaned_data.csv", index=False)

print("\nPreprocessing completed successfully!")

print("\nCleaned Dataset Shape:", data.shape)

print("\nFirst 5 Records:")
print(data.head())

print("\nSaved as reliance_cleaned_data.csv")

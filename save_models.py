
import pandas as pd
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

print("=" * 55)
print("SAVING TRAINED MACHINE LEARNING MODELS")
print("=" * 55)

# 1. Load feature-engineered dataset
data = pd.read_csv("reliance_featured_data.csv")

# 2. Select features
features = [
    "Open", "High", "Low", "Close", "Volume",
    "Daily_Return", "MA_5", "MA_10", "MA_20",
    "Volatility", "RSI", "MACD", "MACD_Signal"
]

X = data[features]
y = data["Target"]

# 3. Chronological train-test split
split_index = int(len(data) * 0.80)

X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]

# 4. Define models
models = {
    "Logistic_Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Decision_Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random_Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf"))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ])
}

# 5. Create models folder
os.makedirs("saved_models", exist_ok=True)

# 6. Train and save each model
for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    filename = f"saved_models/{name}.joblib"

    joblib.dump(model, filename)

    print(f"Saved: {filename}")

# 7. Save feature names
joblib.dump(
    features,
    "saved_models/feature_columns.joblib"
)

print("\n" + "=" * 55)
print("ALL FIVE MODELS SAVED SUCCESSFULLY!")
print("=" * 55)

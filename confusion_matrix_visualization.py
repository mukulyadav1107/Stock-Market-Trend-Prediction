
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Load dataset
data = pd.read_csv("reliance_featured_data.csv")

features = [
    "Open", "High", "Low", "Close", "Volume",
    "Daily_Return", "MA_5", "MA_10", "MA_20",
    "Volatility", "RSI", "MACD", "MACD_Signal"
]

X = data[features]
y = data["Target"]

# Chronological split
split_index = int(len(data) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Define models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, random_state=42
    ),

    "Random Forest": RandomForestClassifier(
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

# Create confusion matrix plots
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for i, (name, model) in enumerate(models.items()):

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    cm = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["DOWN", "UP"],
        yticklabels=["DOWN", "UP"],
        ax=axes[i]
    )

    axes[i].set_title(name)
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")

# Remove unused sixth plot
fig.delaxes(axes[5])

plt.suptitle(
    "Confusion Matrix Comparison of ML Models",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix_comparison.png",
    dpi=300
)

plt.show()

print("Confusion matrices generated successfully!")
print("Saved as confusion_matrix_comparison.png")


import pandas as pd
import matplotlib.pyplot as plt

# Load model comparison results
data = pd.read_csv("model_comparison.csv")

print("Generating model comparison graphs...")

# 1. Accuracy comparison
plt.figure(figsize=(10, 6))

plt.bar(data["Model"], data["Accuracy"] * 100)

plt.title("Accuracy Comparison of ML Models")
plt.xlabel("Machine Learning Algorithms")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=20)
plt.ylim(0, 100)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("accuracy_comparison.png")
plt.show()

# 2. F1-score comparison
plt.figure(figsize=(10, 6))

plt.bar(data["Model"], data["F1 Score"] * 100)

plt.title("F1 Score Comparison of ML Models")
plt.xlabel("Machine Learning Algorithms")
plt.ylabel("F1 Score (%)")
plt.xticks(rotation=20)
plt.ylim(0, 100)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("f1_comparison.png")
plt.show()

# 3. Precision and Recall comparison
plt.figure(figsize=(10, 6))

plt.plot(
    data["Model"],
    data["Precision"] * 100,
    marker="o",
    label="Precision"
)

plt.plot(
    data["Model"],
    data["Recall"] * 100,
    marker="s",
    label="Recall"
)

plt.title("Precision and Recall Comparison")
plt.xlabel("Machine Learning Algorithms")
plt.ylabel("Score (%)")
plt.xticks(rotation=20)
plt.ylim(0, 100)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("precision_recall_comparison.png")
plt.show()

print("\nALL GRAPHS GENERATED SUCCESSFULLY!")

print("\nSaved files:")
print("1. accuracy_comparison.png")
print("2. f1_comparison.png")
print("3. precision_recall_comparison.png")

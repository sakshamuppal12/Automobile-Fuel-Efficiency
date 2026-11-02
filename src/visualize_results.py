from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "outputs" / "cleaned_data"
PREDICTION_FILE = BASE_DIR / "outputs" / "test_predictions.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load cleaned dataset
csv_files = list(DATA_DIR.glob("part-*.csv"))

if not csv_files:
    raise FileNotFoundError("Cleaned dataset CSV not found.")

data = pd.concat(
    [pd.read_csv(file) for file in csv_files],
    ignore_index=True
)

# Load Random Forest predictions
predictions = pd.read_csv(PREDICTION_FILE)

print("Dataset records:", len(data))
print("Prediction records:", len(predictions))

# GRAPH 1: Actual vs Predicted MPG
plt.figure(figsize=(10, 6))

plt.scatter(
    predictions["comb_mpg"],
    predictions["prediction"],
    alpha=0.7
)

minimum = min(
    predictions["comb_mpg"].min(),
    predictions["prediction"].min()
)
maximum = max(
    predictions["comb_mpg"].max(),
    predictions["prediction"].max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="Ideal Prediction"
)

plt.xlabel("Actual Combined MPG")
plt.ylabel("Predicted Combined MPG")
plt.title("Actual vs Predicted Fuel Efficiency")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=300)
plt.close()

# GRAPH 2: Average MPG by Vehicle Class
class_average = (
    data.groupby("class")["comb_mpg"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 7))
class_average.plot(kind="bar")
plt.xlabel("Vehicle Class")
plt.ylabel("Average Combined MPG")
plt.title("Average Fuel Efficiency by Vehicle Type")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "average_mpg_by_vehicle_class.png",
    dpi=300
)
plt.close()

# GRAPH 3: Engine Size vs Fuel Efficiency
plt.figure(figsize=(10, 6))

plt.scatter(
    data["engine_displacement"],
    data["comb_mpg"],
    alpha=0.6
)

plt.xlabel("Engine Displacement (Litres)")
plt.ylabel("Combined Fuel Efficiency (MPG)")
plt.title("Engine Size vs Fuel Efficiency")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "engine_size_vs_mpg.png", dpi=300)
plt.close()

print("\n========== VISUALIZATION COMPLETED ==========")
print("All three graphs saved in:", OUTPUT_DIR)

from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "epa2021.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "cleaned_data"

spark = (
    SparkSession.builder
    .appName("AutomobileFuelEfficiencyPreprocessing")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .config("spark.hadoop.fs.defaultFS", "file:///")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Load dataset
df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(str(DATA_PATH))
)

print("\n========== ORIGINAL DATASET ==========")
print("Records:", df.count())
print("Columns:", len(df.columns))

# Select relevant features and target
selected_columns = [
    "engine_displacement",
    "no_cylinders",
    "no_gears",
    "transmission_desc",
    "drive_desc",
    "class",
    "fuel_usage_desc",
    "comb_mpg"
]

df = df.select(*selected_columns)

# Remove rows with missing selected values
df = df.dropna(subset=selected_columns)

# Remove empty categorical values
categorical_columns = [
    "transmission_desc",
    "drive_desc",
    "class",
    "fuel_usage_desc"
]

for column in categorical_columns:
    df = df.filter(trim(col(column)) != "")

# Remove invalid target values
df = df.filter(col("comb_mpg") > 0)

# Remove duplicate rows


print("\n========== CLEANED DATASET ==========")
print("Records:", df.count())
print("Columns:", len(df.columns))

print("\n========== CLEANED SCHEMA ==========")
df.printSchema()

print("\n========== CLEANED DATA PREVIEW ==========")
df.show(10, truncate=False)

print("\n========== TARGET VARIABLE SUMMARY ==========")
df.select("comb_mpg").describe().show()

# Save cleaned dataset
df.write.mode("overwrite").option("header", True).csv(
    str(OUTPUT_PATH)
)

print("\nCleaned dataset saved at:", OUTPUT_PATH)

spark.stop()

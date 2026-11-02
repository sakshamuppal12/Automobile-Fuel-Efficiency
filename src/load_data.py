
import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "epa2021.csv"

# Create Spark Session
spark = (
    SparkSession.builder
    .appName("AutomobileFuelEfficiency")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .config("spark.hadoop.fs.defaultFS", "file:///")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("\n========== APACHE SPARK SESSION ==========")
print("Spark Version:", spark.version)
print("Application Name:", spark.sparkContext.appName)
print("Master:", spark.sparkContext.master)

# Load CSV into Spark DataFrame
df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .option("mode", "PERMISSIVE")
    .csv(str(DATA_PATH))
)

print("\n========== DATASET INFORMATION ==========")

print("Total Records:", df.count())
print("Total Columns:", len(df.columns))

print("\n========== SCHEMA ==========")
df.printSchema()

print("\n========== FIRST 10 RECORDS ==========")
df.show(10, truncate=False)

# Check missing values
print("\n========== MISSING VALUE ANALYSIS ==========")

missing_values = df.select([
    count(
        when(col(c).isNull(), c)
    ).alias(c)
    for c in df.columns
])

missing_values.show(truncate=False)

# Check duplicate records
print("\n========== DUPLICATE ANALYSIS ==========")

duplicate_count = df.count() - df.dropDuplicates().count()
print("Duplicate Records:", duplicate_count)

print("\nSpark DataFrame loaded successfully!")

spark.stop()


from pathlib import Path
import json

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "outputs" / "cleaned_data"
MODEL_PATH = BASE_DIR / "models"
OUTPUT_PATH = BASE_DIR / "outputs"

MODEL_PATH.mkdir(exist_ok=True)
OUTPUT_PATH.mkdir(exist_ok=True)

# --------------------------------------------------
# SPARK SESSION
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("AutomobileFuelEfficiencyML")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .config("spark.hadoop.fs.defaultFS", "file:///")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:

    # --------------------------------------------------
    # LOAD CLEANED DATASET
    # --------------------------------------------------

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(DATA_PATH))
    )

    print("\n========== DATASET ==========")
    print("Total records:", df.count())

    # --------------------------------------------------
    # FEATURE SELECTION
    # --------------------------------------------------

    numeric_features = [
        "engine_displacement",
        "no_cylinders",
        "no_gears"
    ]

    categorical_features = [
        "transmission_desc",
        "drive_desc",
        "class",
        "fuel_usage_desc"
    ]

    # --------------------------------------------------
    # CATEGORICAL ENCODING
    # --------------------------------------------------

    indexers = [
        StringIndexer(
            inputCol=c,
            outputCol=c + "_index",
            handleInvalid="keep"
        )
        for c in categorical_features
    ]

    encoder = OneHotEncoder(
        inputCols=[c + "_index" for c in categorical_features],
        outputCols=[c + "_encoded" for c in categorical_features]
    )

    # --------------------------------------------------
    # FEATURE ASSEMBLY
    # --------------------------------------------------

    assembler = VectorAssembler(
        inputCols=numeric_features + [
            c + "_encoded" for c in categorical_features
        ],
        outputCol="features"
    )

    # --------------------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------------------

    train_data, test_data = df.randomSplit(
        [0.8, 0.2],
        seed=42
    )

    print("\n========== TRAIN TEST SPLIT ==========")
    print("Training records:", train_data.count())
    print("Testing records:", test_data.count())

    # --------------------------------------------------
    # EVALUATION METRICS
    # --------------------------------------------------

    evaluator_rmse = RegressionEvaluator(
        labelCol="comb_mpg",
        predictionCol="prediction",
        metricName="rmse"
    )

    evaluator_mae = RegressionEvaluator(
        labelCol="comb_mpg",
        predictionCol="prediction",
        metricName="mae"
    )

    evaluator_r2 = RegressionEvaluator(
        labelCol="comb_mpg",
        predictionCol="prediction",
        metricName="r2"
    )

    results = {}

    # --------------------------------------------------
    # MACHINE LEARNING MODELS
    # --------------------------------------------------

    models = {
        "Linear Regression": LinearRegression(
            featuresCol="features",
            labelCol="comb_mpg",
            maxIter=100,
            regParam=0.1
        ),

        "Random Forest": RandomForestRegressor(
            featuresCol="features",
            labelCol="comb_mpg",
            numTrees=100,
            maxDepth=10,
            seed=42
        )
    }

    # --------------------------------------------------
    # MODEL TRAINING AND EVALUATION
    # --------------------------------------------------

    for model_name, algorithm in models.items():

        print(f"\n========== {model_name.upper()} ==========")

        pipeline = Pipeline(
            stages=indexers + [encoder, assembler, algorithm]
        )

        fitted_model = pipeline.fit(train_data)

        predictions = fitted_model.transform(test_data)

        # Calculate evaluation metrics
        rmse = evaluator_rmse.evaluate(predictions)
        mae = evaluator_mae.evaluate(predictions)
        r2 = evaluator_r2.evaluate(predictions)

        results[model_name] = {
            "RMSE": float(rmse),
            "MAE": float(mae),
            "R2": float(r2)
        }

        print("RMSE:", round(rmse, 4))
        print("MAE:", round(mae, 4))
        print("R2 Score:", round(r2, 4))

        print("\nSample predictions:")

        predictions.select(
            "comb_mpg",
            "prediction"
        ).show(10, truncate=False)

        # --------------------------------------------------
        # SAVE RANDOM FOREST TEST PREDICTIONS
        # --------------------------------------------------

        
        if model_name == "Random Forest":

            prediction_output = predictions.select(
                "engine_displacement",
                "no_cylinders",
                "no_gears",
                "transmission_desc",
                "drive_desc",
                "class",
                "fuel_usage_desc",
                "comb_mpg",
                "prediction"
            ).toPandas()

            prediction_output.to_csv(
                OUTPUT_PATH / "test_predictions.csv",
                index=False
            )

            print(
                "\nTest predictions saved successfully!"
            )
            print(
                "Total prediction records:",
                len(prediction_output)
            )
            print(
                "Prediction columns:",
                prediction_output.columns.tolist()
            )


        # --------------------------------------------------
        # SAVE TRAINED MODEL PIPELINE
        # --------------------------------------------------

        safe_name = model_name.lower().replace(" ", "_")

        fitted_model.write().overwrite().save(
            str(MODEL_PATH / safe_name)
        )

        print(f"{model_name} model saved successfully.")

    # --------------------------------------------------
    # SAVE MODEL COMPARISON RESULTS
    # --------------------------------------------------

    with open(
        OUTPUT_PATH / "model_results.json",
        "w"
    ) as file:

        json.dump(results, file, indent=4)

    print("\n========== MODEL COMPARISON ==========")

    for name, metrics in results.items():
        print(name, metrics)

    print("\nResults saved to outputs/model_results.json")
    print("Trained models saved in models/")
    print("Predictions saved in outputs/test_predictions.csv")

finally:
    spark.stop()


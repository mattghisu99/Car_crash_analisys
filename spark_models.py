from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import os
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql.functions import col


def initialize_spark_session(app_name="Crash Car Session", driver_mem="8g", exec_mem="8g") -> SparkSession:
    venv_python_path = r".venv/Scripts/python.exe"
    if os.path.exists(venv_python_path):
        os.environ["PYSPARK_PYTHON"] = venv_python_path
        os.environ["PYSPARK_DRIVER_PYTHON"] = venv_python_path

    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", driver_mem) \
        .config("spark.executor.memory", exec_mem) \
        .getOrCreate()
    return spark


def load_data(spark: SparkSession, file_path: str) -> DataFrame:
    data = spark.read.csv(file_path, header=True, inferSchema=True)
    return data


def apply_string_indexing(data: DataFrame, target_col: str = 'FIRST_CRASH_TYPE') -> (DataFrame, PipelineModel):

    categorical_cols = [
        "POSTED_SPEED_LIMIT", "TRAFFIC_CONTROL_DEVICE", "DEVICE_CONDITION",
        "WEATHER_CONDITION", "LIGHTING_CONDITION", "CRASH_TYPE", "TRAFFICWAY_TYPE",
        "ALIGNMENT", "ROADWAY_SURFACE_COND", "ROAD_DEFECT", "REPORT_TYPE",
        "DAMAGE", "PRIM_CONTRIBUTORY_CAUSE", "STREET_DIRECTION", "NUM_UNITS",
        "MOST_SEVERE_INJURY", "INJURIES_TOTAL", "CRASH_HOUR", "CRASH_DAY_OF_WEEK",
        "CRASH_MONTH", "YEAR"
    ]

    existing_cols = [c for c in categorical_cols + [target_col] if c in data.columns]

    indexers = [
        StringIndexer(inputCol=c, outputCol=c + "_index", handleInvalid="keep")
        for c in existing_cols
    ]

    pipeline = Pipeline(stages=indexers)
    indexer_model = pipeline.fit(data)
    transformed_data = indexer_model.transform(data)
    indexed_col_names = [c + "_index" for c in existing_cols]
    indexed_data = transformed_data.select(indexed_col_names)
    indexed_data = indexed_data.withColumnRenamed(f"{target_col}_index", "target")
    indexed_data = indexed_data.filter(col("target") != 6.0)

    return indexed_data, indexer_model

def vectorize_features(data: DataFrame, target_col: str = 'target') -> (DataFrame, list):
    feature_cols = [c for c in data.columns if c.endswith("_index") and c != f"{target_col}_index"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    vectorized_data = assembler.transform(data).select("features", target_col)
    original_feature_names = [c.replace("_index", "") for c in feature_cols]

    return vectorized_data, original_feature_names


def split_data(data: DataFrame, train_ratio: float = 0.7, seed: int = 42) -> (DataFrame, DataFrame):
    train_data, test_data = data.randomSplit([train_ratio, 1 - train_ratio], seed=seed)
    return train_data, test_data

def train_logistic_regression_cv(train_data: DataFrame, test_data: DataFrame, num_folds: int = 3) -> float:
    lr = LogisticRegression(labelCol="target", featuresCol="features", maxIter=200)

    param_grid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.01, 0.1]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
        .build()

    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="target", predictionCol="prediction", metricName="accuracy"
    )

    cv = CrossValidator(
        estimator=lr,
        estimatorParamMaps=param_grid,
        evaluator=evaluator_acc,
        numFolds=num_folds,
        seed=42
    )

    cv_model = cv.fit(train_data)
    predictions = cv_model.transform(test_data)
    accuracy = evaluator_acc.evaluate(predictions)

    return accuracy


def train_random_forest_cv(train_data: DataFrame, test_data: DataFrame, feature_names: list,
                           num_folds: int = 3) -> float:
    rf = RandomForestClassifier(
        labelCol="target",
        featuresCol="features",
        numTrees=200,
        maxDepth=10,
        minInstancesPerNode=2,
        seed=42
    )

    evaluator = MulticlassClassificationEvaluator(
        labelCol="target", predictionCol="prediction", metricName="accuracy"
    )

    cv = CrossValidator(
        estimator=rf,
        estimatorParamMaps=[{}],
        evaluator=evaluator,
        numFolds=num_folds,
        seed=42
    )

    cv_model = cv.fit(train_data)

    predictions = cv_model.transform(test_data)
    accuracy = evaluator.evaluate(predictions)

    print("Esempio di previsioni Random Forest:")
    predictions.select("target", "prediction", "probability").show(truncate=False)

    best_rf_model = cv_model.bestModel
    plot_feature_importance(best_rf_model, feature_names)

    return accuracy

def plot_feature_importance(rf_model: RandomForestClassificationModel, feature_names: list):
    """
    Crea un grafico a barre per l'importanza delle feature di un modello Random Forest.
    """
    importances = rf_model.featureImportances.toArray()

    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(12, 8))
    bars = plt.barh(feature_importance_df["Feature"], feature_importance_df["Importance"], color="cornflowerblue")
    plt.xlabel("Importanza")
    plt.title("Importanza delle Feature - Random Forest")
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.6)

    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2, f" {width:.4f}", va='center', fontsize=9)

    plt.tight_layout()
    plt.show()
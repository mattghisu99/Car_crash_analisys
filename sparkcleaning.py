import os
import sys
import json
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StringType

#Funzioni di raggruppamento#
def group_units_py(x):
    if x is None: return 'Unknown'
    if x == 1:
        return '1'
    elif x == 2:
        return '2'
    elif 3 <= x <= 5:
        return '3-5'
    return 'Unknown'

def group_hour_py(hour):
    if hour is None: return 'Unknown'
    if 0 <= hour < 6:
        return '0-5'
    elif 6 <= hour < 12:
        return '6-11'
    elif 12 <= hour < 18:
        return '12-17'
    elif 18 <= hour < 24:
        return '18-23'
    return 'Unknown'

def group_speed_limit_py(limite):
    if limite is None: return 'unknown'
    if 0 <= limite < 30:
        return 'under 30'
    elif limite >= 30:
        return 'over 30'
    return 'unknown'

def group_injuries_py(inj):
    if inj is None: return None
    if inj == 0:
        return '0'
    elif inj == 1:
        return '1'
    elif inj >= 2:
        return '2 or more'
    return str(inj)



class SparkDataCleaner:
    def __init__(self, config_path: str, app_name="TrafficCrashesDataCleaning", master="local[*]"):
        print("Inizializzazione di SparkDataCleaner...")
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .master(master) \
            .getOrCreate()
        print(f"SparkSession '{app_name}' creata con successo.")

        with open(config_path, 'r') as f:
            self.grouping_config = json.load(f)
            print(f"Configurazione di raggruppamento caricata da: {config_path}")

        self._register_udfs()

    def _register_udfs(self):
        self.group_units_udf = F.udf(group_units_py, StringType())
        self.group_hour_udf = F.udf(group_hour_py, StringType())
        self.group_speed_limit_udf = F.udf(group_speed_limit_py, StringType())
        self.group_injuries_udf = F.udf(group_injuries_py, StringType())

    def stop(self):
        print("Chiusura della SparkSession.")
        self.spark.stop()

    def load_csv(self, file_path: str):
        try:
            df = self.spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)
            print(f"Dataset caricato con successo da: {file_path}")
            return df
        except Exception as e:
            print(f"Errore durante il caricamento del file Spark da '{file_path}': {e}")
            return None

    def add_year_column(self, df):
        return df.withColumn("YEAR", F.year(F.to_date(F.col("CRASH_DATE"))))

    def remove_columns(self, df, columns_to_remove):
        return df.drop(*columns_to_remove)

    def replace_values(self, df, replacement_dict):
        df_processed = df
        for col_name, replacements in replacement_dict.items():
            if col_name == "CRASH_RECORD_ID" and replacements == "__index__":
                df_processed = df_processed.withColumn("CRASH_RECORD_ID", F.monotonically_increasing_id())
            else:
                case_when_expr = F
                for old_val, new_val in replacements.items():
                    case_when_expr = case_when_expr.when(F.col(col_name) == old_val, new_val)
                df_processed = df_processed.withColumn(col_name, case_when_expr.otherwise(F.col(col_name)))
        return df_processed

    def group_crash_data(self, df):
        df_processed = df
        for column, config in self.grouping_config.items():
            if column in df_processed.columns:
                case_when_expr = F
                for new_group, original_values in config['mapping'].items():
                    case_when_expr = case_when_expr.when(F.col(column).isin(original_values), new_group)
                default_value = config.get('default')
                if default_value is not None:
                    case_when_expr = case_when_expr.otherwise(default_value)
                else:
                    case_when_expr = case_when_expr.otherwise(F.lit(None))
                df_processed = df_processed.withColumn(column, case_when_expr)


        custom_logic_mapping = {
            'NUM_UNITS': self.group_units_udf,
            'CRASH_HOUR': self.group_hour_udf,
            'POSTED_SPEED_LIMIT': self.group_speed_limit_udf,
            'INJURIES_TOTAL': self.group_injuries_udf
        }
        for column, grouping_udf in custom_logic_mapping.items():
            if column in df_processed.columns:
                df_processed = df_processed.withColumn(column, grouping_udf(F.col(column)))

        return df_processed

    def save_cleaned_data(self, df, output_path):
        print(f"Salvataggio del DataFrame pulito in: {output_path}")
        df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)



if __name__ == "__main__":
    venv_python_path = sys.executable
    os.environ["PYSPARK_PYTHON"] = venv_python_path
    os.environ["PYSPARK_DRIVER_PYTHON"] = venv_python_path

    config_file_path = 'group_confi.json'
    file_path = 'dataset/Traffic_Crashes_-_Crashes_20250406.csv'
    output_path = "dataset_spark/Traffic_Crashes_cleaned"

    cleaner = SparkDataCleaner(config_path=config_file_path)

    crashes_df = cleaner.load_csv(file_path=file_path)

    columns_to_remove = ['CRASH_DATE_EST_I', 'LANE_CNT', 'INTERSECTION_RELATED_I', 'NOT_RIGHT_OF_WAY_I',
                         'HIT_AND_RUN_I', 'SEC_CONTRIBUTORY_CAUSE', 'PHOTOS_TAKEN_I', 'STATEMENTS_TAKEN_I',
                         'DOORING_I', 'WORK_ZONE_I', 'WORK_ZONE_TYPE', 'WORKERS_PRESENT_I', 'INJURIES_UNKNOWN',
                         'LOCATION', 'INJURIES_INCAPACITATING', 'INJURIES_FATAL', 'INJURIES_NON_INCAPACITATING',
                         'INJURIES_REPORTED_NOT_EVIDENT', 'INJURIES_NO_INDICATION']
    crashes_df = cleaner.remove_columns(crashes_df, columns_to_remove)

    crashes_df = cleaner.add_year_column(crashes_df)
    crashes_df = crashes_df.drop("CRASH_DATE")

    replacements = {
        "CRASH_DAY_OF_WEEK": {1: 'Sunday', 2: 'Monday', 3: 'Tuesday', 4: 'Wednesday', 5: 'Thursday',
                              6: 'Friday', 7: 'Saturday'},
        "CRASH_MONTH": {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July',
                        8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'},
        "CRASH_RECORD_ID": "__index__"
    }
    crashes_df = cleaner.replace_values(crashes_df, replacements)

    crashes_df = cleaner.group_crash_data(crashes_df)

    cleaner.save_cleaned_data(crashes_df, output_path)

    cleaner.stop()

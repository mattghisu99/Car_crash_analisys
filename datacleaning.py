import pandas as pd
import os
import json



def load_csv(file_path: str):
    try:
        data = pd.read_csv(file_path)
        print(f"Dataset caricato con successo da: {file_path}")
        return data
    except FileNotFoundError:
        print(f"Errore: il file non è stato trovato al percorso '{file_path}'")
        return None


class CrashDataPreprocessor:
    def __init__(self, dataframe: pd.DataFrame, config_path: str):
        self.data = dataframe
        print("PreProcessing iniziato con successo.")
        with open(config_path, 'r') as f:
            self.grouping_config = json.load(f)
        print(f"Configurazione di raggruppamento caricata da: {config_path}")

    def add_year_column(self):
        self.data['CRASH_DATE'] = pd.to_datetime(self.data['CRASH_DATE'], errors='coerce')
        self.data['YEAR'] = self.data['CRASH_DATE'].dt.year

    def remove_columns(self, columns_to_remove):
        cols = [col for col in columns_to_remove if col in self.data.columns]
        self.data.drop(columns=cols, inplace=True)

    def replace_values(self, replacement_dict):
        for col, replacements in replacement_dict.items():
            if replacements == "__index__":
                self.data.reset_index(drop=True, inplace=True)
                self.data[col] = self.data.index
            else:
                self.data[col] = self.data[col].replace(replacements)

    def group_crash_data(self):
        for column, config in self.grouping_config.items():
            inverted_map = {
                original_value: new_group
                for new_group, original_values in config['mapping'].items()
                for original_value in original_values
            }
            mapped_series = self.data[column].map(inverted_map)
            default_value = config.get('default')
            if default_value is not None:
                self.data[column] = mapped_series.fillna(default_value)
            else:
                self.data[column] = mapped_series

        custom_logic_mapping = {
            'NUM_UNITS': group_units,
            'CRASH_HOUR': group_hour,
            'POSTED_SPEED_LIMIT': group_speed_limit,
            'INJURIES_TOTAL': group_injuries
        }

        for column, grouping_function in custom_logic_mapping.items():
            self.data[column] = self.data[column].apply(grouping_function)

    def save_cleaned_data(self, output_path, index=False):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.data.to_csv(output_path, index=index)
        print("Nuovo dataset pulito salvato correttamente")


### === FUNZIONI DI RAGGRUPPAMENTO ESTERNE === ###
def group_units(x):
    if x == 1:
        return '1'
    elif x == 2:
        return '2'
    elif 3 <= x <= 5:
        return '3-5'
    return 'Unknown'


def group_hour(hour):
    if 0 <= hour < 6:
        return '0-5'
    elif 6 <= hour < 12:
        return '6-11'
    elif 12 <= hour < 18:
        return '12-17'
    return '18-23'


def group_speed_limit(limite):
    if 0 <= limite < 30:
        return 'under 30'
    elif limite >= 30:
        return 'over 30'
    return 'unknown'


def group_injuries(inj):
    if inj == 0:
        return '0'
    elif inj == 1:
        return '1'
    elif inj >= 2:
        return '2 or more'
    return str(inj)


if __name__ == "__main__":
    file_path = 'dataset/Traffic_Crashes_-_Crashes_20250406.csv'
    config_file_path = 'group_confi.json'

    crashes_df = load_csv(file_path=file_path)

    preprocessor = CrashDataPreprocessor(crashes_df, config_path=config_file_path)

    preprocessor.add_year_column()

    preprocessor.remove_columns([
        'CRASH_DATE', 'CRASH_DATE_EST_I', 'LANE_CNT', 'INTERSECTION_RELATED_I',
        'NOT_RIGHT_OF_WAY_I', 'HIT_AND_RUN_I', 'SEC_CONTRIBUTORY_CAUSE',
        'PHOTOS_TAKEN_I', 'STATEMENTS_TAKEN_I', 'DOORING_I', 'WORK_ZONE_I',
        'WORK_ZONE_TYPE', 'WORKERS_PRESENT_I', 'INJURIES_UNKNOWN', 'LOCATION',
        'INJURIES_INCAPACITATING', 'INJURIES_FATAL', 'INJURIES_NON_INCAPACITATING',
        'INJURIES_REPORTED_NOT_EVIDENT', 'INJURIES_NO_INDICATION'
    ])

    preprocessor.replace_values({
        "CRASH_DAY_OF_WEEK": {
            1: 'Sunday', 2: 'Monday', 3: 'Tuesday', 4: 'Wednesday',
            5: 'Thursday', 6: 'Friday', 7: 'Saturday'
        },
        "CRASH_MONTH": {
            1: 'January', 2: 'February', 3: 'March', 4: 'April',
            5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September',
            10: 'October', 11: 'November', 12: 'December'
        },
        "CRASH_RECORD_ID": "__index__"
    })

    preprocessor.group_crash_data()

    preprocessor.save_cleaned_data("dataset/Traffic_Crashes_cleaned.csv")
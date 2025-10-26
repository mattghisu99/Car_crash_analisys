import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.multiclass import OutputCodeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_crash_data(filepath, columns_to_keep):
    missing_value_formats = ["n.a.", "na", "NA", "--", "NaN", "None", "NULL", "", " "]
    full_data = pd.read_csv(filepath, low_memory=False, na_values=missing_value_formats)

    actual_columns_present = [col for col in columns_to_keep if col in full_data.columns]
    data = full_data[actual_columns_present].copy()
    return data

def clean_crash_target_and_separate_xy(data, target_column):
    data_to_process = data.copy()
    # Gestione dei valori mancanti specificamente nella colonna target
    nan_in_target_count = data_to_process[target_column].isnull().sum()
    if nan_in_target_count > 0:
        print(f"Trovati {nan_in_target_count} valori mancanti nel target. Le righe corrispondenti verranno rimosse.")
        data_to_process.dropna(subset=[target_column], inplace=True)
    # Separazione in features (X) e target (y)
    y = data_to_process[target_column]
    X = data_to_process.drop(columns=[target_column])
    return X, y

def split_data(X, y, test_size=0.3, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


class CrashDataPreprocessor:
    def __init__(self):
        self.imputer = None
        self.label_encoder = None

    def _align_crash_columns(self, train_df, test_df):
        train_cols, test_cols = set(train_df.columns), set(test_df.columns)
        missing_in_test = train_cols - test_cols
        for c in missing_in_test:
            test_df[c] = 0
        missing_in_train = test_cols - train_cols
        test_df.drop(columns=list(missing_in_train), inplace=True)
        return test_df[train_df.columns]

    def prepare_crash_data(self, X_train, X_test, y_train, y_test, impute_strategy_cat='most_frequent'):
        X_train_proc, X_test_proc = X_train.copy(), X_test.copy()

        for col in X_train_proc.columns:
            if X_train_proc[col].dtype != 'object':
                X_train_proc[col] = X_train_proc[col].astype(str)
                X_test_proc[col] = X_test_proc[col].astype(str)

        ## Imputazione valori mancanti ##
        self.imputer = SimpleImputer(strategy=impute_strategy_cat)
        X_train_proc = pd.DataFrame(self.imputer.fit_transform(X_train_proc), columns=X_train_proc.columns,
                                    index=X_train_proc.index)
        X_test_proc = pd.DataFrame(self.imputer.transform(X_test_proc), columns=X_test_proc.columns,
                                   index=X_test_proc.index)
        ## One-hot-encoding##
        X_train_proc = pd.get_dummies(X_train_proc, columns=X_train_proc.columns, dummy_na=False)
        X_test_proc = pd.get_dummies(X_test_proc, columns=X_test_proc.columns, dummy_na=False)

        X_test_proc = self._align_crash_columns(X_train_proc, X_test_proc)

        ## Label-Encoding ##
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)

        return X_train_proc, X_test_proc, y_train_encoded, y_test_encoded, self.label_encoder



class ECOCClassifier:
    def __init__(self, base_estimator_instance, code_size=1.5, random_state=42, n_jobs=-1):
        self.model = OutputCodeClassifier(estimator=base_estimator_instance, code_size=code_size,
                                          random_state=random_state, n_jobs=n_jobs)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test, class_names):
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, target_names=class_names, zero_division=0)
        conf_matrix = confusion_matrix(y_test, predictions)
        print(f"Accuratezza: {accuracy:.4f}")
        print("\nClassification Report:\n", report)
        return conf_matrix

    def plot_confusion_matrix(self, conf_matrix, class_names):
        plt.figure(figsize=(12, 10))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Matrice di Confusione');
        plt.ylabel('Classe Vera');
        plt.xlabel('Classe Predetta')
        plt.tight_layout();
        plt.show()

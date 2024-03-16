# Databricks notebook source
pip install pyod


# COMMAND ----------

import json
import numpy as np
import pandas as pd
from pyod.models.iforest import IForest
from pyspark.sql import SparkSession
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from scipy import stats


class CustomParameters:
    def __init__(self, n_trees=500, max_samples=0.2, max_features=1.0, bootstrap=False, random_state=42, verbose=0, n_jobs=-1):
        self.n_trees = n_trees
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs


class AlgorithmArgs:
    def __init__(self, executionType, dataInput, dataOutput, customParameters):
        self.executionType = executionType
        self.dataInput = dataInput
        self.dataOutput = dataOutput
        self.customParameters = customParameters

    @staticmethod
    def from_json_string(json_string):
        args_dict = json.loads(json_string)
        custom_parameters_dict = args_dict.get("customParameters", {})
        custom_parameters = CustomParameters(**custom_parameters_dict)
        return AlgorithmArgs(args_dict["executionType"], args_dict["dataInput"], args_dict["dataOutput"], custom_parameters)


def set_random_state(config):
    seed = config.customParameters.random_state
    import random
    random.seed(seed)
    np.random.seed(seed)


def load_data(config):
    spark = SparkSession.builder.getOrCreate()
    df = spark.sql("SELECT * FROM `hive_metastore`.`yahoo_preprocessed`.`a1_benchmark`")
    pandas_df = df.toPandas()

    # Drop rows with missing values in 'value' column
    pandas_df.dropna(subset=['value'], inplace=True)

    # Convert 'value' column to numeric (assuming it contains numerical data)
    pandas_df['value'] = pd.to_numeric(pandas_df['value'], errors='coerce')

    # Drop rows with NaN in the 'value' column
    pandas_df.dropna(subset=['value'], inplace=True)

    # Remove outliers using z-score method on 'value' column
    z_scores = np.abs(stats.zscore(pandas_df['value']))
    pandas_df = pandas_df[z_scores < 3]

    # Drop the 'timestamp' column as it is not relevant for anomaly detection
    pandas_df.drop('timestamp', axis=1, inplace=True)

    data = pandas_df['value'].values
    is_anomaly = pandas_df['is_anomaly'].values

    # Feature scaling on 'value' column
    scaler = StandardScaler()
    data = scaler.fit_transform(data.reshape(-1, 1)).flatten()

    return data, is_anomaly


def main(config):
    set_random_state(config)
    data, is_anomaly = load_data(config)

    clf = IForest(
        n_estimators=config.customParameters.n_trees,
        max_samples=config.customParameters.max_samples,
        max_features=config.customParameters.max_features,
        bootstrap=config.customParameters.bootstrap,
        random_state=config.customParameters.random_state,
        verbose=config.customParameters.verbose,
        n_jobs=config.customParameters.n_jobs,
    )

    # Fine-tune hyperparameters using RandomizedSearchCV
    param_distributions = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_samples': [0.1, 0.2, 0.3, 0.4, 0.5],
        'contamination': [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05],
        'max_features': [0.5, 0.7, 0.9, 1.0],  # Adding more options for max_features
    }

    random_search = RandomizedSearchCV(clf, param_distributions=param_distributions, scoring='roc_auc', n_iter=20, cv=5, n_jobs=-1)
    random_search.fit(data.reshape(-1, 1))  # Reshaping data as per sklearn requirement

    clf = random_search.best_estimator_

    scores = clf.decision_scores_

    # Save scores to output file
    np.savetxt(config.dataOutput, scores, delimiter=",")

    # Calculate AUC-ROC score
    auc_roc_score = roc_auc_score(is_anomaly, scores)

    print("Best Parameters:", random_search.best_params_)
    print("Improved AUC-ROC Score:", auc_roc_score)

    # Plot scatter plot with anomalies highlighted
    anomalies = np.where(scores > np.percentile(scores, 95))[0]  # Anomalies are those with a higher score
    plt.scatter(range(len(scores)), scores, c='blue', label='Normal')  # Plot all samples in blue
    plt.scatter(anomalies, scores[anomalies], c='red', label='Anomaly')  # Plot anomalies in red
    plt.xlabel('Sample Index')
    plt.ylabel('Anomaly Score')
    plt.title('Anomaly Detection using Isolation Forest')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    json_string = '{"executionType": "execute", "dataInput": "", "dataOutput": "/Workspace/Shared/IForest_Yahoo_Dataset/output_scores.csv", "customParameters": {"n_trees": 500, "max_samples": 0.2}}'
    config = AlgorithmArgs.from_json_string(json_string)
    main(config)


# COMMAND ----------

import json
import numpy as np
import pandas as pd
from pyod.models.iforest import IForest
from pyspark.sql import SparkSession
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from scipy import stats

class CustomParameters:
    def __init__(self, n_trees=500, max_samples=0.2, max_features=1.0, bootstrap=False, random_state=42, verbose=0, n_jobs=-1):
        self.n_trees = n_trees
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs

class AlgorithmArgs:
    def __init__(self, executionType, dataInput, dataOutput, customParameters):
        self.executionType = executionType
        self.dataInput = dataInput
        self.dataOutput = dataOutput
        self.customParameters = customParameters

    @staticmethod
    def from_json_string(json_string):
        args_dict = json.loads(json_string)
        custom_parameters_dict = args_dict.get("customParameters", {})
        custom_parameters = CustomParameters(**custom_parameters_dict)
        return AlgorithmArgs(args_dict["executionType"], args_dict["dataInput"], args_dict["dataOutput"], custom_parameters)

def set_random_state(config):
    seed = config.customParameters.random_state
    import random
    random.seed(seed)
    np.random.seed(seed)

def load_data(config):
    spark = SparkSession.builder.getOrCreate()
    df = spark.sql("SELECT * FROM `hive_metastore`.`yahoo_preprocessed`.`a1_benchmark`")
    pandas_df = df.toPandas()

    # Drop rows with missing values in 'value' column
    pandas_df.dropna(subset=['value'], inplace=True)

    # Convert 'value' column to numeric (assuming it contains numerical data)
    pandas_df['value'] = pd.to_numeric(pandas_df['value'], errors='coerce')

    # Drop rows with NaN in the 'value' column
    pandas_df.dropna(subset=['value'], inplace=True)

    # Remove outliers using z-score method on 'value' column
    z_scores = np.abs(stats.zscore(pandas_df['value']))
    pandas_df = pandas_df[z_scores < 3]

    # Drop the 'timestamp' column as it is not relevant for anomaly detection
    pandas_df.drop('timestamp', axis=1, inplace=True)

    data = pandas_df['value'].values
    is_anomaly = pandas_df['is_anomaly'].values

    # Feature scaling on 'value' column
    scaler = StandardScaler()
    data = scaler.fit_transform(data.reshape(-1, 1)).flatten()

    return data, is_anomaly

def main(config):
    set_random_state(config)
    data, is_anomaly = load_data(config)

    clf = IForest(
        n_estimators=config.customParameters.n_trees,
        max_samples=config.customParameters.max_samples,
        max_features=config.customParameters.max_features,
        bootstrap=config.customParameters.bootstrap,
        random_state=config.customParameters.random_state,
        verbose=config.customParameters.verbose,
        n_jobs=config.customParameters.n_jobs,
    )

    param_distributions = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_samples': [0.1, 0.2, 0.3, 0.4, 0.5],
        'contamination': [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05],
        'max_features': [0.5, 0.7, 0.9, 1.0],
    }

    random_search = RandomizedSearchCV(clf, param_distributions=param_distributions, scoring='roc_auc', n_iter=20, cv=5, n_jobs=-1)
    random_search.fit(data.reshape(-1, 1))  # Reshaping data as per sklearn requirement

    clf = random_search.best_estimator_

    scores = clf.decision_scores_

    np.savetxt(config.dataOutput, scores, delimiter=",")

    auc_roc_score = roc_auc_score(is_anomaly, scores)

    print("Best Parameters:", random_search.best_params_)
    print("Improved AUC-ROC Score:", auc_roc_score)

    anomalies = np.where(scores > np.percentile(scores, 95))[0]
    plt.scatter(range(len(scores)), scores, c='blue', label='Normal')
    plt.scatter(anomalies, scores[anomalies], c='red', label='Anomaly')
    plt.xlabel('Sample Index')
    plt.ylabel('Anomaly Score')
    plt.title('Anomaly Detection using Isolation Forest')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    json_string = '{"executionType": "execute", "dataInput": "", "dataOutput": "/Workspace/Shared/IForest_Yahoo_Dataset/output_scores.csv", "customParameters": {"n_trees": 500, "max_samples": 0.2}}'
    config = AlgorithmArgs.from_json_string(json_string)
    main(config)


# COMMAND ----------

import numpy as np
import pandas as pd
from pyod.models.iforest import IForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from scipy import stats
import json
from pyspark.sql import SparkSession

class CustomParameters:
    def __init__(self, n_trees=500, max_samples=0.2, max_features=1.0, bootstrap=False, random_state=42, verbose=0, n_jobs=-1):
        self.n_trees = n_trees
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs

class AlgorithmArgs:
    def __init__(self, executionType, dataInput, dataOutput, customParameters, dataset):
        self.executionType = executionType
        self.dataInput = dataInput
        self.dataOutput = dataOutput
        self.customParameters = customParameters
        self.dataset = dataset

    @staticmethod
    def from_json_string(json_string):
        args_dict = json.loads(json_string)
        custom_parameters_dict = args_dict.get("customParameters", {})
        custom_parameters = CustomParameters(**custom_parameters_dict)
        return AlgorithmArgs(args_dict["executionType"], args_dict["dataInput"], args_dict["dataOutput"], custom_parameters, args_dict["dataset"])

def set_random_state(config):
    seed = config.customParameters.random_state
    import random
    random.seed(seed)
    np.random.seed(seed)

def load_data(config):
    spark = SparkSession.builder.getOrCreate()
    df = spark.sql("SELECT * FROM `hive_metastore`.`yahoo_preprocessed`.`a1_benchmark`")
    pandas_df = df.toPandas()

    # Drop rows with missing values in 'value' column
    pandas_df.dropna(subset=['value'], inplace=True)

    # Convert 'value' column to numeric (assuming it contains numerical data)
    pandas_df['value'] = pd.to_numeric(pandas_df['value'], errors='coerce')

    # Drop rows with NaN in the 'value' column
    pandas_df.dropna(subset=['value'], inplace=True)

    # Remove outliers using z-score method on 'value' column
    z_scores = np.abs(stats.zscore(pandas_df['value']))
    pandas_df = pandas_df[z_scores < 3]

    # Drop the 'timestamp' column as it is not relevant for anomaly detection
    pandas_df.drop('timestamp', axis=1, inplace=True)

    data = pandas_df['value'].values
    is_anomaly = pandas_df['is_anomaly'].values

    # Feature scaling on 'value' column
    scaler = StandardScaler()
    data = scaler.fit_transform(data.reshape(-1, 1)).flatten()

    return data, is_anomaly

def preprocess_data(config):
    spark = SparkSession.builder.getOrCreate()

    if config.dataset == "synthetic-cbf":
        table_name = "synthetic_datasets.cylinder_bell_funnel"
    elif config.dataset == "synthetic-sine":
        table_name = "synthetic_datasets.sine_platform"
    elif config.dataset == "synthetic-ecg":
        table_name = "synthetic_datasets.ecg_noise_10per1"
    elif config.dataset == "synthetic-sine-mean":
        table_name = "synthetic_datasets.sine_mean"
    elif config.dataset == "swat" and config.executionType == "train":
        table_name = "default.copy_of_swat_dataset_normal_v0_csv"
    elif config.dataset == "swat" and config.executionType == "execute":
        table_name = "swat_preprocessed.swat_2015"
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")

    ts_data = spark.sql(f"SELECT * FROM {table_name}")
    ts_data = ts_data.toPandas()

    if config.dataset == "swat" and config.executionType == "train":
        ts_data = ts_data.drop(columns='timestamp')
        ts_data.rename(columns={'Normal/Attack': 'is_anomaly'}, inplace=True)
        ts_data['is_anomaly'].replace({'Normal': 0, 'Attack': 1}, inplace=True)
        ts_data = ts_data.astype(float)
    elif config.dataset == "swat" and config.executionType == "execute":
        ts_data = ts_data.drop(columns='Timestamp')
        ts_data = ts_data.astype(float)

    return ts_data




def main(config):
    set_random_state(config)
    ts_data = preprocess_data(config)

    # ... (same as before)

if __name__ == "__main__":
    json_string = '{"executionType": "execute", "dataInput": "", "dataOutput": "/Workspace/Shared/IForest_Yahoo_Dataset/output_scores.csv", "customParameters": {"n_trees": 500, "max_samples": 0.2}, "dataset": "synthetic-cbf"}'
    config = AlgorithmArgs.from_json_string(json_string)
    main(config)


# ... (imports and class definitions remain the same)

def select_dataset():
    print("Available datasets:")
    print("1. synthetic-cbf")
    print("2. synthetic-sine")
    print("3. synthetic-ecg")
    print("4. synthetic-sine-mean")
    print("5. swat")

    dataset_option = int(input("Select a dataset (enter the number): "))
    dataset_options = [
        "synthetic-cbf",
        "synthetic-sine",
        "synthetic-ecg",
        "synthetic-sine-mean",
        "swat"
    ]

    if dataset_option < 1 or dataset_option > len(dataset_options):
        print("Invalid option selected. Exiting...")
        exit()

    return dataset_options[dataset_option - 1]

def main():
    execution_type = input("Enter execution type (train/execute): ")
    dataset = select_dataset()

    data_input = ""
    data_output = "/Workspace/Shared/IForest_Yahoo_Dataset/output_scores.csv"
    custom_parameters = {"n_trees": 500, "max_samples": 0.2}
    json_string = {
        "executionType": execution_type,
        "dataInput": data_input,
        "dataOutput": data_output,
        "customParameters": custom_parameters,
        "dataset": dataset
    }

    config = AlgorithmArgs.from_json_string(json.dumps(json_string))
    main(config)

if __name__ == "__main__":
    main()



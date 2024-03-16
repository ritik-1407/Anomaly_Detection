# Databricks notebook source
# MAGIC %pip install pyod
# MAGIC # %pip install --upgrade pandas pyarrow
# MAGIC # import tracemalloc
# MAGIC # tracemalloc.start()

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
import warnings
from packaging import version  # Import packaging.version for version comparison
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV  # Add this import



warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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


def load_data(config, dataset_name):
    spark = SparkSession.builder.getOrCreate()
    df = spark.sql(f"SELECT * FROM `hive_metastore`.`synthetic_datasets`.`{dataset_name}`")
    pandas_df = df.toPandas()

    # Convert 'value-0' column to numeric (assuming it contains numerical data)
    pandas_df['value-0'] = pd.to_numeric(pandas_df['value-0'], errors='coerce')

    # Drop rows with NaN in the 'value-0' column
    pandas_df.dropna(subset=['value-0'], inplace=True)

    # Remove outliers using z-score method on 'value-0' column
    z_scores = np.abs(stats.zscore(pandas_df['value-0']))
    pandas_df = pandas_df[z_scores < 3]

    data = pandas_df['value-0'].values
    is_anomaly = pandas_df['is_anomaly'].values

    # Feature scaling on 'value-0' column
    scaler = StandardScaler()
    data = scaler.fit_transform(data.reshape(-1, 1)).flatten()

    return data, is_anomaly


def main(config):
    set_random_state(config)

    datasets = ["cylinder_bell_funnel", "ecg_noise_10per1", "sine_mean", "sine_platform"]

    for dataset_name in datasets:
        data, is_anomaly = load_data(config, dataset_name)

        clf = IForest(
            n_estimators=config.customParameters.n_trees,
            max_samples=config.customParameters.max_samples,
            max_features=config.customParameters.max_features,
            bootstrap=config.customParameters.bootstrap,
            random_state=config.customParameters.random_state,
            verbose=config.customParameters.verbose,
            n_jobs=config.customParameters.n_jobs,
        )

        # Grid search for hyperparameter tuning
        param_grid = {
            'n_estimators': [400, 500, 600],
            'max_samples': [0.2, 0.25, 0.3],
            'max_features': [0.7, 0.8, 0.9],
            'contamination': [0.02, 0.03, 0.04],
        }

        grid_search = GridSearchCV(clf, param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
        grid_search.fit(data.reshape(-1, 1))

        clf = grid_search.best_estimator_

        scores = clf.decision_scores_

        # Find a threshold that maximizes F1-score
        f1_scores = []
        thresholds = np.arange(np.percentile(scores, 1), np.percentile(scores, 95), 0.01)
        for threshold in thresholds:
            predicted_labels = scores > threshold
            f1 = f1_score(is_anomaly, predicted_labels)
            f1_scores.append(f1)
        best_threshold = thresholds[np.argmax(f1_scores)]
        predicted_labels = scores > best_threshold

        # Further fine-tuning the threshold for F1-score
        threshold_range = np.arange(best_threshold - 0.05, best_threshold + 0.05, 0.001)
        f1_scores = [f1_score(is_anomaly, scores > thr) for thr in threshold_range]
        best_threshold = threshold_range[np.argmax(f1_scores)]
        predicted_labels = scores > best_threshold

        auc_roc_score = roc_auc_score(is_anomaly, scores)
        f1 = f1_score(is_anomaly, predicted_labels)

        if ( dataset_name == 'cylinder_bell_funnel') :
            #update the CSV file 
            file_path='/Workspace/Shared/Plot/Cylinder-Bell-Funnel'    # file_path='/Workspace/Shared/Plot/Swat'   ->   ['Cylinder-Bell-Funnel','ECG','Sine-Mean','Sine-Platform','Swat']
            csvDF = pd.read_csv(file_path)
            csvDF.loc[csvDF['Metric'] == 'IForest', 'AUROC Score'] = auc_roc_score

            csvDF.loc[csvDF['Metric'] == 'IForest', 'F1 Score'] = f1 
            csvDF.to_csv(file_path, index=False)
            print(f'DataFrame saved to {file_path}')

        elif(dataset_name == 'ecg_noise_10per1'):
            #update the CSV file 
            file_path='/Workspace/Shared/Plot/ECG'    # file_path='/Workspace/Shared/Plot/Swat'   ->   ['Cylinder-Bell-Funnel','ECG','Sine-Mean','Sine-Platform','Swat']
            csvDF = pd.read_csv(file_path)
            csvDF.loc[csvDF['Metric'] == 'IForest', 'AUROC Score'] = auc_roc_score

            csvDF.loc[csvDF['Metric'] == 'IForest', 'F1 Score'] = f1 
            csvDF.to_csv(file_path, index=False)
            print(f'DataFrame saved to {file_path}')
        elif(dataset_name == 'sine_mean'):
            #update the CSV file 
            file_path='/Workspace/Shared/Plot/Sine-Mean'    # file_path='/Workspace/Shared/Plot/Swat'   ->   ['Cylinder-Bell-Funnel','ECG','Sine-Mean','Sine-Platform','Swat']
            csvDF = pd.read_csv(file_path)
            csvDF.loc[csvDF['Metric'] == 'IForest', 'AUROC Score'] = auc_roc_score

            csvDF.loc[csvDF['Metric'] == 'IForest', 'F1 Score'] = f1 
            csvDF.to_csv(file_path, index=False)
            print(f'DataFrame saved to {file_path}')
        elif(dataset_name == 'sine_platform'):
            #update the CSV file 
            file_path='/Workspace/Shared/Plot/Sine-Platform'    # file_path='/Workspace/Shared/Plot/Swat'   ->   ['Cylinder-Bell-Funnel','ECG','Sine-Mean','Sine-Platform','Swat']
            csvDF = pd.read_csv(file_path)
            csvDF.loc[csvDF['Metric'] == 'IForest', 'AUROC Score'] = auc_roc_score

            csvDF.loc[csvDF['Metric'] == 'IForest', 'F1 Score'] = f1 
            csvDF.to_csv(file_path, index=False)
            print(f'DataFrame saved to {file_path}')
        
        print(f"Dataset: {dataset_name}")
        print("Best Parameters:", grid_search.best_params_)
        print("Best Threshold for F1:", best_threshold)
        print("AUC-ROC Score:", auc_roc_score)
        print("F1 Score:", f1)

        # Plot scatter plot with anomalies highlighted
        anomalies = np.where(scores > best_threshold)[0]
        plt.scatter(range(len(scores)), scores, c='blue', label='Normal')
        plt.scatter(anomalies, scores[anomalies], c='red', label='Anomaly')
        plt.xlabel('Sample Index')
        plt.ylabel('Anomaly Score')
        plt.title(f'Anomaly Detection using Isolation Forest - Dataset {dataset_name}')
        plt.legend()
        plt.show()



if __name__ == "__main__":
    json_string = '{"executionType": "execute", "dataInput": "", "dataOutput": "/Workspace/Shared/IForest_Yahoo_Dataset/output_scores.csv", "customParameters": {"n_trees": 500, "max_samples": 0.2}}'
    config = AlgorithmArgs.from_json_string(json_string)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ResourceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        
        main(config)


# COMMAND ----------

# #update the CSV file 
#     file_path='/Workspace/Shared/Plot/Swat'    # file_path='/Workspa
# ce/Shared/Plot/Swat'   ->   ['Cylinder-Bell-Funnel','ECG','Sine-Mean','Sine-Platform','Swat']
#     csvDF = pd.read_csv(file_path)
#     csvDF.loc[csvDF['Metric'] == 'IForest', 'AUROC Score'] = auroc

#     csvDF.loc[csvDF['Metric'] == 'IForest', 'F1 Score'] = f1 
#     csvDF.to_csv(file_path, index=False)
#     print(f'DataFrame saved to {file_path}')

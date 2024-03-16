# Databricks notebook source
# MAGIC %pip install pyod
# MAGIC %pip install --upgrade pandas pyarrow
# MAGIC import tracemalloc
# MAGIC tracemalloc.start()
# MAGIC %pip install --upgrade numba
# MAGIC %pip install --upgrade scipy
# MAGIC

# COMMAND ----------

import json
import numpy as np
import pandas as pd
from pyod.models.iforest import IForest
from pyspark.sql import SparkSession
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt

anomaly_scores = None
features = None
is_anomaly = None


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

def load_data(config):
    spark = SparkSession.builder.getOrCreate()
    ts_data = spark.sql("SELECT * FROM `hive_metastore`.`swat_preprocessed`.`swat_2015`")
    ts_data = ts_data.toPandas()

    # Drop rows with missing values in relevant columns
    relevant_columns = [
        'FIT101', 'LIT101', 'MV101', 'P101', 'P102', 'AIT201', 'AIT202', 'AIT203',
        'FIT201', 'MV201', 'P201', '_P202', 'P203', '_P204', 'P205', 'P206', 
        'DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302', '_MV303', 'MV304',
        'P301', 'P302', 'AIT401', 'AIT402', 'FIT401', 'LIT401', 'P401', 
        'P402', 'P403', 'P404', 'UV401', 'AIT501', 'AIT502', 'AIT503', 'AIT504',
        'FIT501', 'FIT502', 'FIT503', 'FIT504', 'P501', 'P502', 'PIT501', 
        'PIT502', 'PIT503', 'FIT601', 'P601', 'P602', 'P603', 'isAnomoly'
    ]
    ts_data.dropna(subset=relevant_columns, inplace=True)

    # Extract relevant columns for features
    features = ts_data[relevant_columns[:-1]].values

    # Feature scaling
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    is_anomaly = ts_data['isAnomoly'].values
    return features, is_anomaly

def main(config):
    features, is_anomaly = load_data(config)

    clf = IForest(
        n_estimators=config.customParameters.n_trees,
        max_samples=config.customParameters.max_samples,
        max_features=config.customParameters.max_features,
        bootstrap=config.customParameters.bootstrap,
        random_state=config.customParameters.random_state,
        verbose=config.customParameters.verbose,
        n_jobs=config.customParameters.n_jobs,
    )

    clf.fit(features)
    anomaly_scores = -clf.decision_function(features)
    return anomaly_scores, is_anomaly

def optimize_threshold(is_anomaly, anomaly_scores):
    best_f1 = 0
    best_threshold = 0
    for threshold in np.linspace(0, 1, num=90):
        predicted_labels = np.where(anomaly_scores > threshold, 1, 0)
        f1 = f1_score(is_anomaly, predicted_labels)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold

if __name__ == "__main__":
    json_string = '{"executionType": "execute", "dataInput": "", "dataOutput": "/Workspace/Shared/IForest_Yahoo_Dataset/output_scores.csv", "customParameters": {"n_trees": 800, "max_samples": 0.2}}'
    config = AlgorithmArgs.from_json_string(json_string)
    
    anomaly_scores, is_anomaly = main(config)
    
    best_threshold = optimize_threshold(is_anomaly, anomaly_scores)

    predicted_labels = np.where(anomaly_scores > best_threshold, 1, 0)
    roc_auc = roc_auc_score(is_anomaly, anomaly_scores)
    f1 = f1_score(is_anomaly, predicted_labels)

    #update the CSV file 
    file_path='/Workspace/Shared/Plot/Swat'
    csvDF = pd.read_csv(file_path)
    csvDF.loc[csvDF['Metric'] == 'IForest', 'AUROC Score'] = roc_auc  # Corrected 'auroc' to 'roc_auc'


    csvDF.loc[csvDF['Metric'] == 'IForest', 'F1 Score'] = f1 
    csvDF.to_csv(file_path, index=False)
    print(f'DataFrame saved to {file_path}')

    print("ROC AUC Score:", roc_auc)
    print("F1 Score:", f1)
    
    # Plot anomalies
    anomalies = np.where(anomaly_scores > np.percentile(anomaly_scores, 90))[0]
    plt.scatter(range(len(anomaly_scores)), anomaly_scores, c='blue', label='Normal')
    plt.scatter(anomalies, anomaly_scores[anomalies], c='red', label='Anomaly')
    plt.axhline(y=best_threshold, color='green', linestyle='--', label='Optimal Threshold')
    plt.xlabel('Sample Index')-
    plt.ylabel('Anomaly Score')
    plt.title('Anomaly Detection using Isolation Forest')
    plt.legend()
    plt.show()



# COMMAND ----------



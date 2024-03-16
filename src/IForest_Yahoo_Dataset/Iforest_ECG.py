# Databricks notebook source
# MAGIC %pip install pyod

# COMMAND ----------

import json
import numpy as np
import pandas as pd
from pyod.models.iforest import IForest
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import ParameterGrid

class CustomParameters:
    def __init__(self, n_trees=100, max_samples=None, max_features=1.0, bootstrap=False, random_state=42, verbose=0, n_jobs=1):
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
    df = spark.sql("SELECT * FROM `hive_metastore`.`default`.`ecg_1_csv`")
    pandas_df = df.toPandas()
    data = pandas_df.iloc[:, 1:-1].values
    labels = pandas_df.iloc[:, -1].values.astype(float)
    contamination = min(labels.sum() / len(labels), 0.5)
    contamination = np.nextafter(0, 0.5) if contamination == 0. else contamination
    return data, labels, contamination, pandas_df

def evaluate_model(config, data, labels):
    clf = IForest(
        contamination=config.customParameters.max_samples,
        n_estimators=config.customParameters.n_trees,
        max_samples=config.customParameters.max_samples or "auto",
        max_features=config.customParameters.max_features,
        bootstrap=config.customParameters.bootstrap,
        random_state=config.customParameters.random_state,
        verbose=config.customParameters.verbose,
        n_jobs=config.customParameters.n_jobs,
    )
    clf.fit(data)
    scores = clf.decision_scores_

    # Ground truth and predicted anomalies are the same
    predicted_labels = clf.predict(data)
    ground_truth = np.where(labels == 1.0)[0]  # Index of ground truth anomalies
    predicted_anomalies = np.where(predicted_labels == 1)[0]  # Index of predicted anomalies

    if np.array_equal(ground_truth, predicted_anomalies):
        print("Ground truth and predicted anomalies are the same.")
    else:
        print("Ground truth and predicted anomalies are different.")
        print("Ground truth anomalies:", ground_truth)
        print("Predicted anomalies:", predicted_anomalies)

    # Calculate evaluation metrics
    precision = precision_score(labels, predicted_labels)
    recall = recall_score(labels, predicted_labels)
    f1 = f1_score(labels, predicted_labels)
    auc_roc = roc_auc_score(labels, scores)

    # Print evaluation metrics
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("AUC-ROC:", auc_roc)

    return scores, predicted_labels

def main(config):
    set_random_state(config)
    data, labels, contamination, pandas_df = load_data(config)

    # Perform parameter search
    param_grid = {
        'n_trees': [50, 100, 200],
        'max_samples': [0.1, 0.2, 0.3],
        'max_features': [0.5, 0.8, 1.0]
    }
    best_scores = None
    best_predicted_labels = None
    best_config = None
    best_f1 = 0

    for params in ParameterGrid(param_grid):
        config.customParameters.n_trees = params['n_trees']
        config.customParameters.max_samples = params['max_samples']
        config.customParameters.max_features = params['max_features']

        scores, predicted_labels = evaluate_model(config, data, labels)

        f1 = f1_score(labels, predicted_labels)
        if f1 > best_f1:
            best_scores = scores
            best_predicted_labels = predicted_labels
            best_config = params
            best_f1 = f1

    # Display the dataset
    print(pandas_df)

    # Print best configuration and metrics
    print("Best Configuration:", best_config)
    print("Best F1-score:", best_f1)

    # Plot anomaly scores
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(best_scores)), best_scores, color='blue')
    plt.xlabel('Sample Index')
    plt.ylabel('Anomaly Score')
    plt.title('Anomaly Scores')
    plt.show()

    # Plot anomalous data with color separation
    plt.figure(figsize=(10, 4))
    plt.scatter(range(len(best_scores)), best_scores, c=np.where(best_predicted_labels == 1, 'red', 'blue'))
    plt.xlabel('Sample Index')
    plt.ylabel('Anomaly Score')
    plt.title('Anomalous Data with Color Separation')
    plt.show()

json_string = '{"executionType": "execute", "dataInput": "", "dataOutput": "/Workspace/Shared/IForest_Yahoo_Dataset/output_scores.csv", "customParameters": {"n_trees": 100, "max_samples": 0.1}}'
config = AlgorithmArgs.from_json_string(json_string)
main(config)


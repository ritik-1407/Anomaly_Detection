# Databricks notebook source
# MAGIC %pip install pyod
# MAGIC

# COMMAND ----------

import json
import numpy as np
import pandas as pd
from pyod.models.iforest import IForest

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
    #df = pd.read_csv(config.dataInput)
    df = spark.sql("select * from ").toPandas()
    df = df.astype(float)
    print(df)
    data = df.iloc[:, 1:-1].values
    labels = df.iloc[:, -2].values
    print("labels" +str(type(labels)))
    contamination = labels.sum() / len(labels)

    # Use the smallest positive float as contamination if there are no anomalies in the dataset
    contamination = np.nextafter(0, 1) if contamination == 0. else contamination
    return data, contamination

def main(config):
    set_random_state(config)
    data, contamination = load_data(config)

    clf = IForest(
        contamination=contamination,
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
    np.savetxt(config.dataOutput, scores, delimiter=",")
    anomalies = np.where(scores > 0.5)[0]  # Adjust the threshold as needed
    plt.scatter(range(len(data)), data, c='b', label='Normal')
    plt.scatter(anomalies, data[anomalies], c='r', label='Anomaly')
    plt.xlabel('Data Point')
    plt.ylabel('Data Value')
    plt.title('Anomaly Detection using Isolation Forest')
    plt.legend()
    plt.show()

json_string = '{"executionType": "execute", "dataInput": "", "dataOutput": "/Workspace/Shared/iForest_Yahoo_Dataset/output_scores.csv", "customParameters": {"n_trees": 100, "max_samples": 0.1}}'
config = AlgorithmArgs.from_json_string(json_string)
main(config)



# COMMAND ----------



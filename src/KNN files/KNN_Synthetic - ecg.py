# Databricks notebook source
pip install pyod

# COMMAND ----------

#!/usr/bin/env python3
import argparse
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, f1_score
from dataclasses import dataclass
from pyod.models.knn import KNN


@dataclass
class CustomParameters:
    n_neighbors: int = 11
    leaf_size: int = 30
    method: str = "largest"
    radius: float = 1.0
    distance_metric_order: int = 2
    n_jobs: int = 1
    algorithm: str = "auto"  # using default is fine
    distance_metric: str = "minkowski"  # using default is fine
    random_state: int = 42


class AlgorithmArgs(argparse.Namespace):
    @staticmethod
    def from_sys_args(json_string: str) -> 'AlgorithmArgs':
        args: dict = json.loads(json_string)
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)

def set_random_state(config: AlgorithmArgs) -> None:
    seed = config.customParameters.random_state
    import random
    random.seed(seed)
    np.random.seed(seed)

def load_data() -> np.ndarray:

        ts_data = spark.sql("select * from synthetic_datasets.ecg_noise_10per1")
        ts_data = ts_data.toPandas()
        ts_data = ts_data.drop(columns='timestamp')
        ts_data = ts_data.astype(float)
        #data = ts_data.iloc[:, 0].values
        #print("in load")
        #print(data)
        labels = ts_data.iloc[:, -1].values
        # print("labels",labels)
        # display(ts_data)
    
        #x_values = ts_data['value-0']
        x_values = ts_data.iloc[:, :-1]
        # print("x_values")
        # display(x_values)
        #ground_truth_values = ts_data['is_anomaly']  
        ground_truth_values = ts_data['is_anomaly']
        # Create a mask to filter out points where Ground_Truth is 1 (anomaly)
        anomaly_mask = (ground_truth_values == 1)

        contamination = labels.sum() / len(labels)
        data= pd.DataFrame(x_values)
        data['labels'] = labels

        #contamination= max(min(contamination, 1), 0.001)
        # Use smallest positive float as contamination if there are no anomalies in dataset
        contamination = np.nextafter(0, 1) if contamination == 0. else contamination
        return data, contamination
        
def plot_anomalies(data, scores, threshold):
    anomalies = np.where(scores > threshold)[0]
    normal = np.where(scores <= threshold)[0]
    true_positives = np.where((scores > threshold) & (data["labels"] == 1))[0]
    false_positives = np.where((scores > threshold) & (data["labels"] == 0))[0]

    plt.scatter(np.arange(len(data)), scores, c='b', label='Normal')
    plt.scatter(anomalies, scores[anomalies], c='r', label='Anomaly')
    plt.scatter(true_positives, scores[true_positives], c='g', marker='x', label='True Positive')
    plt.scatter(false_positives, scores[false_positives], c='m', marker='x', label='False Positive')

    plt.xlabel('Data Point')
    plt.ylabel('Anomaly Score')
    plt.title('Anomaly Detection using KNN')
    plt.legend()

    # Calculate AUROC
    y_true = data["labels"]
    auroc = roc_auc_score(y_true, scores)
    
    
    #update the CSV file 
    file_path='/Workspace/Shared/Plot/ECG'
    csvDF = pd.read_csv(file_path)
    csvDF.loc[csvDF['Metric'] == 'Knn', 'AUROC Score'] = auroc 

    # Calculate F1 score
    predicted_labels = (scores > threshold).astype(int)
    f1 = f1_score(y_true, predicted_labels)

    csvDF.loc[csvDF['Metric'] == 'Knn', 'F1 Score'] = f1
    display(csvDF)

    print(csvDF.loc[csvDF['Metric'] == 'ECG', 'F1 Score'])
    csvDF.to_csv(file_path, index=False)
    print(f'DataFrame saved to {file_path}')

    # Calculate F1 score
    predicted_labels = (scores > threshold).astype(int)
    f1 = f1_score(y_true, predicted_labels)

    print("AUROC:", auroc)
    print("F1 Score:", f1)

    plt.show()
 


def main(config: AlgorithmArgs):
    set_random_state(config)
    data, contamination = load_data()
    data = pd.DataFrame(data)
    print("datatype :",type(data))
    clf = KNN(
        contamination=contamination,
        n_neighbors=config.customParameters.n_neighbors,
        method=config.customParameters.method,
        radius=config.customParameters.radius,
        leaf_size=config.customParameters.leaf_size,
        n_jobs=config.customParameters.n_jobs,
        algorithm=config.customParameters.algorithm,
        metric=config.customParameters.distance_metric,
        metric_params=None,
        p=config.customParameters.distance_metric_order,
    )
    clf.fit(data)
    scores = clf.decision_scores_

    data["scores"] = scores
    display(data)
    #display(data)
    np.savetxt(config.dataOutput, scores, delimiter=",")
    threshold = np.percentile(scores, 97)

    plot_anomalies(data, scores, threshold)
    #plot_anomalies(data, scores)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Wrong number of arguments specified; expected a single json-string!")
        exit(1)
  
    json_string = '{"executionType": "execute", "dataInput": "", "dataOutput": "/Workspace/Shared/KNN files/output_scores_synthetic_datasets.csv", "customParameters": {"n_neighbors": 11 , "method": "largest", "radius": 1.0, "leaf_size": 30, "n_jobs": -1, "algorithm": "auto", "distance_metric": "minkowski", "distance_metric_order": 2}, "executionType": "execute"}'
    config = AlgorithmArgs.from_sys_args(json_string)

    print(f"Config: {config}")

    if config.executionType == "train":
        print("Nothing to train, finished!")
    elif config.executionType == "execute":
        main(config)
    else:
        raise ValueError(f"Unknown execution type '{config.executionType}'; expected either 'train' or 'execute'!")

# COMMAND ----------



# COMMAND ----------


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Load your data
ts_data = spark.sql("select * from synthetic_datasets.sine_mean")
ts_data = ts_data.toPandas()
ts_data = ts_data.drop(columns='timestamp')
ts_data = ts_data.astype(float)
data = ts_data.iloc[:, 0].values
labels = ts_data.iloc[:, -1].values

ts_data = spark.sql("select * from synthetic_datasets.ecg_noise_10per1")
ts_data = ts_data.toPandas()
ts_data = ts_data.drop(columns='timestamp')
        # ts_data = ts_data[["MV101","P101","P102","P602","P201","P501","LIT301","AIT402","AIT202","AIT504","AIT502","LIT301","DPIT301","FIT502","FIT401","MV304","_MV303","LIT101","LIT401","UV401","AIT502","P302","P203","P401","P205","P501","DPIT301","UV401"]]
ts_data = ts_data.astype(float)
data = ts_data
labels = ts_data.iloc[:, -1].values

# Reshape data to be a 2D array
#data = data.reshape(-1, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Create a KNN classifier
knn = KNeighborsClassifier()

# Define the hyperparameter grid to search
param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15],  # Values of n_neighbors to try
    'metric': ['euclidean', 'manhattan']        # Different distance metrics to try
}

# Create GridSearchCV with the KNN classifier and hyperparameter grid
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and corresponding accuracy
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print("Test Accuracy of Best Model:", test_accuracy)




# COMMAND ----------



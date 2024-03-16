# Databricks notebook source
pip install TimeEval

# COMMAND ----------

import matplotlib.pyplot as plt

# COMMAND ----------

from timeeval.utils.window import ReverseWindowing
import numpy as np
# post-processing for DeepAnT
def post_deepant(scores: np.ndarray) -> np.ndarray:
    window_size = 45
    prediction_window_size = 1
    size = window_size + prediction_window_size

    return ReverseWindowing(window_size=size).fit_transform(scores)

# COMMAND ----------

import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

def plot_anomalies(data, scores, threshold,name):
    anomalies = np.where(scores > threshold)[0]
    normal = np.where(scores <= threshold)[0]
    true_positives = np.where((scores > threshold) & (data["is_anomaly"] == 1))[0]
    false_positives = np.where((scores > threshold) & (data["is_anomaly"] == 0))[0]
    
    print("false_positives", len(false_positives))
    print("true_positives", len(true_positives))
    
    plt.scatter(normal, scores[normal], c='b', label='Normal')
    plt.scatter(anomalies, scores[anomalies], c='r', label='Anomaly')
    plt.scatter(true_positives, scores[true_positives], c='g', marker='x', label='True Positive')
    plt.scatter(false_positives, scores[false_positives], c='m', marker='x', label='False Positive')

    plt.xlabel('Data Point')
    plt.ylabel('Anomaly Score')
    plt.title('Anomaly Detection using KNN')
    plt.legend()

    # Calculate AUROC
    y_true = data["is_anomaly"]
    auroc = roc_auc_score(y_true, scores)

    # Calculate F1 score
    predicted_labels = (scores > threshold).astype(int)
    f1 = f1_score(y_true, predicted_labels)
    
    file_path='/Workspace/Shared/Plot/'+name
    csvDF = pd.read_csv(file_path)
    csvDF.loc[csvDF['Metric'] == 'DeeP-ANT', 'AUROC Score'] = auroc 

    # Calculate F1 score
    predicted_labels = (scores > threshold).astype(int)
    f1 = f1_score(y_true, predicted_labels)

    csvDF.loc[csvDF['Metric'] == 'DeeP-ANT', 'F1 Score'] = f1
    display(csvDF)

    print(csvDF.loc[csvDF['Metric'] == 'DeeP-ANT', 'F1 Score'])
    csvDF.to_csv(file_path, index=False)
    print(f'DataFrame saved to {file_path}')

    print("AUROC:", auroc)
    print("F1 Score:", f1)
    

    plt.show()


# COMMAND ----------

import numpy as np
data = np.genfromtxt('/Workspace/Shared/deep_ant/Models/anomalies_synthetic-cbf.csv', delimiter=',')
print(data)
a = post_deepant(data)
print(len(a))
ts_data = spark.sql("select * from  synthetic_datasets.cylinder_bell_funnel")
ts_data =ts_data.toPandas()
print(ts_data)
a=a/max(a)
ts_data["s"] = a
print(type(a
    ))
print(ts_data)
ts_data.to_csv('/Workspace/Shared/CISS Trials/an.csv')
print(ts_data['is_anomaly'].sum())
plot_anomalies(ts_data,a,.7,"Cylinder-Bell-Funnel")
ts_data.to_csv("/Workspace/Shared/CISS Trials/an.csv")






# COMMAND ----------





# COMMAND ----------



# COMMAND ----------

import numpy as np
data = np.genfromtxt('/Workspace/Shared/deep_ant/Models/anomalies_synthetic-sine.csv', delimiter=',')
print(data)
a = post_deepant(data)
print(len(a))
ts_data = spark.sql("select * from  synthetic_datasets.sine_platform")
ts_data =ts_data.toPandas()
print(ts_data)
a=a/max(a)

ts_data["s"] = a
print(type(a
    ))
print(ts_data)
ts_data.to_csv('/Workspace/Shared/CISS Trials/an.csv')
print(ts_data['is_anomaly'].sum())
plot_anomalies(ts_data,a,.7,"Sine-Platform")
ts_data.to_csv("/Workspace/Shared/CISS Trials/an.csv")






# COMMAND ----------

import numpy as np
data = np.genfromtxt('/Workspace/Shared/deep_ant/Models/anomalies_synthetic-ecg.csv', delimiter=',')
print(data)
a = post_deepant(data)
print(len(a))
ts_data = spark.sql("select * from  synthetic_datasets.ecg_noise_10per1")
ts_data =ts_data.toPandas()
print(ts_data)
a=a/max(a)

ts_data["s"] = a
print(type(a
    ))
print(ts_data)
ts_data.to_csv('/Workspace/Shared/CISS Trials/an.csv')
print(ts_data['is_anomaly'].sum())
plot_anomalies(ts_data,a,.5,"ECG")
ts_data.to_csv("/Workspace/Shared/CISS Trials/an.csv")

# COMMAND ----------

import numpy as np
data = np.genfromtxt('/Workspace/Shared/deep_ant/Models/anomalies_synthetic-sine-mean.csv', delimiter=',')
print(data)
a = post_deepant(data)
print(len(a))
ts_data = spark.sql("select * from  synthetic_datasets.sine_mean")
ts_data =ts_data.toPandas()
print(ts_data)
a=a/max(a)

ts_data["s"] = a
print(type(a
    ))
print(ts_data)
ts_data.to_csv('/Workspace/Shared/CISS Trials/an.csv')
print(ts_data['is_anomaly'].sum())
plot_anomalies(ts_data,a,.7,"Sine-Mean")
ts_data.to_csv("/Workspace/Shared/CISS Trials/an.csv")






# COMMAND ----------

import numpy as np
data = np.genfromtxt('/Workspace/Shared/deep_ant/Models/anomalies_swat.csv', delimiter=',')
print(data)
a = post_deepant(data)
print(len(a))
ts_data = spark.sql("select * from  swat_preprocessed.swat_2015")
ts_data =ts_data.toPandas()
print(ts_data)

ts_data["s"] = a
print(type(a
    ))
print(ts_data)
ts_data.to_csv('/Workspace/Shared/CISS Trials/an.csv')
ts_data.rename(columns={'isAnomoly': 'is_anomaly'}, inplace=True)
print(ts_data['is_anomaly'].sum())

plot_anomalies(ts_data,a,1.122096865615781,"Swat")
ts_data.to_csv("/Workspace/Shared/CISS Trials/an.csv")


# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc

# # Assuming you have decision scores from your model
# decision_scores = model.decision_function(X_test)  # Replace with your actual scores

# True labels
y_true = ts_data["is_anomaly"]
decision_scores = post_deepant(data)
# Calculate ROC curve
fpr, tpr, thresholds_roc = roc_curve(y_true, decision_scores)
roc_auc = auc(fpr, tpr)

# Calculate F1 scores for different thresholds
thresholds_f1 = np.arange(min(decision_scores), max(decision_scores), 0.01)
f1_scores = []

for threshold in thresholds_f1:
    binary_predictions = (decision_scores >= threshold).astype(int)
    f1_scores.append(f1_score(y_true, binary_predictions))

# Combine F1 scores and ROC AUC into a single metric (You can adjust the weights)
combined_metric = [0.5 * (f1 + roc_auc) for f1 in f1_scores]

# Find the threshold that maximizes the combined metric
best_threshold = thresholds_f1[np.argmax(combined_metric)]
best_f1 = f1_scores[np.argmax(combined_metric)]
best_roc_auc = roc_auc

# Plot ROC curve
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')

# Plot F1 score vs. threshold
plt.subplot(1, 2, 2)
plt.plot(thresholds_f1, f1_scores, label='F1 Score', color='blue')
plt.axvline(x=best_threshold, color='red', linestyle='--', label='Best Threshold (F1 Maximized)')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('F1 Score vs. Threshold')
plt.legend()

plt.tight_layout()
plt.show()

print(f'Best Threshold for F1 Score: {best_threshold}')
print(f'Best F1 Score: {best_f1}')
print(f'ROC AUC: {best_roc_auc}')

# COMMAND ----------



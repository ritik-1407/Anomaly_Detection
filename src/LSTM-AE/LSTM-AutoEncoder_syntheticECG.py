# Databricks notebook source
pip install tensorflow

# COMMAND ----------

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Input, Dropout
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Model
import seaborn as sns

# COMMAND ----------

dataframe = spark.sql("select * from synthetic_datasets.ecg_noise_10per1")
df = dataframe[['timestamp', 'value-0']]
df = df.toPandas()
display(df)

# COMMAND ----------

plt.figure(figsize=(12, 6)) 
sns.lineplot(x=df['timestamp'], y=df['value-0'])

# COMMAND ----------

df2 = dataframe
df2 = df2.toPandas()
train, test, test1 = df.iloc[:400],df.iloc[401:],df2.iloc[401:]
print(len(train))
print(len(test))

# COMMAND ----------

display(test1)

# COMMAND ----------

scaler = StandardScaler()
scaler = scaler.fit(train[['value-0']])
train['value-0'] = scaler.transform(train[['value-0']])
test['value-0'] = scaler.transform(test[['value-0']])

# COMMAND ----------


seq_size = 24  # Number of time steps to look back 

def to_sequences(x, y, seq_size=1):
    x_values = []
    y_values = []

    for i in range(len(x)-seq_size):
        x_values.append(x.iloc[i:(i+seq_size)].values)
        y_values.append(y.iloc[i+seq_size])
        
    return np.array(x_values), np.array(y_values)

trainX, trainY = to_sequences(train[['value-0']], train['value-0'], seq_size)
testX, testY = to_sequences(test[['value-0']], test['value-0'], seq_size)

# COMMAND ----------

model = Sequential()
model.add(LSTM(128 ,input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dropout(rate=0.2))

model.add(RepeatVector(trainX.shape[1]))

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(rate=0.2))
model.add(TimeDistributed(Dense(trainX.shape[2])))

#complie
model.compile(optimizer='adam', loss='mae')
model.summary()

# COMMAND ----------

history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

# COMMAND ----------

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()

# COMMAND ----------

#Plot train data
trainPredict = model.predict(trainX)
trainMAE = np.mean(np.abs(trainPredict - trainX), axis=1)
plt.hist(trainMAE, bins=30)


# COMMAND ----------

testPredict = model.predict(testX)
testMAE = np.mean(np.abs(testPredict - testX), axis=1)
plt.hist(testMAE, bins=30)

# COMMAND ----------

max_trainMAE = 0.27
#90% value of max as threshold.

# COMMAND ----------

anomaly_df = pd.DataFrame(test[seq_size:])
anomaly_df['testMAE'] = testMAE
anomaly_df['max_trainMAE'] = max_trainMAE
anomaly_df['anomaly'] = anomaly_df['testMAE'] > anomaly_df['max_trainMAE']
anomaly_df['value-0'] = test[seq_size:]['value-0']


# COMMAND ----------

display(anomaly_df)

# COMMAND ----------

#Plot anomalies - Showing Anomalies
sns.lineplot(x=anomaly_df['timestamp'], y=anomaly_df['testMAE'])
sns.lineplot(x=anomaly_df['timestamp'], y=anomaly_df['max_trainMAE'])
plt.title("Plot anomalies - Showing Anomalies")

# COMMAND ----------

anomalies = anomaly_df.loc[anomaly_df['anomaly'] == True]
display(anomalies)

# COMMAND ----------

#Plot anomalies - Showing Anomaly days
sns.lineplot(x=anomaly_df['timestamp'], y=scaler.inverse_transform(anomaly_df[['value-0']]).flatten())
sns.scatterplot(x=anomalies['timestamp'], y=scaler.inverse_transform(anomalies[['value-0']]).flatten(), color='r')
plt.title("Plot anomalies - Showing Anomaly days")

# COMMAND ----------

df = dataframe.toPandas()
test = df.iloc[401:]
display(test)

# COMMAND ----------

# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import f1_score

# ground_t= test1["is_anomaly"][seq_size:]
# predicted = anomaly_df['anomaly']

# print(len(ground_t))
# count_of_ones = (test1["is_anomaly"][seq_size:]== 1).sum()
# print(count_of_ones)
# count_of_one = (anomaly_df['anomaly']== 1).sum()
# print(count_of_one)
# print(count_of_ones)
# f1 = f1_score(ground_t, predicted)  # or 'macro', 'micro', etc.

# print("F1 Score:", f1)
# roc_auc = roc_auc_score(ground_t, predicted) 
# print("ROC AUC Score:", roc_auc)

# COMMAND ----------


def plot_anomalies(data, scores, threshold):
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
    plt.title('Anomaly Detection')
    plt.legend()

    # Calculate AUROC
    y_true = data["is_anomaly"]
    auroc = roc_auc_score(y_true, scores)

    # Calculate F1 score
    predicted_labels = (scores > threshold).astype(int)
    f1 = f1_score(y_true, predicted_labels)

    #update the CSV file 
    csv_file_path='/Workspace/Shared/Plot/ECG'
    csvDF = pd.read_csv(csv_file_path)
    display(csvDF)
    csvDF.loc[csvDF['Metric'] == 'LSTM-AE', 'AUROC Score'] = auroc 

    csvDF.loc[csvDF['Metric'] == 'LSTM-AE', 'F1 Score'] = f1 
    csvDF.to_csv(csv_file_path, index=False)
    print(f'DataFrame saved to {csv_file_path}')

    print("AUROC:", auroc)
    print("F1 Score:", f1)

    plt.show()

# COMMAND ----------

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
plot_anomalies(test1[seq_size:],anomaly_df['testMAE'].values,0.27)

# COMMAND ----------

# #update the CSV file 
# csv_file_path='/Workspace/Shared/Plot/ECG'
# csvDF = pd.read_csv(csv_file_path)
# display(csvDF)
# csvDF.loc[csvDF['Metric'] == 'LSTM-AE', 'AUROC Score'] = auroc 

# csvDF.loc[csvDF['Metric'] == 'LSTM-AE', 'F1 Score'] = f1 
# csvDF.to_csv(csv_file_path, index=False)
# print(f'DataFrame saved to {csv_file_path}')

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc

# # Assuming you have decision scores from your model
# decision_scores = model.decision_function(X_test)  # Replace with your actual scores

# True labels
y_true = test1["is_anomaly"][seq_size:]
decision_scores = anomaly_df['testMAE'].values
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


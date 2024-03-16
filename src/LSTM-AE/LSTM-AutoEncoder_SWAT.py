# Databricks notebook source
# MAGIC %pip install tensorflow

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

dataframe = spark.sql("SELECT * FROM swat_preprocessed.swat_2015")
df=dataframe.toPandas()
df = df.drop(columns='Timestamp')
display(df)

# COMMAND ----------

#Plot dataframe
# sns.lineplot(x=df['timestamp'], y=df)

# COMMAND ----------

#Divide train and test
split_ratio=.4
split_len = int(split_ratio * len(df))

train, test,test1 = df.iloc[:split_len],df.iloc[split_len:],df.iloc[split_len:]
print(len(train))
print(len(test))

# COMMAND ----------

#Scaling
scaler = StandardScaler() 
scaler = scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

train = pd.DataFrame(train)
test = pd.DataFrame(test)

# COMMAND ----------

#Apply sequence

seq_size = 30  # Number of time steps to look back 

def create_sequences(data, seq_size):
    sequences = []
    for i in range(len(data) - seq_size):
        sequence = data[i:i+seq_size]
        sequences.append(sequence)
    return np.array(sequences)

train = create_sequences(train, seq_size)
print(train.shape)
test = create_sequences(test, seq_size)
print(test
      .shape)

# COMMAND ----------

model = Sequential()

# Encoder
model.add(LSTM(128, activation='relu', input_shape=(seq_size, 52), return_sequences=True))
model.add(LSTM(64, activation='relu', return_sequences=False))
model.add(RepeatVector(seq_size))

# Decoder
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(52)))

model.compile(optimizer='adam', loss='mse') 

# COMMAND ----------

#Plot loss
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()

# COMMAND ----------

#Plot train data
trainPredict = model.predict(train)
trainMAE = np.mean(np.abs(trainPredict - train), axis=1)
print(trainMAE)
plt.hist(trainMAE, bins=30)


# COMMAND ----------

#Plot test data
testPredict = model.predict(test)
print((testPredict.shape))
testMAE = np.mean(np.abs(testPredict - test), axis=1)
print((testMAE.shape))
plt.hist(testMAE, bins=30)

# COMMAND ----------

mae_values = []

# Iterate through each array along the rows
for i in range(testPredict.shape[0]):  # Assuming testPredict and test have the same number of rows
    mae = np.mean(np.abs(testPredict[i] - test[i]))  # Calculate MAE for the current pair of arrays
    mae_values.append(mae)
    
print(len(mae_values))

# COMMAND ----------

max_trainMAE = .7  #90% value of max as threshold.

# COMMAND ----------

#Capture all details in a DataFrame for easy plotting
# flattened_data = test.reshape(-1, 30 * 52)
anomaly_df = pd.DataFrame(test1[seq_size:]) 
print(len(anomaly_df.columns))
display(testMAE)
anomaly_df['testMAE'] = mae_values
anomaly_df['max_trainMAE'] = max_trainMAE
anomaly_df['anomaly'] = anomaly_df['testMAE'] > anomaly_df['max_trainMAE']
anomaly_df = anomaly_df.replace({True: 1, False: 0})
# anomaly_df['value-0'] = test[seq_size:]['value-0']


# COMMAND ----------

display(anomaly_df)
print(len(anomaly_df))

# COMMAND ----------

#Plot anomalies
# sns.lineplot(x=anomaly_df['timestamp'], y=anomaly_df['testMAE'])
# sns.lineplot(x=anomaly_df['timestamp'], y=anomaly_df['max_trainMAE'])

# COMMAND ----------

#Create anomaly dataframe
anomalies = anomaly_df.loc[anomaly_df['anomaly'] == 1]
print(len(anomalies))

# COMMAND ----------

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
ground_t= test1["isAnomoly"][seq_size:]
predicted = anomaly_df['anomaly']

print(len(ground_t))
count_of_ones = (test1["isAnomoly"][seq_size:]== 1).sum()
print(count_of_ones)
count_of_one = (anomaly_df['anomaly']== 1).sum()
print(count_of_one)
print(count_of_ones)
f1 = f1_score(ground_t, predicted)  # or 'macro', 'micro', etc.

print("F1 Score:", f1)
roc_auc = roc_auc_score(ground_t, predicted) 
print("ROC AUC Score:", roc_auc)

# COMMAND ----------


def plot_anomalies(data, scores, threshold):
    anomalies = np.where(scores > threshold)[0]
    normal = np.where(scores <= threshold)[0]
    true_positives = np.where((scores > threshold) & (data["isAnomoly"] == 1))[0]
    false_positives = np.where((scores > threshold) & (data["isAnomoly"] == 0))[0]
    
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
    y_true = data["isAnomoly"]
    auroc = roc_auc_score(y_true, scores)

    # Calculate F1 score
    predicted_labels = (scores > threshold).astype(int)
    f1 = f1_score(y_true, predicted_labels)

    #update the CSV file 
    csv_file_path='/Workspace/Shared/Plot/Swat'
    csvDF = pd.read_csv(csv_file_path)
    csvDF.loc[csvDF['Metric'] == 'LSTM-AE', 'AUROC Score'] = auroc 

    csvDF.loc[csvDF['Metric'] == 'LSTM-AE', 'F1 Score'] = f1 
    csvDF.to_csv(csv_file_path, index=False)
    print(f'DataFrame saved to {csv_file_path}')

    print("AUROC:", auroc)
    print("F1 Score:", f1)

    plt.show()

# COMMAND ----------

plot_anomalies(test1[seq_size:],anomaly_df['testMAE'].values,1.19724)

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc

# # Assuming you have decision scores from your model
# decision_scores = model.decision_function(X_test)  # Replace with your actual scores

# True labels
y_true = test1["isAnomoly"][seq_size:]
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


# COMMAND ----------



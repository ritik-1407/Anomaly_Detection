# Databricks notebook source
pip install PyNomaly

# COMMAND ----------

pip install pydataset

# COMMAND ----------

# pip install pytest

# COMMAND ----------

# pip uninstall scikit-learn --yes

# COMMAND ----------

# db = DBSCAN(eps=0.6, min_samples=50).fit(df)
# print(db)

# COMMAND ----------

from sklearn.utils import check_random_state
import numpy as np
from sklearn.metrics import roc_auc_score
from pyspark.sql import SparkSession
from sklearn.cluster import DBSCAN
import pandas as pd
from sklearn.metrics import f1_score
from PyNomaly import loop

# cylinder_bell_funnel
df = spark.sql("select * from synthetic_datasets.cylinder_bell_funnel")
df = df.toPandas()
print(len(df))
df.set_index('timestamp', inplace=True)
df = df.astype(float)

X_outliers = df[df['is_anomaly'] == 1]
print(len(X_outliers))
X_test = df

X_labels = np.r_[np.repeat(1, df[df['is_anomaly'] == 0].shape[0]), np.repeat(-1, X_outliers.shape[0])]

clf = loop.LocalOutlierProbability(
X_test,
n_neighbors=20
)

score = clf.fit().local_outlier_probabilities
share_outlier = X_outliers.shape[0] / X_test.shape[0]
X_pred = [-1 if s > share_outlier else 1 for s in score]

f1 = f1_score(X_pred, X_labels)
print("F1 Score:", f1)
auroc = str(roc_auc_score(X_pred, X_labels))
print("Auroc:", auroc)

#update the CSV file 
csv_file_path='/Workspace/Shared/Plot/Cylinder-Bell-Funnel'
csvDF = pd.read_csv(csv_file_path)
csvDF.loc[csvDF['Metric'] == 'LoOP', 'AUROC Score'] = auroc 

csvDF.loc[csvDF['Metric'] == 'LoOP', 'F1 Score'] = f1 
csvDF.to_csv(csv_file_path, index=False)
print(f'DataFrame saved to {csv_file_path}')

# COMMAND ----------

from sklearn.utils import check_random_state
import numpy as np
from sklearn.metrics import roc_auc_score
from pyspark.sql import SparkSession
from sklearn.cluster import DBSCAN
import pandas as pd
from sklearn.metrics import f1_score
from PyNomaly import loop

#sine_platform

df = spark.sql("select * from synthetic_datasets.sine_platform")
df = df.toPandas()
print(len(df))
df.set_index('timestamp', inplace=True)
df = df.astype(float)

X_outliers = df[df['is_anomaly'] == 1]
print(len(X_outliers))
X_test = df

X_labels = np.r_[np.repeat(1, df[df['is_anomaly'] == 0].shape[0]), np.repeat(-1, X_outliers.shape[0])]

clf = loop.LocalOutlierProbability(
X_test,
n_neighbors=20
)

score = clf.fit().local_outlier_probabilities
share_outlier = X_outliers.shape[0] / X_test.shape[0]
print(share_outlier)
X_pred = [-1 if s > share_outlier else 1 for s in score]

f1 = f1_score(X_pred, X_labels)
print("F1 Score:", f1)
auroc = str(roc_auc_score(X_pred, X_labels))
print("Auroc:", auroc)

#update the CSV file 
csv_file_path='/Workspace/Shared/Plot/Sine-Platform'
csvDF = pd.read_csv(csv_file_path)
csvDF.loc[csvDF['Metric'] == 'LoOP', 'AUROC Score'] = auroc 

csvDF.loc[csvDF['Metric'] == 'LoOP', 'F1 Score'] = f1 
csvDF.to_csv(csv_file_path, index=False)
print(f'DataFrame saved to {csv_file_path}')


# COMMAND ----------

from sklearn.utils import check_random_state
import numpy as np
from sklearn.metrics import roc_auc_score
from pyspark.sql import SparkSession
from sklearn.cluster import DBSCAN
import pandas as pd
from sklearn.metrics import f1_score
from PyNomaly import loop

#sine_mean

df = spark.sql("select * from synthetic_datasets.sine_mean")
df = df.toPandas()
print(len(df))
df.set_index('timestamp', inplace=True)
df = df.astype(float)

X_outliers = df[df['is_anomaly'] == 1]
print(len(X_outliers))
X_test = df

X_labels = np.r_[np.repeat(1, df[df['is_anomaly'] == 0].shape[0]), np.repeat(-1, X_outliers.shape[0])]

clf = loop.LocalOutlierProbability(
X_test,
n_neighbors=20
)

score = clf.fit().local_outlier_probabilities
share_outlier = X_outliers.shape[0] / X_test.shape[0]
X_pred = [-1 if s > share_outlier else 1 for s in score]

f1 = f1_score(X_pred, X_labels)
print("F1 Score:", f1)
auroc = str(roc_auc_score(X_pred, X_labels))
print("Auroc:", auroc)

#update the CSV file 
csv_file_path='/Workspace/Shared/Plot/Sine-Mean'
csvDF = pd.read_csv(csv_file_path)
csvDF.loc[csvDF['Metric'] == 'LoOP', 'AUROC Score'] = auroc 

csvDF.loc[csvDF['Metric'] == 'LoOP', 'F1 Score'] = f1 
csvDF.to_csv(csv_file_path, index=False)
print(f'DataFrame saved to {csv_file_path}')

# COMMAND ----------

from sklearn.utils import check_random_state
import numpy as np
from sklearn.metrics import roc_auc_score
from pyspark.sql import SparkSession
from sklearn.cluster import DBSCAN
import pandas as pd
from sklearn.metrics import f1_score
from PyNomaly import loop

#ecg_noise_10per1

df = spark.sql("select * from synthetic_datasets.ecg_noise_10per1")
df = df.toPandas()
print(len(df))
df.set_index('timestamp', inplace=True)
df = df.astype(float)

X_outliers = df[df['is_anomaly'] == 1]
print(len(X_outliers))
X_test = df

X_labels = np.r_[np.repeat(1, df[df['is_anomaly'] == 0].shape[0]), np.repeat(-1, X_outliers.shape[0])]

clf = loop.LocalOutlierProbability(
X_test,
n_neighbors=20
)

score = clf.fit().local_outlier_probabilities
share_outlier = X_outliers.shape[0] / X_test.shape[0]
X_pred = [-1 if s > share_outlier else 1 for s in score]

f1 = f1_score(X_pred, X_labels)
print("F1 Score:", f1)
auroc = str(roc_auc_score(X_pred, X_labels))
print("Auroc:", auroc)

#update the CSV file 
csv_file_path='/Workspace/Shared/Plot/ECG'
csvDF = pd.read_csv(csv_file_path)
csvDF.loc[csvDF['Metric'] == 'LoOP', 'AUROC Score'] = auroc 

csvDF.loc[csvDF['Metric'] == 'LoOP', 'F1 Score'] = f1 
csvDF.to_csv(csv_file_path, index=False)
print(f'DataFrame saved to {csv_file_path}')

# COMMAND ----------

from sklearn.utils import check_random_state
import numpy as np
from sklearn.metrics import roc_auc_score,f1_score
from PyNomaly import loop
from pyspark.sql import SparkSession
from sklearn.cluster import DBSCAN
import pandas as pd

# Generate train/test data
# rng = check_random_state(2)
# X_n120 = 0.3 * rng.randn(120, 2)

# df = spark.sql("select * from synthetic_datasets.sine_mean")
# df = df.toPandas()
# df.set_index('timestamp', inplace=True)
df = spark.sql("select * from swat_preprocessed.swat_2015")
    # ts_data = spark.sql("select * from default.copy_of_swat_dataset_normal_v0_csv")
df = df.toPandas()
df = df.drop(columns='Timestamp')
df = df[["MV101","P102","LIT301","AIT202","AIT504","LIT301","DPIT301","FIT401","MV304","_MV303","LIT101","LIT401","UV401","AIT502","P203","P401","P205","isAnomoly"]]
df = df.astype(float)
# df=df.drop("timestamp")
X_outliers = df[df['isAnomoly'] == 1]

# X_test = np.r_[X_n120, X_outliers]
X_test = df


# X_labels = np.r_[np.repeat(1, X_n120.shape[0]), np.repeat(-1, X_outliers.shape[0])]
X_labels = np.r_[np.repeat(1, df[df['isAnomoly'] == 0].shape[0]), np.repeat(-1, X_outliers.shape[0])]

print("X_outliers")
print(X_outliers)
print("X_test")
print(X_test)
print((X_labels))

clf = loop.LocalOutlierProbability(
X_test,
n_neighbors=X_test.shape[0] - 100
)

score = clf.fit().local_outlier_probabilities
share_outlier = X_outliers.shape[0] / X_test.shape[0]
print(X_outliers.shape[0])
print(share_outlier)
X_pred = [-1 if s > .5 else 1 for s in score]
f1 = f1_score(X_pred, X_labels)
print("F1 Score:", f1)
auroc = str(roc_auc_score(X_pred, X_labels))
print("Auroc:", auroc)
#update the CSV file 
csv_file_path='/Workspace/Shared/Plot/Swat'
csvDF = pd.read_csv(csv_file_path)
csvDF.loc[csvDF['Metric'] == 'LoOP', 'AUROC Score'] = auroc

csvDF.loc[csvDF['Metric'] == 'LoOP', 'F1 Score'] = f1
df.to_csv(csv_file_path, index=False)
print(f'DataFrame saved to {csv_file_path}')


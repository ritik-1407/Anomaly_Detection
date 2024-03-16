# Databricks notebook source

import pandas as pd
df = spark.sql("SELECT * FROM default.swat_dataset_attack_v0_csv")
df=df.toPandas()
#display(df)
df1 = df[["Timestamp","LIT101"]]
df1['Timestamp'] = pd.to_datetime(df['Timestamp'], format=' %d/%m/%Y %I:%M:%S %p')


df1['LIT101'] = pd.to_numeric(df1['LIT101'])
min_value = df1['LIT101'].min()
max_value = df1['LIT101'].max()
#df1['LIT101_normalized'] = (df1['LIT101'] - min_value) / (max_value - min_value)
df1['isAnomaly']= df1['Timestamp'].between('1/01/2016 14:21:12','1/01/2016 14:28:35').astype(int)
df1
display(df1[df1["isAnomaly"] == 1])
df1.set_index('Timestamp', inplace=True, drop=True)
#df1.drop('LIT101', axis=1, inplace=True)

df1


# COMMAND ----------

df1['LIT101_normalized'] = (df1['LIT101'] - mean_LIT101) / std_LIT101

# COMMAND ----------

df1

# COMMAND ----------


    df = spark.sql("SELECT * FROM default.swat_dataset_attack_v0_csv")
    df=df.toPandas()
#display(df)
    df1 = df[["Timestamp","LIT101"]]
    df1['Timestamp'] = pd.to_datetime(df['Timestamp'], format=' %d/%m/%Y %I:%M:%S %p')

#display(df1[df1["isAnomaly"] == 1])
    df1['LIT101'] = pd.to_numeric(df1['LIT101'])
    min_value = df1['LIT101'].min()
    max_value = df1['LIT101'].max()
    #df1['LIT101_normalized'] = (df1['LIT101'] - min_value) / (max_value - min_value)
    df1['isAnomoly']= df1['Timestamp'].between('1/01/2016 14:21:12','1/01/2016 14:28:35').astype(int)
 

    df1.set_index('Timestamp', inplace=True, drop=True)
    #df1.drop('LIT101', axis=1, inplace=True)
    df=df1[:5000]

# COMMAND ----------

#Swat data loading / Preprocessing
    df = spark.sql("select * from default.swat_dataset_attack_v0_csv")
    df = df.toPandas()
    replacement_dict = {"Normal": 0, "Attack": 1}
    df['isAnomoly'] = df['isAnomoly'].replace(replacement_dict)
    df = df[["MV101","P102","LIT301","AIT202","AIT504","LIT301","DPIT301","FIT401","MV304","_MV303","LIT101","LIT401","UV401","AIT502","P203","P401","P205","isAnomoly"]]
    df= df[:50000]  


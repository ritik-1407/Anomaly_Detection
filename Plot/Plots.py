# Databricks notebook source
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_names = ['Swat', 'ECG', 'Sine-Mean', 'Sine-Platform', 'Cylinder-Bell-Funnel']


base_path = '/Workspace/Shared/Plot/'


dataframes = {}

for file_name in file_names:
    file_path = f'{base_path}{file_name}'  
    df = pd.read_csv(file_path)  
    dataframes[file_name] = df  

for file_name, df in dataframes.items():
    sns.set(style="whitegrid")  
    df1 = pd.DataFrame(df)
    print(file_name)
    df1_melted = pd.melt(df1, id_vars=['Metric'], value_vars=['F1 Score', 'AUROC Score'])
    ax1 = sns.barplot(x="Metric", y="value", hue="variable", data=df1_melted, palette="Blues")
    ax1.set(ylabel="Scores", title=f"F1 Score vs AUROC Score for {file_name} dataset")
    plt.xticks(rotation=0)  # Rotate x-axis labels if needed
    plt.show()


# COMMAND ----------




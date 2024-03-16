# Databricks notebook source
df = spark.sql("select * from wadl.wadi_14days_new_csv")
df= df.toPandas()
#path = "/Workspace/Shared/KNN files/wadl.csv"
path = "/Shared/KNN files/wadl.csv"
#df.write.format("csv").option("header", "true").mode("overwrite").save(path)
df.tocsv('/Workspace/Shared/KNN files/wadl.csv')
#df.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save("/Workspace/Shared/KNN files/wadl3.csv")



# COMMAND ----------



# COMMAND ----------

dbutils.fs.ls("/Workspace/Shared/KNN files/wadl3.csv")

# COMMAND ----------

df = spark.sql("select * from swat_preprocessing.df_june_20210623")
df= df.toPandas()
output_path = ''  # Update the path to an absolute file path

df.to_csv('/Workspace/Shared/KNN files/Swat_july_20210702.csv')


# COMMAND ----------



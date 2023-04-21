from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

from train_functions import train_models
df = spark.read.options(inferSchema='True', header='True')\
    .csv("hdfs://namenode:8020/complete_data.csv")

train_models(df)
sc.stop()
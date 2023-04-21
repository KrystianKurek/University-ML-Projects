
from pyspark.sql import SparkSession
from pyspark.sql.types import  StructType, StructField, StringType, LongType, DoubleType, IntegerType, ArrayType
from pyspark.sql.functions import expr, from_json, col, concat, round, lit, year, month, dayofweek, hour, coalesce, lit, count
from pyspark.streaming import StreamingContext
from pyspark.ml.regression import GBTRegressionModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

import os
import shutil
import json
print(os.getcwd(), 'PAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAtth')

with open("/home/base_models_learning/medians.json") as f:
    medians = json.load(f)

spark = SparkSession \
    .builder \
    .appName("Spark Kafka Streaming") \
    .master("spark://spark-master:7077") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.1") \
    .config("spark.cassandra.connection.host", "cassandra")\
    .config("spark.scheduler.mode", "FAIR")\
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
df_air = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:29092") \
    .option("subscribe", "air-data") \
    .load()


df_weather = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:29092") \
    .option("subscribe", "weather-data") \
    .load()

schema_weather = StructType([
    StructField("coord_lon", DoubleType()),
    StructField("coord_lat", DoubleType()),
    StructField("weather", ArrayType(StructType([
        StructField("id", IntegerType()),
        StructField("main", StringType()),
        StructField("description", StringType()),
        StructField("icon", StringType())
    ]))),
    StructField("base", StringType()),
    StructField("main_temp", DoubleType()),
    StructField("main_feels_like", DoubleType()),
    StructField("main_temp_min", DoubleType()),
    StructField("main_temp_max", DoubleType()),
    StructField("main_pressure", IntegerType()),
    StructField("main_humidity", IntegerType()),
    StructField("visibility", IntegerType()),
    StructField("wind_speed", DoubleType()),
    StructField("wind_deg", DoubleType()),
    StructField("wind_gust", DoubleType()),
    StructField("clouds_all", IntegerType()),
    StructField("dt_all", IntegerType()),
    StructField("sys_type", IntegerType()),
    StructField("sys_id", IntegerType()),
    StructField("sys_country", StringType()),
    StructField("sys_sunrise", IntegerType()),
    StructField("sys_sunset", IntegerType()),
    StructField("timezone", IntegerType()),
    StructField("id", IntegerType()),
    StructField("name", StringType()),
    StructField("cod", IntegerType()),
    StructField("snow_1h", DoubleType()),
    StructField("rain_1h", DoubleType()),
])
schema_air = StructType([
    StructField("station_id", StringType()),
    StructField("station_name", StringType()),
    StructField("station_gegrLat", DoubleType()),
    StructField("station_gegrLon", DoubleType()),
    StructField("sensor_id", IntegerType()),
    StructField("sensor_param_id", IntegerType()),
    StructField("sensor_param_code", StringType()),
    StructField("sensor_param_formula", StringType()),
    StructField("sensor_param_name", StringType()),
    StructField("measurement_date", StringType()),
    StructField("measurement_value", DoubleType()),
])


value_df_weather = df_weather.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)", "CAST(timestamp AS TIMESTAMP)")
json_df_weather = value_df_weather.select(from_json(col("value"),schema_weather).alias("value"), 'timestamp')
exploaded_df_weather = json_df_weather.select("value.*", "timestamp")
exploaded_df_weather = exploaded_df_weather\
    .withColumn("lon", round("coord_lon", 4))\
    .withColumn("lat", round("coord_lat", 4))
exploaded_df_weather = exploaded_df_weather.withColumn("weather_timestamp", exploaded_df_weather["timestamp"])
watermarked_weather = exploaded_df_weather.withWatermark("weather_timestamp", "90 seconds")

value_df_air = df_air.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)", "CAST(timestamp AS TIMESTAMP)")
json_df_air = value_df_air.select(from_json(col("value"),schema_air).alias("value"), "timestamp")
exploaded_df_air = json_df_air.select("value.*", "timestamp")
exploaded_df_air = exploaded_df_air\
    .withColumn("lon", round("station_gegrLon", 4))\
    .withColumn("lat", round("station_gegrLat", 4))
exploaded_df_air = exploaded_df_air.withColumn("air_timestamp", exploaded_df_air["timestamp"])
watermarked_air = exploaded_df_air.withWatermark("air_timestamp", "90 seconds")

joined_df = watermarked_air.alias("air").join(watermarked_weather.alias("weather"), expr("""
    air.lon = weather.lon AND
    air.lat = weather.lat
"""))

no2_table = joined_df.filter("sensor_param_code = 'NO2'").select("*", month("measurement_date").alias("month"),hour("measurement_date").alias("hour"), dayofweek("measurement_date").alias("weekday"))
o3_table = joined_df.filter("sensor_param_code = 'O3'").select("*", month("measurement_date").alias("month"),hour("measurement_date").alias("hour"), dayofweek("measurement_date").alias("weekday"))
pm25_table = joined_df.filter("sensor_param_code = 'PM10'").select("*", month("measurement_date").alias("month"),hour("measurement_date").alias("hour"), dayofweek("measurement_date").alias("weekday"))
pm10_table = joined_df.filter("sensor_param_code = 'PM2.5'").select("*", month("measurement_date").alias("month"),hour("measurement_date").alias("hour"), dayofweek("measurement_date").alias("weekday"))

for col_name, median in medians.items():
    no2_table = no2_table.withColumn(col_name, coalesce(col(col_name), lit(median)))
    o3_table = o3_table.withColumn(col_name, coalesce(col(col_name), lit(median)))
    pm25_table = pm25_table.withColumn(col_name, coalesce(col(col_name), lit(median)))
    pm10_table = pm10_table.withColumn(col_name, coalesce(col(col_name), lit(median)))

# no2_table_out.awaitTermination()
modelno2 = GBTRegressionModel.load("/home/base_models_learning/2023_01_15_16_34_18_GB_NO2.model")
modelo3 = GBTRegressionModel.load("/home/base_models_learning/2023_01_15_16_34_18_GB_O3.model")
modelpm10 = GBTRegressionModel.load("/home/base_models_learning/2023_01_15_16_34_18_GB_PM10.model")
modelpm25 = GBTRegressionModel.load("/home/base_models_learning/2023_01_15_16_34_18_GB_PM25.model")

assembler = VectorAssembler(inputCols=["measurement_value", "main_temp", "main_pressure", "main_humidity", "wind_speed", "wind_deg", "clouds_all", "snow_1h", "rain_1h", "month", "hour", "weekday"], outputCol="features")
final_df_no2 = assembler.setHandleInvalid("skip").transform(no2_table)
final_df_o3 = assembler.setHandleInvalid("skip").transform(o3_table)
final_df_pm25 = assembler.setHandleInvalid("skip").transform(pm25_table)
final_df_pm10 = assembler.setHandleInvalid("skip").transform(pm10_table)

# Drukowanie liczby wierszy:
# def batch_write(output_df, batch_id):
#     print(batch_id)
#     output_df.select([count(when(isnan(c), c)).alias(c) for c in ["measurement_value", "main_temp", "main_pressure", "main_humidity", "wind_speed", "wind_deg", "clouds_all", "snow_1h", "rain_1h", "month", "hour", "weekday"]]).show()
#     print(batch_id)
    # print("inside foreachBatch for batch_id:{0}, rows in passed dataframe: {1}".format(batch_id, output_df.count()))
# final_df_no2.writeStream\
#          .foreachBatch(batch_write)\
#          .start()\
#          .awaitTermination()


# # Drukowanie danych
# test = final_df_pm25.writeStream.outputMode("append").format("console").start()


checkpoint_idx = 0
def create_final_query(model, table):
    global checkpoint_idx
    checkpoint_idx += 1
    checkpoint_path = f'/tmp/checkpoint{checkpoint_idx}/'
    shutil.rmtree(checkpoint_path, ignore_errors=True)
    return model\
        .transform(table)\
        .select(
            col('station_gegrLon').alias('longtitude'),
            col('station_gegrLat').alias('latitude'),
            col('station_name'),
            col('weather_timestamp').alias('timestamp'),
            col('sensor_param_code').alias('particle'), 
            col('prediction')
            )\
        .writeStream\
        .option("checkpointLocation", checkpoint_path)\
        .format("org.apache.spark.sql.cassandra")\
        .options(table="realtime_views", keyspace="apache_air")\
        .start()


pm25_out_query = create_final_query(modelpm25, final_df_pm25)
no2_out_query = create_final_query(modelno2, final_df_no2)
o3_out_query = create_final_query(modelo3, final_df_o3)
pm10_out_query = create_final_query(modelpm10, final_df_pm10)

spark.streams.awaitAnyTermination()

#  /spark/bin/spark-submit --master spark://spark-master:7077 --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.1,com.datastax.spark:spark-cassandra-connector_2.12:3.2.0 --conf spark.sql.extensions=com.datastax.spark.connector.CassandraSparkExtensions  /home/base_models_learning/streaming.py

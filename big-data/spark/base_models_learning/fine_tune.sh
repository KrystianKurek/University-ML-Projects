#!/bin/bash
../../spark/bin/spark-submit --master spark://spark-master:7077 --packages com.datastax.spark:spark-cassandra-connector_2.12:3.2.0 \
  --conf spark.sql.extensions=com.datastax.spark.connector.CassandraSparkExtensions train_base.py
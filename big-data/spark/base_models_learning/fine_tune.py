# from pyspark.context import SparkContext
# from pyspark.sql.session import SparkSession
# import pyspark.sql.functions as sf
# from datetime import date
# import datetime
# import os

#sc = SparkContext.getOrCreate()
#spark = SparkSession(sc)
particles = ['NO2', 'O3', 'PM25', 'PM10']
weather_columns = ['coord_lon','coord_lat',
'weather','base','main_temp',
'main_feels_like','main_temp_min','main_temp_max',
'main_pressure','main_humidity','visibility','wind_speed','wind_deg','wind_gust','clouds_all',
'dt','sys_type','sys_id','sys_country','sys_sunrise','sys_sunset','timezone','id',
'name','cod','main_sea_level','main_grnd_level','time','snow_1h','rain_1h']
final_columns = ['NO2_present', 'O3_present', 'PM10_present', 'PM25_present', 'NO2_future',
                 'O3_future', 'PM10_future', 'PM25_future', 'main_temp', 'main_pressure', 'main_humidity',
                 'wind_speed', 'wind_deg', 'clouds_all', 'snow_1h', 'rain_1h', 'month', 'hour', 'weekday'
                 ]

def list_files(path):
    URI = sc._gateway.jvm.java.net.URI
    Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
    FileSystem = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
    Configuration = sc._gateway.jvm.org.apache.hadoop.conf.Configuration

    fs = FileSystem.get(URI("hdfs://namenode:8020"), Configuration())

    status = fs.listStatus(Path(path))
    return [str(file_status.getPath()) for file_status in status]


def return_date_str_from_filename(filename):
    basename = os.path.basename(filename)
    str_date = basename.replace('parquet_data_', '')
    return str_date


def read_date_from_filename(filename):
    str_date = return_date_str_from_filename(filename)
    file_date = datetime.datetime.strptime(
        str_date, "%Y-%m-%d-%H-%M-%S").date()
    return file_date


def return_newest_files(path, time_offset_in_days=7):
    today = datetime.date.today()
    all_files = list_files(path)
    filtered_files = [file for file in all_files if (today -
                      read_date_from_filename(file)).days <= time_offset_in_days]
    return filtered_files


def read_union_and_add_date_col(path):
    files = return_newest_files(path)
    whole_data = None
    for file in files:
        df = spark.read.parquet(file)
        str_date = return_date_str_from_filename(file)
        
        # delete seconds
        str_date = '-'.join(str_date.split('-')[:-1])
        df = df.select('*', sf.lit(str_date).alias('time'))
        if 'snow_1h' not in df.columns and 'coord_lon' in df.columns:
            df = df.select('*', sf.lit(float('nan')).alias('snow_1h'))

        if 'rain_1h' not in df.columns and 'coord_lon' in df.columns:
            df = df.select('*', sf.lit(float('nan')).alias('rain_1h'))
        if 'coord_lon' in df.columns: 
            df = df.select(*weather_columns)
        if whole_data is None:
            whole_data = df
        else:
            whole_data = whole_data.union(df)
        
    return whole_data


air = read_union_and_add_date_col('/tmp/AIRRESULTS')
weather = read_union_and_add_date_col('/tmp/WEATHERRESULTS')

air = air.filter(air.sensor_param_formula.isin(particles))\
    .select('*',
            sf.round('station_gegrLon', 4).alias('lng'),
            sf.round('station_gegrLat', 4).alias('lat'))\
    .drop(*['station_id',
            'station_name',
            'sensor_param_formula',
            'sensor_id',
            'sensor_param_id',
            'sensor_param_code',
            'measurement_date',
            'sensor_param_name',
            'station_gegrLon',
            'station_gegrLat'])\
    .withColumnRenamed('measurement_value', 'present')\
    .groupBy(['time', 'lat', 'lng'])\
    .pivot('measurement_key')\
    .agg(
        sf.first('present').alias('present'),
        sf.first('present').alias('tmp'))
air = air.drop(*[col for col in air.columns if 'tmp' in col])
not_present_particles = [
    f'{particle}_present' for particle in particles if f'{particle}_present' not in air.columns]
for col in not_present_particles:
    air = air.select('*', sf.lit(None).alias(col))
air = air.select('*', sf.unix_timestamp('time',
                 format="yyyy-MM-dd-kk-mm").alias('time_present'))
future_air = air
for col in future_air.columns:
    if 'time' not in col:
        future_air = future_air.withColumnRenamed(
            col, col.replace('present', 'future'))
future_air = future_air.select('*', (sf.col('time_present') + sf.lit(3600)).alias('time_future'))\
    .drop('time_present')
air = air.alias('air').join(future_air.alias('future_air'), (air.time_present == future_air.time_future) & (
    air.lat == future_air.lat) & (
    air.lng == future_air.lng), 'left')\
    .select('air.*', *[f'future_air.{particle}_future' for particle in particles])\
    .select('*', sf.to_date(sf.from_unixtime('time_present')).alias('time_date'))\
    .select('*',
        ((sf.dayofweek('time_date') + sf.lit(5)) % sf.lit(7)).alias('weekday'),
        sf.hour('time_date').alias('hour'),
        sf.month('time_date').alias('month'))\
    .drop('time_date')


weather = weather.select('time',
                         'main_temp',
                         'main_pressure',
                         'main_humidity',
                         'wind_speed',
                         'wind_deg',
                         'clouds_all',
                         'snow_1h',
                         'rain_1h',
                         sf.round('coord_lon', 4).alias('lng'),
                         sf.round('coord_lat', 4).alias('lat'))
data = weather.alias('weather').join(air.alias('air'),
                                     (weather.time == air.time) & (
    weather.lat == air.lat) & (
    weather.lng == air.lng), 'inner')
data = data.select(
    *[f'air.{col}' for col in air.columns],
    *[f'weather.{col}' for col in weather.columns if col not in ['time', 'lat', 'lng']]
)
data = data.select(*final_columns)

from train_functions import train_models
df = spark.read.options(inferSchema='True', header='True')\
    .csv("hdfs://namenode:8020/complete_data.csv")
df.select(*final_columns)
train_models(data.union(df), types=('GB',))
sc.stop()

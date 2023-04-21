CREATE KEYSPACE IF NOT EXISTS apache_air
        WITH REPLICATION = {
            'class' : 'SimpleStrategy',
            'replication_factor' : 1
        };

-------------- batch_views ---
DROP TABLE IF EXISTS apache_air.batch_views;

CREATE TABLE apache_air.batch_views (
    timestamp timestamp,
    particle text,
    interval tinyint,
    mre float,
    PRIMARY KEY((particle), interval)
);


-------------- realtime_views ---
DROP TABLE IF EXISTS apache_air.realtime_views;

CREATE TABLE apache_air.realtime_views (
    longtitude float,
    latitude float,
    station_name text,
    timestamp timestamp,
    particle text,
    prediction float,
    PRIMARY KEY((longtitude, latitude), particle)
);

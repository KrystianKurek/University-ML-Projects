#!/usr/bin/env python3
import glob
import os
from collections import Counter
from datetime import datetime
import requests
import helpers
import json
from functools import wraps
import traceback
import tqdm
import warnings

"""
Instruction:
crontab -e (how to change editor https://www.howtogeek.com/410995/how-to-change-the-default-crontab-editor/)
1 * * * * cd path/to/files && python3 airAPI.py &>> airAPI.log
"""
variables = helpers.read_project_variables()

functions_to_paths = {
    'get_sensors_measurements': variables['measurements_dir'],
    'get_measurement_sensors_data': variables['sensors_dir'],
    'get_stations': variables['stations_path']
}


def error_handler(func):
    @wraps(func)
    def with_except(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.ConnectionError as e:
            print(f'Function {func.__name__} called with args: {args} thrown requests.exceptions.ConnectionError')
            print(e)
            print(traceback.format_exc())
            path = functions_to_paths[func.__name__]
            path = os.path.join(path, str(args[0])) if args else path
            try:
                with open(path, 'r') as file:
                    return json.load(file)
            except FileNotFoundError:
                return []

    return with_except


def deprecated_function(func):
    @wraps(func)
    def with_except(*args, **kwargs):
        warnings.warn(f"Function {func.__name__} is deprecated! ", DeprecationWarning)
        return func(*args, **kwargs)

    return with_except


@error_handler
def get_stations():
    response = requests.get(variables['stations_url'])
    return response.json()


def save_stations(stations):
    if not os.path.exists(os.path.dirname(variables['stations_path'])):
        os.mkdir(os.path.dirname(variables['stations_path']))
    with open(variables['stations_path'], 'w') as file:
        json.dump(stations, file)


@error_handler
def get_sensors_measurements(sensors_id):
    station_url = os.path.join(variables['sensors_measurements_url'], str(sensors_id))
    response = requests.get(station_url)
    return response.json()


@error_handler
def get_measurement_sensors_data(station_id):
    station_url = os.path.join(variables['sensors_data_url'], str(station_id))
    response = requests.get(station_url)
    return response.json()


def save_sensors(sensors, station_id):
    if not os.path.exists(variables['sensors_dir']):
        os.mkdir(variables['sensors_dir'])
    with open(os.path.join(variables['sensors_dir'], f'{station_id}.json'), 'w') as file:
        json.dump(sensors, file)


def save_measurements(measurements, sensor_idx):
    if not os.path.exists(variables['measurements_dir']):
        os.mkdir(variables['measurements_dir'])
    with open(os.path.join(variables['measurements_dir'], f'{sensor_idx}.json'), 'w') as file:
        json.dump(measurements, file)


def perform_measurement_update_if_needed(sensor_id):
    if check_measurement_update(sensor_id):
        measurement = get_sensors_measurements(sensor_id)
        save_measurements(measurement, sensor_id)
        print(f"Updated sensor: {sensor_id}")
    else:
        print(f"Sensor {sensor_id} did not need to be updated")


@deprecated_function
def check_update():
    maxes = Counter()
    for file in glob.glob(os.path.join(variables['measurements_dir'], '*json')):
        with open(file, 'r') as f:
            data = json.load(f)
        if data and data['values']:
            dates = [datetime.strptime(row['date'], '%Y-%m-%d %H:%M:%S') for row in data['values']]
            maxes[max(dates)] += 1
        else:
            maxes[datetime.strptime('1970-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')] += 1
    now = datetime.now()
    if maxes:
        dates_flag = (now - maxes.most_common()[0][0]).seconds / 3600 > 1.002  # hour and 7.2seconds
    else:
        dates_flag = True
    return dates_flag and not read_save_control()['saving']


def check_measurement_update(sensor_id):
    measurements = read_measurements_from_file(sensor_id)
    if 'error' in measurements:  # If file does not exist and it should
        sensors_ids = read_sensors_ids()
        return int(sensor_id) in sensors_ids  # we download it

    if measurements and measurements['values']:
        dates = [datetime.strptime(row['date'], '%Y-%m-%d %H:%M:%S') for row in measurements['values']]
        max_date = max(dates)
    else:
        max_date = datetime.strptime('1970-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    now = datetime.now()
    dates_flag = (now - max_date).seconds / 3600 > 1.002  # hour and 7.2seconds
    return dates_flag


@deprecated_function
def perform_update_if_needed():
    if check_update():
        save_save_control({'saving': True})
        save_all()
        save_save_control({'saving': False})


@deprecated_function
def read_save_control():
    try:
        with open(variables['save_control_path'], 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {'saving': False}


@deprecated_function
def save_save_control(save_control):
    dir_name = os.path.dirname(variables['save_control_path'])
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    with open(variables['save_control_path'], 'w') as file:
        return json.dump(save_control, file)


def read_measurements_from_file(sensor_id):
    try:
        with open(os.path.join(variables['measurements_dir'], f'{sensor_id}.json'), 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {'error': f'sensor {sensor_id} not found'}


def save_all():
    stations = get_stations()
    all_sensor_ids = []
    save_stations(stations)
    for station in tqdm.tqdm(stations):
        # time.sleep(0.1)
        sensors = get_measurement_sensors_data(station['id'])
        save_sensors(sensors, station['id'])
        for sensor in sensors:
            # time.sleep(0.1)
            all_sensor_ids.append(sensor['id'])
            measurements = get_sensors_measurements(sensor['id'])
            save_measurements(measurements, sensor['id'])
    return all_sensor_ids


def read_sensors_ids():
    if not os.path.exists(variables['sensor_ids_list']):
        raise FileNotFoundError("You have to first launch airAPI.py")
    with open(variables['sensor_ids_list'], 'r') as file:
        return json.load(file)


def save_sensors_ids(ids):
    with open(variables['sensor_ids_list'], 'w') as file:
        json.dump(ids, file)


if __name__ == '__main__':
    save_all()

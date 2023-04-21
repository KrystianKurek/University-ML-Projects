#!/usr/bin/env python3
import os

import helpers
from flask import jsonify, Flask
import json
import numpy as np
from datetime import datetime
import airAPI
variables = helpers.read_project_variables()
app = Flask(__name__)

"""
nohup /path/to/files/mockAPI.py >> /path/to/files/mockAPI.log &
"""


def print_log(log):
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f'{now} - {log}')


@app.route('/station/findAll', methods=['GET'])
def send_stations():
    with open(variables['stations_path'], 'r') as file:
        return jsonify(json.load(file))


@app.route('/station/sensors/<station_id>', methods=['GET'])
def send_sensors(station_id):
    try:
        with open(os.path.join(variables['sensors_dir'], f'{station_id}.json'), 'r') as file:
            return jsonify(json.load(file))
    except FileNotFoundError:
        return jsonify({'error': f'station {station_id} not found'})


def add_noise(value, mean, var):
    if value is None:
        return None
    randomized_value = value + np.random.normal(mean, var)
    return np.abs(randomized_value)


@app.route('/data/getData/<sensor_id>', methods=['GET'])
def send_measurements(sensor_id):
    airAPI.perform_measurement_update_if_needed(sensor_id)

    measurements = airAPI.read_measurements_from_file(sensor_id)
    if 'error' in measurements:
        return jsonify(measurements)

    key = measurements['key']
    mean, var = variables[f'{key}_mean'], variables[f'{key}_var']

    measurements['values'] = [{'date': adict['date'], 'value': add_noise(adict['value'], mean, var)}
                              for adict in measurements['values']]
    return jsonify(measurements)


if __name__ == '__main__':
    app.run(port=variables['port'], host="0.0.0.0")

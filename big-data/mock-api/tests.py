import os
import glob
import requests
from requests import HTTPError

import helpers

variables = helpers.read_project_variables()

URL = 'http://127.0.0.1:' + str(variables['port'])


def test_random_results():
    print('Test randomness of results:')
    first_request = f'{URL}/data/getData/50'
    print(f'First request to {first_request}')
    try:
        first_response = requests.get(first_request).json()
    except requests.exceptions.ConnectionError:
        print(f'Error occurred - make sure that mock-API is running.')
        return
    first_value = first_response['values'][0]['value']
    first_date = first_response['values'][0]['date']

    second_request = f'{URL}/data/getData/50'
    print(f'Second request to {second_request}')

    second_response = requests.get(second_request).json()
    try:
        second_value = [row['value'] for row in second_response['values'] if row['date'] == first_date][0]
    except IndexError:
        print('No values in second request. Test failed.')
        return

    if first_value != second_value:
        print('Values are not the same. Test passed.')
    else:
        print('Values are the same. Test failed.')


def test_update():
    print('Test updating measurements from source API: ')
    file_to_delete = glob.glob(os.path.join(variables['measurements_dir'], '*'))[0]
    print(f'Deleting file {file_to_delete}')
    os.remove(file_to_delete)
    measurement_id = os.path.basename(file_to_delete).replace('.json', '')
    request_url = f'{URL}/data/getData/{measurement_id}'
    print(f'Request to {request_url}')

    try:
        requests.get(request_url)
    except requests.exceptions.ConnectionError:
        print(f'Error occurred - make sure that mock-API is running.')
        return

    if os.path.exists(file_to_delete):
        print(f'File {file_to_delete} exists. Test passed.')
    else:
        print(f'File {file_to_delete} does not exist. Test failed.')


test_random_results()
print()
test_update()

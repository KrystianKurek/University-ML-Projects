import yaml


def read_project_variables():
    with open("variables.yaml", 'r') as file:
        return yaml.safe_load(file)

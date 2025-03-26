from pathlib import Path
import os.path
import yaml

def read_config_setting(key: str) -> str:
    # Read data source file location from yaml file
    current_directory = Path(__file__).resolve().parent
    parent_directory = current_directory.parent.parent

    with open(os.path.join(parent_directory, 'config\\app_config.yaml'), 'r') as file:
        config = yaml.safe_load(file)
    return config['settings'][key]

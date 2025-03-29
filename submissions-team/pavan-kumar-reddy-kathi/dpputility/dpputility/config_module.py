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

def get_tuning_result_file_path(root_path:str, file_name:str):
    """
    Constructs and returns the file path where tuning results gets saved.
    :param root_path: Root path, passed from caller
    :param file_name: Name of the file to be saved
    :return: Storage location to GridSearchCV results file
    """
    folder_path = os.path.join(root_path, read_config_setting('grid_search_cv_un_tuned'))
    return os.path.join(folder_path, file_name)
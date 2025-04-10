import numpy as np
import pandas as pd
from sklearn.metrics import (r2_score, root_mean_squared_error,
                             mean_absolute_error, make_scorer)
import re
from typing import Any
import json

from sklearn.model_selection import KFold, GridSearchCV

from dpputility.json_module import NumpyEncoder


def calculate_model_metrics(y_train:np.ndarray, y_test:np.ndarray,
                            y_train_predicted:np.ndarray, y_test_predicted:np.ndarray,
                            features_count:int=9) -> pd.DataFrame:
    """
    Method to calculate metrics for input regression model
    :param y_train: Training set for dependent variable vector
    :param y_test: Test set for dependent variable vector
    :param y_train_predicted: predicted value of dependent variable vector based on Training set
    :param y_test_predicted: predicted value of dependent variable vector based on Test set
    :param features_count: predicted value of dependent variable vector based on Test set
    :return: Pandas DataFrame with metrics information
    """
    # calculate r2 score
    r2_score_train = r2_score(y_train, y_train_predicted)
    r2_score_test  = r2_score(y_test, y_test_predicted)

    # calculate adjusted r2 score
    n = len(y_train)
    adjusted_r2_score_train = 1 - ((1 - r2_score_train) * ((n - 1) / (n - 1 - features_count)))
    n = len(y_test)
    adjusted_r2_score_test = 1 - ((1 - r2_score_test) * ((n - 1) / (n - 1 - features_count)))

    # calculate rmse score
    rmse_train = root_mean_squared_error(y_train, y_train_predicted)
    rmse_test = root_mean_squared_error(y_test, y_test_predicted)

    #calculate mae
    mae_train = mean_absolute_error(y_train, y_train_predicted)
    mae_test = mean_absolute_error(y_test, y_test_predicted)

    # calculate train and test metrics delta
    r2_delta = abs(r2_score_train - r2_score_test)
    adjusted_r2_delta = abs(adjusted_r2_score_train - adjusted_r2_score_test)
    rmse_delta = abs(rmse_train - rmse_test)
    mae_delta = abs(mae_train - mae_test)

    return pd.DataFrame([[r2_score_train, r2_score_test, r2_delta, adjusted_r2_score_train, adjusted_r2_score_test,
                         adjusted_r2_delta, rmse_train, rmse_test, rmse_delta, mae_train, mae_test, mae_delta]],
                        columns=['r2_score_train', 'r2_score_test', 'r2_delta', 'adjusted_r2_score_train',
                                 'adjusted_r2_score_test','adjusted_r2_delta', 'rmse_train', 'rmse_test', 'rmse_delta',
                                 'mae_train', 'mae_test', 'mae_delta'])




def calculate_grid_search_cv_metrics(results:dict) -> pd.DataFrame:
    """
    Captures critical metrics from input grid_search_cv results and returns them as DataFrame Instance
    :param results: Dictionary containing GridSearchCV results
    :return: DataFrame Instance populated with critical metrics from GridSearchCV results
    """
    split_test_r2_values = get_values(r'split[0-9]+_test_r2', results)
    split_train_r2_values = get_values(r'split[0-9]+_train_r2', results)
    split_test_rmse_values = get_values(r'split[0-9]+_test_rmse', results)
    split_train_rmse_values = get_values(r'split[0-9]+_train_rmse', results)
    split_test_mae_values = get_values(r'split[0-9]+_test_mae', results)
    split_train_mae_values = get_values(r'split[0-9]+_train_mae', results)

    return pd.DataFrame(
                            [
                                        [
                                            min(split_test_r2_values), min(split_train_r2_values),
                                            max(split_test_r2_values),  max(split_train_r2_values),
                                            results['mean_test_r2'], results['mean_train_r2'],
                                            results['mean_test_r2'] - results['mean_train_r2'],
                                            min(split_test_rmse_values), min(split_train_rmse_values),
                                            max(split_test_rmse_values), max(split_train_rmse_values),
                                            results['mean_test_rmse'],  results['mean_train_rmse'],
                                            results['mean_test_rmse'] - results['mean_train_rmse'],
                                            min(split_test_mae_values), min(split_train_mae_values),
                                            max(split_test_mae_values), max(split_train_mae_values),
                                            results['mean_test_mae'], results['mean_train_mae'],
                                            results['mean_test_mae'] - results['mean_train_mae']
                                        ]
                                  ],
                            columns=[
                                        'test_r2_min_value', 'train_r2_min_value',
                                        'test_r2_max_value', 'train_r2_max_value',
                                        'test_r2_mean_value', 'train_r2_mean_value',
                                        'delta_r2_mean_value',
                                        'test_rmse_min_value', 'train_rmse_min_value',
                                        'test_rmse_max_value', 'train_rmse_max_value',
                                        'test_rmse_mean_value', 'train_rmse_mean_value',
                                        'delta_rmse_mean_value',
                                        'test_mae_min_value', 'train_mae_min_value',
                                        'test_mae_max_value', 'train_mae_max_value',
                                        'test_mae_mean_value', 'train_mae_mean_value',
                                        'delta_mae_mean_value',
                                    ]
                        )


def get_matching_keys(pattern:str, results:dict) -> list:
    """
    Returns list of all matching keys (from input results: dictionary) with input pattern.
    :param pattern: Regex pattern to look for in input results: dictionary
    :param results: Instance of dict
    :return: list of all matching keys to input pattern
    """
    return [key for key in results if re.match(pattern, key)]

def get_values(pattern:str, results:dict) -> list:
    """
    Returns list of values, from input results: dictionary, whose keys are
    matching with input pattern.
    :param pattern: Regex pattern to look for in input results: dictionary
    :param results: Instance of dict
    :return: list of values whose keys are
    matching with input pattern.
    """
    keys = get_matching_keys(pattern, results)
    return [results[key][0] for key in keys]

def perform_tuning(model: Any, param_grid: dict | list,
                        X:np.ndarray, y:np.ndarray, path_to_save:str,
                   n_splits:int=10)->GridSearchCV:
    """
    Performs hyperparameter tuning and saves results as json file at :path_to_save
    :param model: Model to fit
    :param param_grid: Hyperparameters to tune
    :param X: Matrix of features
    :param y: Dependent variable vector
    :param path_to_save: Location where tuning results saved
    :param n_splits: Number of folds. Must be at least 2. Default value 10.
    :return: Instance of GridSearchCV
    """

    scores_dictionary = {'r2': make_scorer(r2_score),
                         'rmse': make_scorer(root_mean_squared_error),
                         'mae': make_scorer(mean_absolute_error)}

    k_fold = KFold(n_splits=10, shuffle=True)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               cv=k_fold, scoring=scores_dictionary,
                               return_train_score=True, verbose=False,
                               refit='rmse', n_jobs=-1)

    grid_search.fit(X, y)

    with open(path_to_save, "w") as file:
        json.dump(grid_search.cv_results_, file, indent=4, cls=NumpyEncoder)

    return grid_search


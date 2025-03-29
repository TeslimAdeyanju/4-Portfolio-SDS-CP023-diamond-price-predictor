import json
from typing import Any

import numpy as np
from sklearn.metrics import (make_scorer, root_mean_squared_error,
                             r2_score, mean_absolute_error)
from sklearn.model_selection import KFold, GridSearchCV

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert ndarray to list
        return super().default(obj)

def perform_tuning(model: Any, param_grid: dict | list,
                        X:np.ndarray, y:np.ndarray, path_to_save:str)->GridSearchCV:
    """
    Performs hyperparameter tuning and saves results as json file at :path_to_save

    :param model: Model to fit
    :param param_grid: Hyperparameters to tune
    :param X: Matrix of features
    :param y: Dependent variable vector
    :param path_to_save: Location where tuning results saved
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
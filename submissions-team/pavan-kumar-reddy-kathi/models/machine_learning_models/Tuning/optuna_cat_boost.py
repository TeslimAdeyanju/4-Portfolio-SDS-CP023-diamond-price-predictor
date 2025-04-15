import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from catboost import CatBoostRegressor
import optuna

from dpputility import data_set_module as dsm

# Hyper tuning parameter didn't result in any significant improvement,
# either in R2 score or reduction in Test set RMSE, without raising
# in difference between Train and Test RMSE values.
# Hence, sticking to default model parameters.

pd.set_option('display.max_columns', None)

# Load data set
dataset = dsm.get_data_frame()
# print(dataset.describe())

# Split the dataset into independent variable 2d matrix and dependent variable vector
X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values

# Split the data into Training and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

def optuna_objective(trial):
    params = {
        # "iterations": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.01, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_int("l2_leaf_reg",2, 10),
        "random_strength": trial.suggest_int("random_strength", 0, 10)
        # "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        # "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
        # "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }

    model = CatBoostRegressor(**params, silent=True)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_predict)
    return rmse


study = optuna.create_study(direction='minimize')
study.optimize(optuna_objective, n_trials=40)
print(study.best_value)
print(study.best_params)

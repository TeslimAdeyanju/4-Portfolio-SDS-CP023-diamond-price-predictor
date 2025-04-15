import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from lightgbm import LGBMRegressor

from dpputility import data_set_module as dsm

# Hyper tuning parameter didn't result in any significant improvement in RMSE

pd.set_option('display.max_columns', None)

# Load data set
dataset = dsm.get_data_frame()
# print(dataset.head())

# Split the dataset into independent variable 2d matrix and dependent variable vector
X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values

# Split the data into Training and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

def objective_lightgbm(trial):
    params = {
        'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.01, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'feature_fraction': trial.suggest_float('feature_fraction',0.1,1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 0, 10),
        'max_depth': trial.suggest_int('max_depth', 15, 100),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 200)
    }
    model = LGBMRegressor(**params, silent=True)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_predict)
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective_lightgbm, n_jobs=-1, n_trials=50)
print(study.best_value)
print(study.best_params)











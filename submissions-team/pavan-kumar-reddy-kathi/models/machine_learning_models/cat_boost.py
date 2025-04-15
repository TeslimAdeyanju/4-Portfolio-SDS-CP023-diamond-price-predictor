import os
from pathlib import Path
import joblib

import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

from dpputility import (data_set_module as dsm,
                        config_module as cm, metrics_module as mm)

# Hyper tuning parameter doesn't result in any significant improvement,
# either in R2 score or reduction in Test set RMSE, without raising
# in difference between Train and Test RMSE values.
# Hence, sticking to default model parameters.

pd.set_option('display.max_columns', None)

# Load data set
dataset = dsm.get_data_frame(False)
# print(dataset.info())

# Split the dataset into independent variable 2d matrix and dependent variable vector
X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values

# Split the data into Training and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Perform tuning using GridSearchCV and save results
path_to_save = cm.get_tuning_result_file_path(os.path.abspath('../'),
                                              'cat_boost.json')

params_grid = [{'learning_rate' : [0.05, 0.07, 0.09, 1.1]}]
grid_search_cv = mm.perform_tuning(CatBoostRegressor(verbose=False), [{}],
                        X_train, y_train, path_to_save)
print(grid_search_cv.best_params_)
print(grid_search_cv.best_score_)
print(mm.calculate_grid_search_cv_metrics(grid_search_cv.cv_results_))

# Build and train model
# Do sampling
model = CatBoostRegressor(verbose=False)
# Do Sampling
model.fit(X_train, y_train)

# Inference
y_test_predicted = model.predict(X_test)
y_train_predicted = model.predict(X_train)
print(model.get_params())
# display model metrics
print(mm.calculate_model_metrics(y_train, y_test, y_train_predicted, y_test_predicted, (X_test.shape[1])))

# save the model
root_directory = Path.cwd().parent.parent
relative_path = 'api/model/prediction_model.pkl'
final_path = root_directory/relative_path
with open(final_path, 'wb') as file:
    joblib.dump(model,file)

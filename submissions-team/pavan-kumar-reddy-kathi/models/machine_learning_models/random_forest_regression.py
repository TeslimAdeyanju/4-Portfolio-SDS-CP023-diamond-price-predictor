import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from dpputility import (data_set_module as dsm,
                        config_module as cm, metrics_module as mm)
from dpputility.json_module import perform_tuning

pd.set_option('display.max_columns', None)

# Load data set
dataset = dsm.get_data_frame()

# Split the dataset into independent variable 2d matrix and dependent variable vector
X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values

# Split the data into Training and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Perform tuning using GridSearchCV and save results
path_to_save = cm.get_tuning_result_file_path(os.path.abspath('../'),
                                              'random_forest_regression.json')

grid_search_cv = perform_tuning(RandomForestRegressor(random_state=0), [{}],
                        X_train, y_train, path_to_save)

print(mm.calculate_grid_search_cv_metrics(grid_search_cv.cv_results_))

# Build and train model
model = RandomForestRegressor(random_state=0)
model.fit(X_train, y_train)

# Inference
y_test_predicted = model.predict(X_test)
y_train_predicted = model.predict(X_train)

# display model metrics
print(mm.calculate_model_metrics(y_train, y_test, y_train_predicted, y_test_predicted, (X_test.shape[1])))

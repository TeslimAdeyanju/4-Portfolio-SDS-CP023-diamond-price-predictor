import os

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.svm import LinearSVR

from dpputility.json_module import perform_tuning
from dpputility import (data_set_module as dsm,
                        config_module as cm, metrics_module as mm)

pd.set_option('display.max_columns', None)

# Load data set
dataset = dsm.get_data_frame()

# Split the dataset into independent variable 2d matrix and dependent variable vector
X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values

# Split the data into Training and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Apply standard scaling to all independent variables except categorical
numerical_features = [0, 1, 2, 3, 4, 5]
categorical_features = [6, 7, 8]

# Define transformer for numerical features
standard_scaler_transformer = StandardScaler()

column_transformer_object = ColumnTransformer(transformers=[('numerical',standard_scaler_transformer,numerical_features),
                                                            ('categorical','passthrough',categorical_features)])
X_train = column_transformer_object.fit_transform(X_train)

# Perform tuning using GridSearchCV and save results
path_to_save = cm.get_tuning_result_file_path(os.path.abspath('../'),
                                              'linear_svr.json')

grid_search_cv = perform_tuning(LinearSVR(), [{}],
                        X_train, y_train, path_to_save)

print(mm.calculate_grid_search_cv_metrics(grid_search_cv.cv_results_))

# Train Model
model = LinearSVR()
model.fit(X_train, y_train)

# Inference
y_test_predicted = model.predict(column_transformer_object.transform(X_test))
y_train_predicted = model.predict(X_train)

# display model metrics
print(mm.calculate_model_metrics(y_train, y_test, y_train_predicted, y_test_predicted, (X_test.shape[1])))



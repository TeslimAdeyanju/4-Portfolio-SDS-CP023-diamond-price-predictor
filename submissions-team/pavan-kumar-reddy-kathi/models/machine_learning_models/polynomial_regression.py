import os

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from dpputility import (data_set_module as dsm,
                        config_module as cm, metrics_module as mm)

def get_polynomial_pipeline() -> Pipeline:
    # Apply standard scaling to all independent variables except categorical
    numerical_features = [0, 1, 2, 3, 4, 5]
    categorical_features = [6, 7, 8]
    standard_scaler_object = StandardScaler()
    column_transformer_object = ColumnTransformer(transformers=[('numerical'
                                                                 , Pipeline([('scale', standard_scaler_object),
                                                                             ('polynomial',
                                                                              PolynomialFeatures(degree=1))]),
                                                                 numerical_features),
                                                                ('categorical', 'passthrough', categorical_features)])
    # Define & return the pipeline
    return Pipeline([('preprocessor', column_transformer_object),
                                ('linearregression', LinearRegression())])

pd.set_option('display.max_columns', None)

# Load data set
dataset = dsm.get_data_frame()

# Split the dataset into independent variable 2d matrix and dependent variable vector
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Split the data into Training and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

# Perform tuning using GridSearchCV and save results
path_to_save = cm.get_tuning_result_file_path(os.path.abspath('../'),
                                              'polynomial_regression.json')

grid_search_cv = mm.perform_tuning(get_polynomial_pipeline(), [{}],
                        X_train, y_train, path_to_save)

print(mm.calculate_grid_search_cv_metrics(grid_search_cv.cv_results_))

# Train Model
# Create Fresh Model
pipeline_object = get_polynomial_pipeline()
pipeline_object.fit(X_train, y_train)

# Inference
y_test_predicted = pipeline_object.predict(X_test)
y_train_predicted = pipeline_object.predict(X_train)

# display model metrics
print(mm.calculate_model_metrics(y_train, y_test, y_train_predicted, y_test_predicted, (X_test.shape[1])))




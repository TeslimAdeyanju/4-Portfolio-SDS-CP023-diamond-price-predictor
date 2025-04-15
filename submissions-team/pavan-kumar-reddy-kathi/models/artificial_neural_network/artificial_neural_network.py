import os

import  pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from dpputility import data_set_module as dsm, metrics_module as mm
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)
os.environ['OMP_NUM_THREADS'] = '2'

pd.set_option('display.max_columns', None)
def get_ann_regression_model() -> tf.keras.models.Sequential:
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=6, activation='relu'))
    model.add(tf.keras.layers.Dense(units=6, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error',  metrics=['mse'])
    return model

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

ann_regression = get_ann_regression_model()

ann_regression.fit(X_train, y_train, epochs=100, batch_size=32,verbose=0)

y_test_predicted = ann_regression.predict(column_transformer_object.transform(X_test))
y_train_predicted = ann_regression.predict(X_train)

# display model metrics
print(mm.calculate_model_metrics(y_train, y_test, y_train_predicted, y_test_predicted, (X_test.shape[1])))





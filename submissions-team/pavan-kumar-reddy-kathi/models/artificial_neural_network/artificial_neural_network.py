import os
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from dpputility import data_set_module as dsm
import tensorflow as tf
from scikeras.wrappers import KerasRegressor
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)
os.environ['OMP_NUM_THREADS'] = '2'

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

ann_regression = tf.keras.models.Sequential()
ann_regression.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann_regression.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann_regression.add(tf.keras.layers.Dense(units=1))

ann_regression.compile(optimizer='adam', loss='mean_squared_error')
ann_regression.fit(X_train, y_train, epochs=100, batch_size=32,verbose=0)

y_predict = ann_regression.predict(column_transformer_object.transform(X_test))
print(r2_score(y_true=y_test, y_pred=y_predict)) # 0.9545066356658936

# calculate adjusted r2 score
n = len(y_test)
print(1- ((1-r2_score(y_test, y_predict))*((n-1)/(n-1-9)))) #0.9544686330411482

ann_model =  KerasRegressor(build_fn = ann_regression,
                                           epochs=100, batch_size=32, verbose=0)

# K fold cross validation
# k_fold = KFold(n_splits=10, shuffle=True)
# accuracies = cross_val_score(ann_model, X_train, y_train, cv=k_fold, scoring='r2', n_jobs=-1)
# # print(accuracies)
# print(accuracies.mean())
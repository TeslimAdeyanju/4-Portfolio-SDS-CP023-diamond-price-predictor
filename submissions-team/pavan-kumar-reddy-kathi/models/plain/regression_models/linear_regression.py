from dpputility import data_set_module as dsm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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
# standard_scaler_y = StandardScaler()
column_transformer_object = ColumnTransformer(transformers=[('numerical',standard_scaler_transformer,numerical_features),
                                                            ('categorical','passthrough',categorical_features)])
X_train = column_transformer_object.fit_transform(X_train)
# y_train = standard_scaler_y.fit_transform(y_train.reshape(-1,1)).flatten()
# print(y_train)

# Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Perform predictions
y_predict = model.predict(column_transformer_object.transform(X_test))
# y_predict = standard_scaler_y.inverse_transform(y_predict.reshape(-1,1)).flatten()

# calculate r2 score
print(r2_score(y_test, y_predict))

# calculate adjusted r2 score
n = len(y_test)
print(1- ((1-r2_score(y_test, y_predict))*((n-1)/(n-1-9))))

# without applying Standard Scaler to y
# 0.9121553119762901 - R2
# 0.9120819314126912 - Adjusted R2

# Applying Standard Scaler to y
# 0.9121553119762901 - R2
# 0.9120819314126912 - Adjusted R2

# K fold cross validation
k_fold = KFold(n_splits=10, shuffle=True)
accuracies = cross_val_score(model, X_train, y_train, cv=k_fold, scoring='r2', n_jobs=-1)
print(accuracies)
print(accuracies.mean()) #0.9011613882692473 (no scaling to y) 0.9012567630086693(scaling applied to y)



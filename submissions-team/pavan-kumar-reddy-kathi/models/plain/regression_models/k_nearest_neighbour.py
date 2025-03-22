from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from dpputility import data_set_module as dsm
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor

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

# Build and train model
model = KNeighborsRegressor(n_neighbors=10, weights='distance')
model.fit(X_train, y_train)

y_predict = model.predict(column_transformer_object.transform(X_test))

print(r2_score(y_test, y_predict)) #0.9730994634942931

n = len(y_test)
print(1- ((1-r2_score(y_test, y_predict))*((n-1)/(n-1-9)))) #0.9730769922831782

# K fold cross validation
k_fold = KFold(n_splits=10, shuffle=True)
accuracies = cross_val_score(model, X_train, y_train, cv=k_fold, scoring='r2', n_jobs=-1)
print(accuracies)
print(accuracies.mean()) #0.9713804811855878
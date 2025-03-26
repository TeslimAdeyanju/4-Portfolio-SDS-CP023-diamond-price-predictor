from dpputility import data_set_module as dsm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.svm import LinearSVR,SVR

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

# Train Model
model = LinearSVR()
model.fit(X_train, y_train)

# Perform predictions
y_predict = model.predict(column_transformer_object.transform(X_test))

# calculate r2 score
print(r2_score(y_test, y_predict)) #0.8747595867077493

# calculate adjusted r2 score
n = len(y_test)
print(1- ((1-r2_score(y_test, y_predict))*((n-1)/(n-1-9)))) #0.8746549678364267

# K fold cross validation
k_fold = KFold(n_splits=10, shuffle=True)
accuracies = cross_val_score(model, X_train, y_train, cv=k_fold, scoring='r2', n_jobs=-1)
print(accuracies)
print(accuracies.mean())  #0.8676067363801101


from sklearn.linear_model import LinearRegression
from dpputility import data_set_module as dsm
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# Load data set
dataset = dsm.get_data_frame()

# Split the dataset into independent variable 2d matrix and dependent variable vector
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Split the data into Training and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

# Apply standard scaling to all independent variables except categorical
numerical_features = [0, 1, 2, 3, 4, 5]
categorical_features = [6, 7, 8]

standard_scaler_object = StandardScaler()

column_transformer_object = ColumnTransformer(transformers=[('numerical'
                            ,Pipeline([('scale',standard_scaler_object),
                                       ('polynomial', PolynomialFeatures(degree=2))]), numerical_features),
                            ('categorical','passthrough', categorical_features)])

# Define the pipeline
pipeline_object = Pipeline([('preprocessor', column_transformer_object),
                            ('linearregression',LinearRegression())])

# Train Model
pipeline_object.fit(X_train, y_train)

# Predictions
y_predict = pipeline_object.predict(X_test)

# R2 score
print(r2_score(y_test,y_predict))

# Adjusted R2 score
n = len(y_test)
print(1- ((1-r2_score(y_test, y_predict))*((n-1)/(n-1-9))))

# r2_score
# 0.5776145570089803

# K fold cross validation
k_fold = KFold(n_splits=10, shuffle=True)
accuracies = cross_val_score(pipeline_object, X_train, y_train, cv=k_fold, scoring='r2', n_jobs=-1)
print(accuracies)
print(accuracies.mean()) #-1.413860788581872



from dpputility import data_set_module as dsm
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# Load data set
dataset = dsm.get_data_frame()

# Split the dataset into independent variable 2d matrix and dependent variable vector
X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values

# Split the data into Training and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Build and train model
model = DecisionTreeRegressor(random_state=0)
model.fit(X_train, y_train)

y_predict = model.predict(X_test)

print(r2_score(y_test, y_predict))

n = len(y_test)
print(1- ((1-r2_score(y_test, y_predict))*((n-1)/(n-1-9))))

# 0.965769444642528 - R2
# 0.9657408503415983 - Adjusted R2

# K fold cross validation
k_fold = KFold(n_splits=10, shuffle=True)
accuracies = cross_val_score(model, X_train, y_train, cv=k_fold, scoring='r2', n_jobs=-1)
print(accuracies)
# print(accuracies.mean()) 0.9660345595339328
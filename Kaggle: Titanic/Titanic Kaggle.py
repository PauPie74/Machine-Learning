import pandas as pd
from sklearn.tree import DecisionTreeRegressor

titanic_path = "train.csv"
titanic_path_test = "test.csv"

titanic_data = pd.read_csv(titanic_path, index_col="PassengerId")
titanic_data_test = pd.read_csv(titanic_path_test, index_col="PassengerId")

feature_names = ['Survived','Pclass','Sex','Age', 'Fare']

#print(feature_names)

X_train = titanic_data[feature_names]
X_valid = titanic_data[feature_names]

y_train = titanic_data.Survived
y_valid = titanic_data.Survived

#print(X.describe)
#print(X.head)

s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

from sklearn.preprocessing import OrdinalEncoder

# Make copy to avoid changing original data
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

# Apply ordinal encoder to each column with categorical data
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])

print(label_X_train.head())

from sklearn.tree import DecisionTreeRegressor

titanic_model = DecisionTreeRegressor(random_state=7)

titanic_model.fit(label_X_train,y_train)
predictions = titanic_model.predict(X_train)
print(predictions)

print(y.head())


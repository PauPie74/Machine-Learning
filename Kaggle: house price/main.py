import pandas as pd

### Loading the model
iowa_file_path = "train.csv"
home_data = pd.read_csv(iowa_file_path)

### Reviewing
#print(home_data.describe())
#print(home_data.columns)

### y - prediction target
### X - predictive features

### Specify Prediction Target
y = home_data.SalePrice

# Create the list of features below
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_data[feature_names]

### Reviewing data
#print(X.describe)
#print(X.head)

### Fit model

from sklearn.tree import DecisionTreeRegressor

#For model reproducibility, set a numeric value for random_state when specifying the model
iowa_model = DecisionTreeRegressor(random_state=7)
iowa_model.fit(X,y)

### Make Predictions

prediction = iowa_model.predict(X)
#print(prediction)
#print(y.head())

### Model Validation

## Split data for calculating the error

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X,y,random_state = 1)

iowa_model = DecisionTreeRegressor(random_state= 1)

iowa_model.fit(train_X,train_y)

### Predictions with validation data
val_predictions = iowa_model.predict(val_X)


### Mean Absolute Error in Validation Data

from sklearn.metrics import mean_absolute_error

val_mae = mean_absolute_error(val_y,val_predictions)
print(val_mae)




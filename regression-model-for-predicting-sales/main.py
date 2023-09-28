import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Load using pandas
data = pd.read_excel("./Sample - Superstore.xls")

# Exploring the dataset
# print(data.head())
# print(data.info())

# Select relevant features. Used for training and prediction
features = ['Sales', 'Profit', 'Quantity', 'Discount']

# Remove rows with missing values
data = data[features].dropna()

# Split the data into input (X) and target (y) variables
X = data.drop('Sales', axis=1)  # Input
y = data['Sales']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating an instance of the Linear Regression model
model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Example prediction
new_data = pd.DataFrame({'Profit': [100], 'Quantity': [5], 'Discount': [0.1]})
prediction = model.predict(new_data)
print("Prediction:", prediction)

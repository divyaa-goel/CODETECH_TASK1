# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
file_path = r'CodeTech.csv'
data = pd.read_csv(file_path)


# Display first few rows to understand the data
print(data.head())

# Preprocessing the dataset

# Remove any rows with missing values (optional, you can also choose to fill missing values)
data = data.dropna()

# Handle categorical variables
# One-hot encode categorical columns like 'locality', 'furnishing', 'status', 'transaction', 'type'
data = pd.get_dummies(data, columns=['locality', 'furnishing', 'status', 'transaction', 'type'], drop_first=True)

# Define the features and target
X = data.drop('price', axis=1)  # Features (all columns except 'price')
y = data['price']  # Target variable (price)

# Feature scaling (optional, but useful for improving performance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Display coefficients of the features
print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")

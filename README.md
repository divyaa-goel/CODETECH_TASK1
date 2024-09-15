NAME : DIVYA MANISH GOEL
COMPANY : CODETECH IT SOLUTION
ID : CT8ML1842
DOMAIN: MACHINE LEARNING
DURATION: JULY 15TH TO SEPTEMBER 15TH 2024
OVERVIEW OF THE PROJECT: 
Objective : The objective of this project is to develop a machine learning model that predicts housing prices based on various features of properties. The dataset contains information about different property attributes, and the goal is to build a regression model that can accurately estimate the price of a property given its features.
Dataset : 
File: CodeTech.csv
Number of Rows: 1260
Columns -- 
Area: Size of the property in square feet.
BHK: Number of bedrooms, hall, and kitchen.
Bathroom: Number of bathrooms.
Furnishing: Type of furnishing (e.g., Semi-Furnished, Furnished).
Locality: Neighborhood or area where the property is located.
Parking: Number of parking spaces available.
Price: Price of the property (target variable).
Status: Availability status (e.g., Ready_to_move).
Transaction: Whether the property is new or resale.
Type: Type of property (e.g., Builder_Floor, Apartment).
Per_Sqft: Price per square foot (some entries may be missing).
Steps in the Project -
Data Loading : Load the dataset using pandas to read the CSV file.
Data Exploration : Examine the first few rows to understand the structure and content of the data.
Data Preprocessing : 
Handle Missing Values: Address any missing values in the dataset, particularly in the Per_Sqft column.
Categorical Encoding: Convert categorical variables (e.g., Furnishing, Status, Transaction, Type) into numerical format using one-hot encoding.
Feature Scaling: Standardize features to bring them onto a similar scale using StandardScaler.
Feature and Target Definition -
Features (X): All columns except Price.
Target (y): Price column.
Model Training and Evaluation - 
Split Data: Divide the dataset into training and testing sets using train_test_split.
Model Selection: Use Linear Regression from scikit-learn to build the model.
Train Model: Fit the model to the training data.
Predict and Evaluate: Predict prices on the test set and evaluate the model using metrics like Mean Squared Error (MSE) and R-squared.
Output Results - 
Display the performance metrics (MSE and R-squared) to assess the model's accuracy.
Show the coefficients of the features to understand their impact on the predicted price.
Deliverables - 
Code: Python script implementing data loading, preprocessing, model training, and evaluation.
Model Performance Metrics: MSE and R-squared values indicating how well the model predicts housing prices.
Feature Coefficients: Coefficients for each feature in the model, explaining their contribution to the price prediction.

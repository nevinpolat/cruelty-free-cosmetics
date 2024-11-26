#!/usr/bin/env python
# coding: utf-8

"""
train_model.py
Script to train multiple regression models on preprocessed and split data.
"""

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import pickle  # Importing pickle for serialization
import warnings

# Optional: Suppress warnings if desired
# warnings.filterwarnings("ignore", category=DataConversionWarning)

# Load feature datasets
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')

# Load target datasets and reshape to 1D arrays
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

# Verify shapes (optional)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


# After loading and preparing X_train
feature_names = X_train.columns.tolist()

# Ensure the 'models' directory exists
if not os.path.exists('models'):
    os.makedirs('models')

# Save feature names
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print("Feature names saved as 'models/feature_names.pkl'")

# ## Step 5: Model Building and Evaluation
# ### 5.1 Linear Regression

# Initialize the model
lr_model = LinearRegression()

# Train the model
lr_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_lr = lr_model.predict(X_test)

# Calculate evaluation metrics
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Print the results
print("Linear Regression Performance:")
print(f"Mean Squared Error (MSE): {mse_lr:.2f}")
print(f"Mean Absolute Error (MAE): {mae_lr:.2f}")
print(f"R-squared (R²): {r2_lr:.2f}")

# ### Save Linear Regression Model
with open('models/linear_regression_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
print("Linear Regression model saved as 'models/linear_regression_model.pkl'")

# ### 5.2 Decision Tree Regressor

# Initialize the model
dt_model = DecisionTreeRegressor(random_state=42)

# Train the model
dt_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_dt = dt_model.predict(X_test)

# Calculate evaluation metrics
mse_dt = mean_squared_error(y_test, y_pred_dt)
mae_dt = mean_absolute_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

# Print the results
print("\nDecision Tree Regressor Performance:")
print(f"Mean Squared Error (MSE): {mse_dt:.2f}")
print(f"Mean Absolute Error (MAE): {mae_dt:.2f}")
print(f"R-squared (R²): {r2_dt:.2f}")

# ### Save Decision Tree Regressor Model
with open('models/decision_tree_model.pkl', 'wb') as f:
    pickle.dump(dt_model, f)
print("Decision Tree Regressor model saved as 'models/decision_tree_model.pkl'")

# ### 5.3 RandomForestRegressor

# Initialize the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)

# Calculate evaluation metrics
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Print the results
print("\nRandom Forest Regressor Performance:")
print(f"Mean Squared Error (MSE): {mse_rf:.2f}")
print(f"Mean Absolute Error (MAE): {mae_rf:.2f}")
print(f"R-squared (R²): {r2_rf:.2f}")

# ### Save Random Forest Regressor Model
with open('models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("Random Forest Regressor model saved as 'models/random_forest_model.pkl'")

# ## 5.4 XGBRegressor

# Initialize the model
xgb_model = XGBRegressor(n_estimators=100, random_state=42)

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_xgb = xgb_model.predict(X_test)

# Calculate evaluation metrics
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

# Print the results
print("\nXGBoost Regressor Performance:")
print(f"Mean Squared Error (MSE): {mse_xgb:.2f}")
print(f"Mean Absolute Error (MAE): {mae_xgb:.2f}")
print(f"R-squared (R²): {r2_xgb:.2f}")

# ### Save XGBoost Regressor Model
with open('models/xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
print("XGBoost Regressor model saved as 'models/xgboost_model.pkl'")

# Create a DataFrame to compare models
performance_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'XGBoost'],
    'MSE': [mse_lr, mse_dt, mse_rf, mse_xgb],
    'MAE': [mae_lr, mae_dt, mae_rf, mae_xgb],
    'R_squared': [r2_lr, r2_dt, r2_rf, r2_xgb]
})

# Round the metrics for readability
performance_df['MSE'] = performance_df['MSE'].round(2)
performance_df['MAE'] = performance_df['MAE'].round(2)
performance_df['R_squared'] = performance_df['R_squared'].round(2)

print("\nModel Performance Comparison:")
print(performance_df)

# ### 5.5 Tune RandomForestRegressor
# ### GridSearchCV 

# Define the parameter grid
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the Random Forest model
rf = RandomForestRegressor(random_state=42)

# Initialize GridSearchCV
grid_search_rf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid_rf,
    cv=5,
    n_jobs=-1,
    scoring='neg_mean_squared_error'
)

# Fit the grid search to the data
grid_search_rf.fit(X_train, y_train)

print("\nBest Hyperparameters for Random Forest:")
print(grid_search_rf.best_params_)

# Use the best estimator
best_rf_model = grid_search_rf.best_estimator_

# Make predictions
y_pred_best_rf = best_rf_model.predict(X_test)

# Evaluate the model
mse_best_rf = mean_squared_error(y_test, y_pred_best_rf)
mae_best_rf = mean_absolute_error(y_test, y_pred_best_rf)
r2_best_rf = r2_score(y_test, y_pred_best_rf)

# Print the results
print("\nTuned Random Forest Regressor Performance:")
print(f"Mean Squared Error (MSE): {mse_best_rf:.2f}")
print(f"Mean Absolute Error (MAE): {mae_best_rf:.2f}")
print(f"R-squared (R²): {r2_best_rf:.2f}")

# Create a new DataFrame for the new row
new_row_rf = pd.DataFrame({
    'Model': ['Tuned Random Forest'],
    'MSE': [round(mse_best_rf, 2)],
    'MAE': [round(mae_best_rf, 2)],
    'R_squared': [round(r2_best_rf, 2)]
})

# Concatenate the new row to the existing DataFrame
performance_df = pd.concat([performance_df, new_row_rf], ignore_index=True)

# Display the updated DataFrame
print("\nUpdated Model Performance Comparison:")
print(performance_df)

# ### 5.6 Tune XGBoost Regressor
# ### GridSearchCV

# Define the parameter grid for tuning XGBoost
param_grid_xgb = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300],
    'reg_alpha': [0.01, 0.1, 1],
    'reg_lambda': [0.01, 0.1, 1],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9]
}

# Initialize GridSearchCV for XGBoost
grid_search_xgb = GridSearchCV(
    estimator=XGBRegressor(random_state=42),
    param_grid=param_grid_xgb,
    cv=5,
    scoring='r2',
    verbose=1,
    n_jobs=-1
)

# Fit the grid search to the data
grid_search_xgb.fit(X_train, y_train)

# Get the best model and parameters
best_model_xgb = grid_search_xgb.best_estimator_

# Evaluate the best model
y_pred_best_xgb = best_model_xgb.predict(X_test)

mse_best_xgb = mean_squared_error(y_test, y_pred_best_xgb)
mae_best_xgb = mean_absolute_error(y_test, y_pred_best_xgb)
r2_best_xgb = r2_score(y_test, y_pred_best_xgb)

print("\nBest XGBoost Model Parameters:")
print(grid_search_xgb.best_params_)

print("\nBest XGBoost Model Performance:")
print(f"MSE: {mse_best_xgb:.2f}")
print(f"MAE: {mae_best_xgb:.2f}")
print(f"R²: {r2_best_xgb:.2f}")

# ### Save Tuned XGBoost Model
with open('models/tuned_xgboost_model.pkl', 'wb') as f:
    pickle.dump(best_model_xgb, f)
print("Tuned XGBoost model saved as 'models/tuned_xgboost_model.pkl'")

# ### Add Tuned XGBoost to Performance DataFrame
new_row_xgb = pd.DataFrame({
    'Model': ['Tuned XGBoost'],
    'MSE': [round(mse_best_xgb, 2)],
    'MAE': [round(mae_best_xgb, 2)],
    'R_squared': [round(r2_best_xgb, 2)]
})

performance_df = pd.concat([performance_df, new_row_xgb], ignore_index=True)

print("\nFinal Model Performance Comparison:")
print(performance_df)

# ### Save Tuned Random Forest Model
with open('models/tuned_random_forest_model.pkl', 'wb') as f:
    pickle.dump(best_rf_model, f)
print("Tuned Random Forest model saved as 'models/tuned_random_forest_model.pkl'")

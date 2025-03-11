import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset (if not already loaded)
file_path = "Dataset.csv"
df = pd.read_csv(file_path)

# Encode categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns

def encode_features(df):
    df = df.copy()
    label_enc = LabelEncoder()
    for col in categorical_cols:
        df[col] = label_enc.fit_transform(df[col])
    return df

df_encoded = encode_features(df)

# Define features and target variable
X = df_encoded.drop(columns=['Annual_Premium_Amount'])
y = df_encoded['Annual_Premium_Amount']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning for XGBoost
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}

grid_search = GridSearchCV(XGBRegressor(objective='reg:squarederror', random_state=42), 
                            param_grid, cv=3, scoring='r2', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_xgb = grid_search.best_estimator_

# Save the best model and scaler
joblib.dump(best_xgb, "best_xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Evaluate the best model
y_pred_best = best_xgb.predict(X_test)
best_mse = mean_squared_error(y_test, y_pred_best)
best_r2 = r2_score(y_test, y_pred_best)

print("\nBest XGBoost Model Parameters:", grid_search.best_params_)
print(f"Best XGBoost Model: MSE={best_mse:.2f}, R2 Score={best_r2:.4f}")

# If you are going to use the model in the future with new input data, ensure to match the columns during transformation:

def transform_new_input(input_data, model_path="best_xgb_model.pkl", scaler_path="scaler.pkl"):
    # Load the trained model and scaler
    best_xgb = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Ensure the input data has the same columns as the training data
    input_data = input_data[train_data.columns]  # Reorder columns to match the training set

    # Check for missing columns and add them with default values if necessary
    missing_columns = [col for col in train_data.columns if col not in input_data.columns]
    for col in missing_columns:
        input_data[col] = 0  # Add missing columns with default value, e.g., 0

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict using the best model
    predictions = best_xgb.predict(input_data_scaled)
    return predictions

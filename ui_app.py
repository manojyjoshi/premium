import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Load the trained model and scaler
def load_model():
    return joblib.load("best_xgb_model.pkl")


def load_scaler():
    return joblib.load("scaler.pkl")


# Encode categorical input data
def encode_input_data(input_data):
    categorical_cols = ['Gender', 'BMI_Category', 'Smoking_Status', 'Physical_Activity', 'Stress_Level',
                        'Region', 'Marital_status', 'Employment_Status', 'Medical History', 'Insurance_Plan']
    label_enc = LabelEncoder()
    for col in categorical_cols:
        input_data[col] = label_enc.fit_transform(input_data[col])
    return input_data


# Transform input data to match model's expected features
def transform_input_data(input_data):
    # Load the training dataset to get the correct column order (or save the column order separately)
    train_data = pd.read_csv("Dataset.csv")  # You could save the training data column names separately for consistency
    # Drop the target column from the training data and get the column names (excluding 'Annual_Premium_Amount')
    feature_columns = [col for col in train_data.columns if col != 'Annual_Premium_Amount']

    # Add missing columns with default values (e.g., 'Unknown' for categorical or 0 for numerical columns)
    missing_columns = [col for col in feature_columns if col not in input_data.columns]
    for col in missing_columns:
        if col in ['Region', 'Marital_status', 'Medical History', 'Insurance_Plan', 'Employment_Status']:
            input_data[col] = 'Unknown'  # Categorical columns can be 'Unknown'
        else:
            input_data[col] = 0  # Numerical columns can be 0

    # Ensure the input data has the same structure as the training data
    input_data = input_data[feature_columns]

    return input_data


def main():
    st.title("Insurance Premium Prediction")
    st.write("Enter details to predict annual premium amount")

    # User inputs
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi_category = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"])
    smoking_status = st.selectbox("Smoking Status", ["Non-smoker", "Smoker"])
    physical_activity = st.selectbox("Physical Activity", ["Low", "Medium", "High"])
    stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"])
    income_lakhs = st.number_input("Income (in Lakhs)", min_value=1, max_value=100, value=10)

    # Add default values for missing columns
    region = 'Unknown'  # Default value for Region
    marital_status = 'Unknown'  # Default value for Marital_status
    employment_status = 'Unknown'  # Default value for Employment_Status
    medical_history = 'Unknown'  # Default value for Medical History
    insurance_plan = 'Unknown'  # Default value for Insurance_Plan
    number_of_dependants = 0  # Default value for Number Of Dependants
    income_level = 0  # Default value for Income_Level

    if st.button("Predict Premium"):
        model = load_model()
        scaler = load_scaler()

        # Create input data DataFrame from the user input
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'BMI_Category': [bmi_category],
            'Smoking_Status': [smoking_status],
            'Physical_Activity': [physical_activity],
            'Stress_Level': [stress_level],
            'Income_Lakhs': [income_lakhs],
            'Region': [region],
            'Marital_status': [marital_status],
            'Number Of Dependants': [number_of_dependants],
            'Employment_Status': [employment_status],
            'Income_Level': [income_level],
            'Medical History': [medical_history],
            'Insurance_Plan': [insurance_plan]
        })

        # Encode categorical input data
        input_data = encode_input_data(input_data)

        # Ensure the input data has the same structure as the training data
        input_data = transform_input_data(input_data)

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Predict using the model
        prediction = model.predict(input_data_scaled)

        st.success(f"Predicted Annual Premium: ₹{prediction[0]:,.2f}")


if __name__ == "__main__":
    main()

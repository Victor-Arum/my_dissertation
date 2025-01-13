import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import shap
import json
import plotly.graph_objects as go

# Clear cache function
if st.button("Clear Cache"):
    st.cache_data.clear()  # Clears cached data
    st.cache_resource.clear()  # Clears cached resources
    st.write("Cache cleared!")

# Load the trained XGBoost model
xgb_model = joblib.load('xgboost_model.pkl')

# Load feature mappings (if applicable)
with open('feature_mappings.json', 'r') as f:
    feature_mappings = json.load(f)

# Define features used in prediction
new_features = [
    "admission_source_id", "discharge_disposition_id",
    "diabetesMed", "metformin",
    "number_inpatient", "number_diagnoses",
    "diag_1"
]

# Preprocessing function to ensure inputs match the training process
def preprocess_input(data):
    """
    Preprocess user input to match the trained model's expectations.
    - Handle missing values
    - Encode categorical features
    - Apply OneHotEncoder
    - Ensure correct feature ordering
    """
    # Handle missing values for diagnosis codes
    data["diag_1"].fillna(0, inplace=True)  # Replace missing ICD-9 codes with '0'

    # Ordinal encoding for categorical features
    ordinal_encoder = OrdinalEncoder()
    categorical_features = [
        "admission_source_id", "discharge_disposition_id",
        "diabetesMed", "metformin"
    ]
    data[categorical_features] = ordinal_encoder.fit_transform(data[categorical_features])

    # One-hot encoding for selected features
    oh_encoder = OneHotEncoder(handle_unknown="ignore")
    encoded_data = oh_encoder.fit_transform(data[new_features]).toarray()
    encoded_df = pd.DataFrame(encoded_data, columns=oh_encoder.get_feature_names_out(new_features))

    # Combine numerical and one-hot encoded categorical features
    numerical_features = ["number_inpatient", "number_diagnoses"]
    numerical_data = data[numerical_features]

    processed_data = pd.concat([numerical_data, encoded_df], axis=1)

    # Ensure feature order matches the model's training data
    processed_data = processed_data.reindex(columns=xgb_model.feature_names_in_, fill_value=0)
    return processed_data, oh_encoder, ordinal_encoder

# Function to map SHAP features to user inputs
def map_shap_features(shap_feature_names, user_input, oh_encoder, ordinal_encoder, feature_mappings):
    """
    Map SHAP feature names back to user inputs or human-readable names using feature mappings.
    """
    mapped_features = []
    for feature in shap_feature_names:
        # Check if feature is from one-hot encoding
        for original_feature in new_features:
            if feature.startswith(original_feature):
                # Handle one-hot encoded features
                if original_feature in feature_mappings:
                    key = feature.split("_")[-1]  # Extract the key for mapping
                    mapped_features.append(f"{feature_mappings[original_feature].get(key, feature)}")
                    break
                # Handle raw features
                elif original_feature in user_input:
                    mapped_features.append(f"{original_feature}: {user_input[original_feature].iloc[0]}")
                    break
        else:
            mapped_features.append(feature)  # Default to the raw feature name if no mapping found
    return mapped_features

# Streamlit App Interface
st.title("Patient Readmission Prediction System")
st.write("Predict whether a patient will be readmitted within 30 days based on clinical and demographic data.")

# Collect user inputs
admission_source = st.selectbox(
    "Admission Source",
    options=list(feature_mappings['admission_source_id'].values())
)
discharge_disposition = st.selectbox(
    "Discharge Disposition",
    options=list(feature_mappings['discharge_disposition_id'].values())
)
diabetesMed = st.selectbox(
    "Diabetes Medication",
    options=list(feature_mappings['diabetesMed'].values())
)
metformin = st.selectbox(
    "Metformin Status",
    options=list(feature_mappings['metformin'].values())
)
number_inpatient = st.number_input(
    "Number of Prior Inpatient Visits", min_value=0, max_value=20, step=1
)
number_diagnoses = st.number_input(
    "Number of Diagnoses", min_value=1, max_value=20, step=1
)
diag_1 = st.text_input("Primary Diagnosis (ICD-9 Code, e.g., 276.0)")

# Map user inputs to DataFrame
user_input = pd.DataFrame({
    "admission_source_id": [admission_source],
    "discharge_disposition_id": [discharge_disposition],
    "diabetesMed": [diabetesMed],
    "metformin": [metformin],
    "number_inpatient": [number_inpatient],
    "number_diagnoses": [number_diagnoses],
    "diag_1": [float(diag_1) if diag_1 else 0]
})

# Preprocess the input data
processed_input, oh_encoder, ordinal_encoder = preprocess_input(user_input)

# Debugging: Display processed input
st.write("Processed Input Data:")
st.write(processed_input)

# Predict and display results
if st.button("Predict"):
    prediction = xgb_model.predict(processed_input)[0]
    prediction_proba = xgb_model.predict_proba(processed_input)[0]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.write(f"🌟 **This patient will be readmitted within 30 days.**")
        st.write(f"Confidence: {prediction_proba[prediction] * 100:.2f}%")
    else:
        st.write(f"✅ **This patient is not likely to be readmitted within 30 days.**")
        st.write(f"Confidence: {prediction_proba[prediction] * 100:.2f}%")

    # SHAP Bar Chart for Individual Prediction
    st.subheader("Feature Contributions (Bar Chart)")
    st.write("""
    This bar chart shows how each feature influenced the model's prediction for this specific patient.
    - **Red bars**: Increase the likelihood of readmission.
    - **Blue bars**: Decrease the likelihood of readmission.
    """)

    # Create a SHAP explainer and calculate SHAP values for the processed input
    explainer = shap.Explainer(xgb_model)
    shap_values = explainer(processed_input)

    

    # Extract SHAP values for the single input
    shap_values_single = shap_values[0]


    # Map SHAP feature names to user inputs
    shap_values_array = shap_values_single.values
    feature_names = shap_values_single.feature_names
    mapped_features = map_shap_features(
        feature_names, user_input, oh_encoder, ordinal_encoder, feature_mappings
    )

    # Combine SHAP values and mapped features into a DataFrame for sorting
    shap_df = pd.DataFrame({
        "Feature": mapped_features,
        "SHAP Value": shap_values_array
    })

    # Sort features by absolute SHAP value
    shap_df["Abs SHAP Value"] = shap_df["SHAP Value"].abs()
    shap_df = shap_df.sort_values(by="Abs SHAP Value", ascending=False).head(10)  # Show top 10 features

    # Split into positive and negative contributions
    positive_contributions = shap_df[shap_df["SHAP Value"] > 0].sort_values(by="SHAP Value", ascending=True)
    negative_contributions = shap_df[shap_df["SHAP Value"] < 0].sort_values(by="SHAP Value", ascending=False)

    # Create the enhanced Plotly bar chart
    fig = go.Figure()

    # Add negative contributions (blue bars)
    fig.add_trace(go.Bar(
        x=negative_contributions["SHAP Value"],
        y=negative_contributions["Feature"],
        orientation="h",
        marker=dict(color="blue", opacity=0.8),
        name="Decrease Readmission Likelihood"
    ))

    # Add positive contributions (red bars)
    fig.add_trace(go.Bar(
        x=positive_contributions["SHAP Value"],
        y=positive_contributions["Feature"],
        orientation="h",
        marker=dict(color="red", opacity=0.8),
        name="Increase Readmission Likelihood"
    ))

    # Update layout to manage space better
    fig.update_layout(
        title="Top 10 SHAP Feature Contributions",
        xaxis=dict(
            title="SHAP Value (Impact on Prediction)",
            title_font=dict(size=14),
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title="Features",
            title_font=dict(size=14),
            tickfont=dict(size=12),
            automargin=True,  # Ensure no labels are clipped
            categoryorder="total ascending"  # Sort features by SHAP value
        ),
        height=700,  # Increased height for better spacing
        width=1200,  # Increased width for x-axis clarity
        template="plotly_white",
        legend=dict(
            title="Legend",
            orientation="h",  # Horizontal legend
            yanchor="bottom",
            y=-0.2,  # Position below the chart
            xanchor="center",
            x=0.5
        )
    )

    # Display the chart
    st.plotly_chart(fig)

# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model_artifacts = joblib.load('hospital_readmission_model.pkl')
        return model_artifacts
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocess input data
def preprocess_input(input_data, model_artifacts):
    """Preprocess the input data similar to training"""
    df = input_data.copy()
    
    # Convert date
    df['Admission_Date'] = pd.to_datetime(df['Admission_Date'])
    df['Admission_Month'] = df['Admission_Date'].dt.month
    df['Admission_Day'] = df['Admission_Date'].dt.day
    df['Admission_DayOfWeek'] = df['Admission_Date'].dt.dayofweek
    
    # Encode categorical variables
    label_encoders = model_artifacts['label_encoders']
    feature_columns = model_artifacts['feature_columns']
    
    for col in ['Age_Group', 'Gender', 'Condition_Type', 'Department', 'Insurance_Type']:
        le = label_encoders[col]
        # Handle unseen labels by using the most common class
        df[col + '_encoded'] = df[col].apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else le.transform([le.classes_[0]])[0]
        )
    
    # Select features
    X = df[feature_columns]
    
    return X

# Make prediction
def make_prediction(input_data, model_artifacts):
    """Make prediction using the loaded model"""
    model = model_artifacts['model']
    scaler = model_artifacts['scaler']
    model_name = model_artifacts['best_model_name']
    
    # Preprocess input
    X_processed = preprocess_input(input_data, model_artifacts)
    
    # Scale if needed
    if model_name in ['Logistic Regression', 'SVM']:
        X_scaled = scaler.transform(X_processed)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
    else:
        predictions = model.predict(X_processed)
        probabilities = model.predict_proba(X_processed)
    
    return predictions, probabilities

# Main app
def main():
    # Header
    st.title("🏥 Hospital Readmission Prediction System")
    st.markdown("""
    This AI-powered tool predicts the likelihood of patient readmission within 30 days 
    based on clinical and demographic factors.
    """)
    
    # Load model
    model_artifacts = load_model()
    if model_artifacts is None:
        st.error("Please ensure 'hospital_readmission_model.pkl' is in the same directory")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Mode",
        ["Single Patient Prediction", "Batch Prediction", "Model Information"]
    )
    
    if app_mode == "Single Patient Prediction":
        single_patient_prediction(model_artifacts)
    elif app_mode == "Batch Prediction":
        batch_prediction(model_artifacts)
    else:
        model_information(model_artifacts)

def single_patient_prediction(model_artifacts):
    """Single patient prediction interface"""
    st.header("Single Patient Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographic Information")
        age = st.slider("Age", 0, 100, 50)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age_group = st.selectbox("Age Group", ["0-12", "13-24", "25-40", "41-60", "61+"])
        insurance_type = st.selectbox("Insurance Type", ["Private", "Medicare", "Medicaid", "Uninsured"])
    
    with col2:
        st.subheader("Clinical Information")
        condition_type = st.selectbox("Condition Type", [
            "Heart Failure", "Asthma", "COPD", "Stroke", "Diabetes", 
            "Infection", "Appendicitis", "Fracture", "Alzheimer's", "Pregnancy"
        ])
        department = st.selectbox("Department", [
            "ICU", "General Medicine", "Surgery", "Pediatrics", "Orthopedics",
            "Pulmonology", "Endocrinology", "Geriatrics", "Obstetrics"
        ])
        severity_score = st.slider("Severity Score (1-10)", 1, 10, 5)
        length_of_stay = st.slider("Length of Stay (days)", 1, 30, 7)
        admission_date = st.date_input("Admission Date", datetime.now())

        if st.button("Predict Readmission Risk", type="primary"):
            # Create input dataframe
            input_data = pd.DataFrame({
                'Age': [age],
                'Gender': [gender],
                'Age_Group': [age_group],
                'Condition_Type': [condition_type],
                'Department': [department],
                'Insurance_Type': [insurance_type],
                'Severity_Score': [severity_score],
                'Length_of_Stay': [length_of_stay],
                'Admission_Date': [admission_date.strftime('%Y-%m-%d')]
            })
    
    with col3:
        # Display prediction when button is clicked
        
            
            # Make prediction
            predictions, probabilities = make_prediction(input_data, model_artifacts)
            
            # Display results
            display_prediction_results(input_data, predictions[0], probabilities[0])

def display_prediction_results(input_data, prediction, probability):
    """Display prediction results in an attractive way"""
    st.header("Prediction Results")
    
    risk_level = "HIGH" if prediction == 1 else "LOW"
    risk_color = "red" if prediction == 1 else "green"
    risk_probability = probability[1] if prediction == 1 else probability[0]
    
    # Risk indicator
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; border-radius: 10px; 
                    background-color: {'#ffcccc' if prediction == 1 else '#ccffcc'}; 
                    border: 2px solid {risk_color}'>
            <h2 style='color: {risk_color}; margin: 0;'>RISK LEVEL: {risk_level}</h2>
            <h3 style='color: {risk_color}; margin: 0;'>Confidence: {risk_probability:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Patient summary
    st.subheader("Patient Summary")
    summary_cols = st.columns(4)
    with summary_cols[0]:
        st.metric("Age", input_data['Age'].iloc[0])
    with summary_cols[1]:
        st.metric("Condition", input_data['Condition_Type'].iloc[0])
    with summary_cols[2]:
        st.metric("Severity", input_data['Severity_Score'].iloc[0])
    with summary_cols[3]:
        st.metric("Stay Duration", f"{input_data['Length_of_Stay'].iloc[0]} days")
    
    # Probability gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability[1] * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Readmission Probability (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.subheader("Clinical Recommendations")
    if prediction == 1:
        st.warning("""
        **High Readmission Risk Detected - Recommended Actions:**
        - Schedule follow-up appointment within 7 days
        - Provide comprehensive discharge instructions
        - Coordinate with home health services
        - Ensure medication reconciliation is completed
        - Assign case manager for post-discharge follow-up
        """)
    else:
        st.success("""
        **Low Readmission Risk - Standard Care:**
        - Provide standard discharge instructions
        - Schedule routine follow-up as needed
        - Ensure patient understands medication regimen
        - Provide emergency contact information
        """)

def batch_prediction(model_artifacts):
    """Batch prediction interface"""
    st.header("Batch Prediction")
    
    st.markdown("""
    Upload a CSV file with multiple patient records for batch prediction.
    The file should contain the following columns:
    - Age, Gender, Age_Group, Condition_Type, Department, Insurance_Type
    - Severity_Score, Length_of_Stay, Admission_Date
    """)
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            batch_data = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(batch_data)} patient records")
            
            # Show preview
            st.subheader("Data Preview")
            st.dataframe(batch_data.head())
            
            if st.button("Run Batch Prediction", type="primary"):
                with st.spinner("Processing predictions..."):
                    # Make predictions
                    predictions, probabilities = make_prediction(batch_data, model_artifacts)
                    
                    # Add predictions to dataframe
                    results_df = batch_data.copy()
                    results_df['Predicted_Readmission'] = predictions
                    results_df['Readmission_Probability'] = probabilities[:, 1]
                    results_df['Risk_Level'] = results_df['Predicted_Readmission'].map({0: 'Low', 1: 'High'})
                    
                    # Display results
                    st.subheader("Prediction Results")
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    high_risk_count = (results_df['Predicted_Readmission'] == 1).sum()
                    
                    with col1:
                        st.metric("Total Patients", len(results_df))
                    with col2:
                        st.metric("High Risk Patients", high_risk_count)
                    with col3:
                        st.metric("High Risk Percentage", f"{(high_risk_count/len(results_df))*100:.1f}%")
                    with col4:
                        avg_prob = results_df['Readmission_Probability'].mean()
                        st.metric("Average Risk Probability", f"{avg_prob:.1%}")
                    
                    # Display results table
                    st.dataframe(results_df)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name="readmission_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Visualization
                    st.subheader("Risk Distribution")
                    fig = px.pie(results_df, names='Risk_Level', 
                                title='Distribution of Readmission Risk Levels')
                    st.plotly_chart(fig, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error processing file: {e}")

def model_information(model_artifacts):
    """Display model information"""
    st.header("Model Information")
    
    st.subheader("Model Details")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Model Type", model_artifacts['best_model_name'])
        st.metric("Number of Features", len(model_artifacts['feature_columns']))
    
    with col2:
        st.metric("Target Variable", "30-day Readmission")
        st.metric("Training Data", "500+ hospital admissions")
    
    st.subheader("Feature Importance")
    if hasattr(model_artifacts['model'], 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': model_artifacts['feature_columns'],
            'importance': model_artifacts['model'].feature_importances_
        }).sort_values('importance', ascending=True)
        
        fig = px.bar(feature_importance, x='importance', y='feature', 
                    orientation='h', title='Feature Importance')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance not available for this model type")
    
    st.subheader("Supported Features")
    st.markdown("""
    The model considers the following factors:
    - **Demographic**: Age, Gender, Insurance Type
    - **Clinical**: Condition Type, Severity Score, Department
    - **Temporal**: Length of Stay, Admission Date components
    """)
    
    st.subheader("Limitations")
    st.warning("""
    - This is a predictive model and should not replace clinical judgment
    - Model performance may vary across different patient populations
    - Regular model updates are recommended as new data becomes available
    - Always consult with healthcare professionals for medical decisions
    """)

if __name__ == "__main__":
    main()

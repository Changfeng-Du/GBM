import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# Load the new model
model = joblib.load('GBM.pkl')

# Load the test data from X_test.csv to create LIME explainer
X_test = pd.read_csv('X_test.csv')

# Define feature names from the new dataset
feature_names = [
    "Smoker", "Carace",  "Drink", "Hypertension", "HHR", "RIDAGEYR", "INDFMPIR", "BMXBMI","LBXWBCSI"]

# Streamlit user interface
st.title("Co-occurrence of Myocardial Infarction and Stroke Predictor")

# RIDAGEYR: numerical input
RIDAGEYR = st.number_input("Age:", min_value=20, max_value=80, value=40)

# Carace: categorical selection
Carace  = st.selectbox("Carace:", options=[1, 2,3,4,5], format_func=lambda x: "Mexcian American" if x == 2 else "other Hispanic" if x == 3 else "Non-Hispanic White" if x == 4 else "Non-Hispanic Black" if x == 5 else "Other Race")

# Smoker: categorical selection
Smoker = st.selectbox("Smoker:", options=[1, 2,3], format_func=lambda x: "never" if x == 2 else "ever" if x == 3 else "current")

# Drink: categorical selection
Drink = st.selectbox("Drink:", options=[1, 2], format_func=lambda x: "Yes" if x == 2 else "No")

# Hypertension: categorical selection
Hypertension = st.selectbox("Hypertension:", options=[1, 2], format_func=lambda x: "Yes" if x == 2 else "No")

# HHR: numerical input
HHR = st.number_input("HHR:", min_value=0.23, max_value=1.67, value=0.49)

# INDFMPIR: numerical input
INDFMPIR = st.number_input("INDFMPIR:", min_value=0.0, max_value=5.0, value=1.2)

# BMXBMI: categorical selection
BMXBMI = st.number_input("BMXBMI:", min_value=11.5, max_value=67.3, value=23.0)

# LBXWBCSI: categorical selection
LBXWBCSI = st.number_input("LBXWBCSI:", min_value=1.4, max_value=117.2, value=42.0)

# Process inputs and make predictions
feature_values = [ Smoker, Carace,  Drink, Hypertension, HHR, RIDAGEYR, INDFMPIR, BMXBMI,LBXWBCSI]
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class} (1: Disease, 0: No Disease)")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of co-occurrence of myocardial infarction and stroke disease. "
            f"The model predicts that your probability of having co-occurrence of myocardial infarction and stroke disease is {probability:.1f}%. "
            "It's advised to consult with your healthcare provider for further evaluation and possible intervention."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of myocardial infarction and stroke disease. "
            f"The model predicts that your probability of not having co-occurrence of myocardial infarction and stroke disease is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider."
        )

    st.write(advice)

    # SHAP Explanation
    st.subheader("SHAP Force Plot Explanation")
    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    
    # Display the SHAP force plot for the predicted class
    if predicted_class == 1:
        shap.force_plot(explainer_shap.expected_value, shap_values[:,:,1], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    else:
        shap.force_plot(1 -explainer_shap.expected_value, shap_values[:,:,0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)

    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')

    # LIME Explanation
    st.subheader("LIME Explanation")
    lime_explainer = LimeTabularExplainer(
        training_data=X_test.values,
        feature_names=X_test.columns.tolist(),
        class_names=['Not sick', 'Sick'],  # Adjust class names to match your classification task
        mode='classification'
    )
    
    # Explain the instance
    lime_exp = lime_explainer.explain_instance(
        data_row=features.flatten(),
        predict_fn=model.predict_proba
    )

    # Display the LIME explanation without the feature value table
    lime_html = lime_exp.as_html(show_table=False)  # Disable feature value table
    st.components.v1.html(lime_html, height=800, scrolling=True)
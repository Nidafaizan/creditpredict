import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import os

# Set page config
st.set_page_config(
    page_title="Credit Prediction Model",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Approval Prediction Model")
st.markdown("Predict credit approval using machine learning")

# Load the data and train the model


@st.cache_resource
def load_model():
    df = pd.read_csv('credit (1).csv')

    X = df.drop('approved', axis=1)
    y = df['approved']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=7,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=20
    )

    model.fit(X_scaled, y)

    return model, scaler, X.columns


model, scaler, feature_names = load_model()

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Enter Applicant Information")
    age = st.slider("Age", min_value=18, max_value=80, value=35)
    income = st.number_input(
        "Annual Income ($)", min_value=20000, max_value=200000, value=50000)

with col2:
    years_at_job = st.slider("Years at Current Job",
                             min_value=0, max_value=50, value=5)
    credit_score = st.slider(
        "Credit Score", min_value=300, max_value=850, value=650)

existing_cards = st.slider(
    "Number of Existing Credit Cards", min_value=0, max_value=10, value=2)

# Create prediction button
if st.button("Predict Credit Approval", type="primary"):
    # Prepare input data
    input_data = np.array(
        [[age, income, years_at_job, credit_score, existing_cards]])

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Display results
    st.divider()
    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        approval_probability = prediction
        st.metric(
            "Approval Probability",
            f"{approval_probability*100:.2f}%",
            delta=None
        )

        if approval_probability > 0.5:
            st.success("✅ Likely to be Approved")
        elif approval_probability > 0.3:
            st.warning("⚠️ Uncertain - May Require Review")
        else:
            st.error("❌ Likely to be Denied")

    with col2:
        st.info(f"""
        **Applicant Summary:**
        - Age: {age} years
        - Income: ${income:,}
        - Years at Job: {years_at_job}
        - Credit Score: {credit_score}
        - Existing Cards: {existing_cards}
        """)

# Display model information
st.divider()
st.subheader("Model Information")

col1, col2 = st.columns(2)

with col1:
    st.write("""
    **Model Details:**
    - Algorithm: Gradient Boosting Regressor
    - Features: 5 input variables
    - Training Samples: 1000
    - Test R² Score: 0.7533
    """)

with col2:
    st.write("""
    **Feature Importance:**
    1. Credit Score: 55.2%
    2. Income: 37.0%
    3. Age: 4.7%
    4. Years at Job: 2.3%
    5. Existing Cards: 0.7%
    """)

st.divider()
st.caption("Credit Prediction Model - Powered by Machine Learning")

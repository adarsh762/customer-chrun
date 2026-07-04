import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = joblib.load("bank_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align:center;'>🏦 Bank Customer Churn Prediction</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h4 style='text-align:center;color:gray;'>Predict customer churn using Machine Learning</h4>",
    unsafe_allow_html=True
)

st.caption("Machine Learning Powered Customer Churn Prediction System")

st.write("Enter the customer details below to predict the probability of churn.")

st.sidebar.header("📝 Customer Information")

credit_score = st.sidebar.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

tenure = st.sidebar.number_input(
    "Tenure",
    min_value=0,
    max_value=10,
    value=5
)

balance = st.sidebar.number_input(
    "Balance",
    min_value=0.0,
    value=50000.0
)

num_products = st.sidebar.selectbox(
    "Number of Products",
    [1, 2, 3, 4]
)

has_card = st.sidebar.selectbox(
    "Has Credit Card",
    [0, 1]
)

is_active = st.sidebar.selectbox(
    "Is Active Member",
    [0, 1]
)

salary = st.sidebar.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=60000.0
)

country = st.sidebar.selectbox(
    "Country",
    ["France", "Germany", "Spain"]
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male"]
)

# Predict Button
if st.sidebar.button("🔍 Predict Churn", use_container_width=True):

    # Create engineered features
    balance_salary_ratio = balance / (salary + 1)
    product_density = num_products / (age + 1)
    engagement_product = is_active * num_products
    age_tenure = age * tenure

    # One-hot encoding
    geo_germany = 1 if country == "Germany" else 0
    geo_spain = 1 if country == "Spain" else 0
    gender_male = 1 if gender == "Male" else 0

    # Create dataframe
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_products],
        "HasCrCard": [has_card],
        "IsActiveMember": [is_active],
        "EstimatedSalary": [salary],
        "BalanceSalaryRatio": [balance_salary_ratio],
        "ProductDensity": [product_density],
        "EngagementProduct": [engagement_product],
        "AgeTenure": [age_tenure],
        "Geography_Germany": [geo_germany],
        "Geography_Spain": [geo_spain],
        "Gender_Male": [gender_male]
    })

    # # Scale input
    # input_scaled = scaler.transform(input_data)

    st.markdown("---")
    st.subheader("📋 Customer Summary")

    summary = pd.DataFrame({
        "Feature": [
            "Credit Score",
            "Age",
            "Country",
            "Products",
            "Active Member"
        ],
        "Value": [
            credit_score,
            age,
            country,
            num_products,
            "Yes" if is_active else "No"
        ]
    })

    st.table(summary)

    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    st.subheader("Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        if prediction == 1:
            st.error("⚠️ Churn")
        else:
            st.success("✅ Stay")

    with col2:
        st.metric(
            "Probability",
            f"{probability*100:.2f}%"
        )
    st.progress(float(probability))

    with col3:
        if probability < 0.30:
            st.success("🟢 LOW")

        elif probability < 0.70:
            st.warning("🟡 MEDIUM")

        else:
            st.error("🔴 HIGH")

#recomendations

    if probability < 0.30:
        st.info(
            "💡 Recommendation: Customer is at low risk. Continue regular engagement and monitor periodically."
        )

    elif probability < 0.70:
        st.info(
            "💡 Recommendation: Customer shows moderate churn risk. Consider offering personalized discounts or loyalty benefits."
        )

    else:
        st.info(
            "💡 Recommendation: High churn risk detected. Immediate retention actions such as relationship manager outreach or special offers are recommended."
        )

    
    st.markdown("---")
    st.subheader("📈 Top 10 Feature Importance")

    feature_names = input_data.columns
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8,5))

    ax.barh(
        importance_df["Feature"][:10],
        importance_df["Importance"][:10]
    )

    ax.invert_yaxis()

    st.pyplot(fig)

    st.subheader("📌 Top Churn Drivers")

    top5 = importance_df.head(5)

    st.table(top5)

    st.markdown("---")



st.caption(
    "Built using Streamlit | Scikit-Learn | Random Forest Classifier"
)

st.markdown("---")
st.subheader("ℹ️ About This Project")

st.write("""
This application predicts whether a bank customer is likely to churn using a Random Forest Machine Learning model.

It analyzes customer information such as:

Credit Score
Age
Account Balance
Number of Products
Customer Activity
Salary
Geography

The model outputs:

Churn Prediction
Churn Probability
Risk Level
Customer Recommendation
""")

st.markdown("---")

st.caption(
"""
Built by Vishesh

Machine Learning | Streamlit | Scikit-Learn | Random Forest
"""
)
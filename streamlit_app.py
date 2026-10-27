
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)

MODEL_FILE = Path("best_churn_model.pkl")
DATA_FILE = Path("telco_churn_cleaned.csv")
RESULT_FILE = Path("model_comparison.csv")

st.title("📊 AI-Powered Customer Churn Prediction")
st.caption("Customer Analytics and Machine Learning Dashboard")

# Check required files
if not MODEL_FILE.exists() or not DATA_FILE.exists():
    st.error(
        "Required files not found. First run app.py "
        "to generate the trained model and cleaned dataset."
    )
    st.stop()

# Load saved model and dataset
model = joblib.load(MODEL_FILE)
df = pd.read_csv(DATA_FILE)

# Dashboard metrics
total_customers = len(df)
churned = int((df["Churn"] == "Yes").sum())
retained = int((df["Churn"] == "No").sum())
churn_rate = churned / total_customers * 100 if total_customers else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers", f"{total_customers:,}")
c2.metric("Churned Customers", f"{churned:,}")
c3.metric("Retained Customers", f"{retained:,}")
c4.metric("Churn Rate", f"{churn_rate:.2f}%")

st.divider()

# Analytics charts
st.subheader("📈 Customer Analytics")

left, right = st.columns(2)

with left:
    st.write("**Customer Churn Distribution**")
    churn_counts = df["Churn"].value_counts()
    st.bar_chart(churn_counts)

with right:
    st.write("**Contract Type vs Churn**")
    contract_data = pd.crosstab(
        df["Contract"], df["Churn"]
    )
    st.bar_chart(contract_data)

st.write("**Customer Tenure vs Churn**")
tenure_data = df.groupby("Churn")["tenure"].mean()
st.bar_chart(tenure_data)

# Prediction form
st.divider()
st.subheader("🤖 Predict Customer Churn")
st.write("Enter customer details to estimate churn risk.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender", ["Male", "Female"]
        )
        senior = st.selectbox(
            "Senior Citizen", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        partner = st.selectbox(
            "Has Partner?", ["Yes", "No"]
        )
        dependents = st.selectbox(
            "Has Dependents?", ["Yes", "No"]
        )
        tenure = st.slider(
            "Tenure (months)", 0, 72, 12
        )
        phone = st.selectbox(
            "Phone Service", ["Yes", "No"]
        )
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )
        internet = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

    with col2:
        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly = st.number_input(
            "Monthly Charges",
            min_value=0.0, max_value=200.0, value=70.0
        )
        total = st.number_input(
            "Total Charges",
            min_value=0.0, max_value=10000.0, value=840.0
        )

    submitted = st.form_submit_button(
        "Predict Churn", type="primary"
    )

if submitted:
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    # Ensure the input matches the trained model's feature order
    try:
        customer = customer[X_columns] if False else customer
        prediction = int(model.predict(customer)[0])

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error(
                "⚠️ The model predicts that this customer "
                "may churn."
            )
        else:
            st.success(
                "✅ The model predicts that this customer "
                "may stay."
            )

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(customer)[0]
            churn_probability = float(probabilities[1]) * 100
            st.metric(
                "Estimated Churn Probability",
                f"{churn_probability:.2f}%"
            )
            st.progress(
                min(max(churn_probability / 100, 0.0), 1.0)
            )

        st.caption(
            "This is a model estimate, not a guarantee "
            "of customer behavior."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")

# Model comparison
if RESULT_FILE.exists():
    st.divider()
    st.subheader("🏆 Model Comparison")
    results = pd.read_csv(RESULT_FILE)
    st.dataframe(results, use_container_width=True)

# Dataset preview
with st.expander("View Dataset"):
    st.dataframe(df.head(20), use_container_width=True)

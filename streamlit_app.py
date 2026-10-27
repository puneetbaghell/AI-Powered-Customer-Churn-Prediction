
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="ChurnIQ | ML Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "telco_churn_cleaned.csv"
MODEL_PATH = BASE / "best_churn_model.pkl"

# ==================================================
# CUSTOM DESIGN
# ==================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: radial-gradient(ellipse at top, #101d42 0%, #070d1d 52%, #050a16 100%);
    color: #edf4ff;
}
.block-container {
    max-width: 1550px;
    padding: 1.7rem 2rem 3rem;
}
[data-testid="stSidebar"] {
    background: #080f20;
    border-right: 1px solid #1d2b49;
}
[data-testid="stSidebar"] * {
    color: #dbeafe;
}
.hero {
    background: linear-gradient(115deg, #101f46, #172554 52%, #153b52);
    border: 1px solid #254575;
    padding: 27px 30px;
    border-radius: 20px;
    margin-bottom: 22px;
    box-shadow: 0 12px 40px #00000030;
}
.eyebrow {
    color: #67e8f9;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 2px;
    font-weight: 800;
}
.hero h1 {
    color: white;
    font-size: 31px;
    margin: 8px 0;
    font-weight: 800;
}
.hero p {
    color: #bdcce7;
    margin: 0;
    font-size: 14px;
}
.section-title {
    font-size: 21px;
    color: #f0f6ff;
    font-weight: 750;
    margin: 18px 0 3px;
}
.section-sub {
    color: #8ea4c8;
    font-size: 12px;
    margin-bottom: 14px;
}
.kpi {
    background: linear-gradient(145deg, #111e36, #0b1428);
    border: 1px solid #243858;
    border-radius: 16px;
    padding: 18px 19px;
    min-height: 125px;
}
.kpi-label {
    font-size: 12px;
    color: #9fb2d1;
}
.kpi-value {
    color: #f8fbff;
    font-size: 29px;
    font-weight: 800;
    margin-top: 9px;
}
.kpi-note {
    color: #70e1c0;
    font-size: 11px;
    margin-top: 4px;
}
.panel {
    background: linear-gradient(145deg, #0d1930, #0a1326);
    border: 1px solid #1f3455;
    border-radius: 17px;
    padding: 17px 18px;
}
div[data-testid="stMetric"] {
    background: #101d34;
    border: 1px solid #263b5d;
    border-radius: 13px;
    padding: 15px;
}
div[data-testid="stMetricLabel"] {
    color: #9fb2d1;
}
div[data-testid="stMetricValue"] {
    color: #f8fbff;
}
.stButton > button, .stFormSubmitButton > button {
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    border: 1px solid #4b72dc;
    border-radius: 10px;
    color: white;
    font-weight: 700;
    min-height: 42px;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    border-color: #67e8f9;
    color: white;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    background: #101d34;
    border-radius: 9px;
    color: #cbdaf3;
}
div[data-testid="stForm"] {
    background: #0c172b;
    border: 1px solid #253b5d;
    border-radius: 16px;
    padding: 20px;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #243858;
    border-radius: 12px;
}
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD DATA AND MODEL
# ==================================================
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

if not DATA_PATH.exists():
    st.error("Dataset not found: telco_churn_cleaned.csv")
    st.stop()

if not MODEL_PATH.exists():
    st.error("Trained model not found: best_churn_model.pkl")
    st.stop()

try:
    df = load_data(DATA_PATH)
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error(f"Unable to load project files: {exc}")
    st.stop()

# Detect target column
target_col = next(
    (col for col in df.columns if col.lower() == "churn"),
    None
)

if target_col is None:
    st.error("A 'Churn' target column was not found in the dataset.")
    st.stop()

def normalize_churn(series):
    return series.astype(str).str.strip().str.lower().isin(
        ["yes", "1", "true", "churned"]
    )

churn_mask = normalize_churn(df[target_col])
total_customers = len(df)
churned_customers = int(churn_mask.sum())
retained_customers = total_customers - churned_customers
churn_rate = churned_customers / total_customers * 100 if total_customers else 0

def find_col(*terms):
    for col in df.columns:
        name = col.lower().replace("_", " ")
        if any(term in name for term in terms):
            return col
    return None

def style_chart(fig, height=330):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#c7d7f2", size=11),
        margin=dict(l=10, r=10, t=35, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#1d2c47", zeroline=False)
    return fig

def heading(title, subtitle):
    st.markdown(
        f'<div class="section-title">{title}</div>'
        f'<div class="section-sub">{subtitle}</div>',
        unsafe_allow_html=True
    )

def kpi(label, value, note, icon):
    st.markdown(f"""
    <div class="kpi">
      <div class="kpi-label">{icon} &nbsp; {label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)

# ==================================================
# SIDEBAR NAVIGATION
# ==================================================
with st.sidebar:
    st.markdown("## 🧠 ChurnIQ")
    st.caption("CUSTOMER INTELLIGENCE PLATFORM")
    st.divider()

    page = st.radio(
        "WORKSPACE",
        ["Overview", "Customer Analytics", "Churn Prediction", "Model Insights"],
        label_visibility="visible"
    )

    st.divider()
    st.markdown("**PROJECT STATUS**")
    st.success("Dataset loaded")
    st.success("Model loaded")
    st.caption(f"{total_customers:,} customer records")
    st.markdown("---")
    st.caption("Machine Learning • Customer Retention")

# ==================================================
# HERO HEADER
# ==================================================
st.markdown("""
<div class="hero">
  <div class="eyebrow">AI / Machine Learning Workspace</div>
  <h1>🧠 Customer Churn Intelligence</h1>
  <p>Explore customer behaviour, identify churn patterns and estimate customer risk.</p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# KPI ROW
# ==================================================
c1, c2, c3, c4 = st.columns(4)

with c1:
    kpi("Total Customers", f"{total_customers:,}", "Dataset records", "👥")
with c2:
    kpi("Churned Customers", f"{churned_customers:,}", "Observed churn", "⚠️")
with c3:
    kpi("Retained Customers", f"{retained_customers:,}", "Observed retention", "✓")
with c4:
    kpi("Churn Rate", f"{churn_rate:.2f}%", "Share of customers", "📉")

st.write("")

# ==================================================
# OVERVIEW PAGE
# ==================================================
if page == "Overview":
    left, right = st.columns([1.15, 1])

    with left:
        heading(
            "Customer Churn Distribution",
            "Observed churn compared with retained customers"
        )

        pie_data = pd.DataFrame({
            "Customer Status": ["Churned", "Retained"],
            "Customers": [churned_customers, retained_customers]
        })

        fig = px.pie(
            pie_data,
            names="Customer Status",
            values="Customers",
            hole=0.72,
            color="Customer Status",
            color_discrete_map={
                "Churned": "#8b5cf6",
                "Retained": "#06b6d4"
            }
        )
        fig.update_traces(
            textinfo="percent",
            textfont_size=13,
            marker=dict(line=dict(color="#0b1428", width=3))
        )
        fig.add_annotation(
            text=f"<b>{total_customers:,}</b><br><sup>Customers</sup>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="white")
        )
        st.plotly_chart(style_chart(fig, 350), use_container_width=True)

    with right:
        contract_col = find_col("contract")
        heading(
            "Contract vs Churn",
            "Customer count by contract and churn outcome"
        )

        if contract_col:
            chart_df = df.copy()
            chart_df["Churn Status"] = np.where(
                churn_mask, "Churned", "Retained"
            )
            cross = (
                chart_df.groupby([contract_col, "Churn Status"])
                .size().reset_index(name="Customers")
            )
            fig = px.bar(
                cross,
                x=contract_col,
                y="Customers",
                color="Churn Status",
                barmode="group",
                color_discrete_map={
                    "Churned": "#8b5cf6",
                    "Retained": "#06b6d4"
                }
            )
            st.plotly_chart(style_chart(fig, 350), use_container_width=True)
        else:
            st.info("No contract column was found.")

    tenure_col = find_col("tenure")
    monthly_col = find_col("monthly charges", "monthlycharges")

    left, right = st.columns(2)

    with left:
        heading("Tenure Analysis", "Average tenure by observed churn status")
        if tenure_col:
            tenure_data = df.copy()
            tenure_data[tenure_col] = pd.to_numeric(
                tenure_data[tenure_col], errors="coerce"
            )
            tenure_data["Customer Status"] = np.where(
                churn_mask, "Churned", "Retained"
            )
            avg_tenure = (
                tenure_data.groupby("Customer Status")[tenure_col]
                .mean().reset_index()
            )
            fig = px.bar(
                avg_tenure,
                x="Customer Status",
                y=tenure_col,
                color="Customer Status",
                color_discrete_map={
                    "Churned": "#8b5cf6",
                    "Retained": "#06b6d4"
                },
                labels={tenure_col: "Average tenure"}
            )
            st.plotly_chart(style_chart(fig, 300), use_container_width=True)
        else:
            st.info("Tenure column not found.")

    with right:
        heading("Monthly Charges", "Charge distribution by churn status")
        if monthly_col:
            charge_data = df.copy()
            charge_data[monthly_col] = pd.to_numeric(
                charge_data[monthly_col], errors="coerce"
            )
            charge_data["Customer Status"] = np.where(
                churn_mask, "Churned", "Retained"
            )
            fig = px.histogram(
                charge_data,
                x=monthly_col,
                color="Customer Status",
                barmode="overlay",
                opacity=0.72,
                color_discrete_map={
                    "Churned": "#8b5cf6",
                    "Retained": "#06b6d4"
                }
            )
            st.plotly_chart(style_chart(fig, 300), use_container_width=True)
        else:
            st.info("Monthly charges column not found.")

# ==================================================
# CUSTOMER ANALYTICS PAGE
# ==================================================
elif page == "Customer Analytics":
    heading(
        "Customer Analytics Explorer",
        "Filter and explore the dataset used in this project."
    )

    search = st.text_input(
        "🔎 Search columns",
        placeholder="Search by tenure, contract, charges..."
    )

    available_cols = list(df.columns)
    if search:
        available_cols = [
            col for col in available_cols if search.lower() in col.lower()
        ]

    selected_cols = st.multiselect(
        "Columns to display",
        available_cols,
        default=available_cols[:min(7, len(available_cols))]
    )

    if selected_cols:
        st.dataframe(
            df[selected_cols],
            use_container_width=True,
            height=350
        )

    st.download_button(
        "⬇️ Download dataset as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="customer_churn_data.csv",
        mime="text/csv"
    )

    st.divider()
    heading("Feature Summary", "Basic statistics for numeric columns")

    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        st.dataframe(
            numeric_df.describe().T.round(2),
            use_container_width=True
        )
    else:
        st.info("No numeric columns available.")

# ==================================================
# CHURN PREDICTION PAGE
# ==================================================
elif page == "Churn Prediction":
    heading(
        "Customer Risk Assessment",
        "Enter customer information and run the saved ML model."
    )

    id_cols = [
        col for col in df.columns
        if col.lower() in ["customerid", "customer_id", "id"]
    ]
    feature_cols = [
        col for col in df.columns
        if col != target_col and col not in id_cols
    ]

    with st.form("churn_prediction_form"):
        st.markdown("#### Customer profile")
        st.caption("Inputs start with typical values from the dataset.")

        values = {}
        form_cols = st.columns(2)

        for i, col in enumerate(feature_cols):
            series = df[col]
            with form_cols[i % 2]:
                if pd.api.types.is_numeric_dtype(series):
                    median = pd.to_numeric(
                        series, errors="coerce"
                    ).median()
                    if pd.isna(median):
                        median = 0.0
                    values[col] = st.number_input(
                        col.replace("_", " ").title(),
                        value=float(median)
                    )
                else:
                    options = series.dropna().astype(str).unique().tolist()
                    if not options:
                        options = ["Unknown"]

                    modes = series.dropna().astype(str).mode()
                    default = modes.iloc[0] if not modes.empty else options[0]
                    index = options.index(default) if default in options else 0

                    values[col] = st.selectbox(
                        col.replace("_", " ").title(),
                        options,
                        index=index
                    )

        submitted = st.form_submit_button(
            "⚡ Estimate Churn Risk",
            use_container_width=True
        )

    if submitted:
        customer = pd.DataFrame([values])

        try:
            model_features = getattr(model, "feature_names_in_", None)

            if model_features is not None:
                expected = list(model_features)

                # If the model expects the raw input columns, preserve them.
                if set(customer.columns).issubset(set(expected)):
                    model_input = customer.reindex(
                        columns=expected, fill_value=0
                    )
                else:
                    # For models trained after one-hot encoding.
                    encoded = pd.get_dummies(customer)
                    model_input = encoded.reindex(
                        columns=expected, fill_value=0
                    )
            else:
                model_input = customer

            prediction = model.predict(model_input)[0]

            probability = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(model_input)[0]
                classes = list(getattr(model, "classes_", []))

                churn_index = next(
                    (
                        i for i, label in enumerate(classes)
                        if str(label).strip().lower()
                        in ["yes", "1", "true", "churned"]
                    ),
                    None
                )

                if churn_index is not None:
                    probability = float(probabilities[churn_index]) * 100

            pred_text = str(prediction).strip().lower()
            predicted_churn = pred_text in [
                "yes", "1", "true", "churned"
            ]

            st.divider()
            heading("Prediction Result", "Output from your saved model")

            r1, r2 = st.columns([1, 1])

            with r1:
                if predicted_churn:
                    st.error("⚠️ Model predicts: Likely to churn")
                else:
                    st.success("✅ Model predicts: Likely to stay")

            with r2:
                if probability is not None:
                    st.metric(
                        "Estimated churn probability",
                        f"{probability:.2f}%"
                    )
                else:
                    st.info(
                        "This model does not provide probability scores."
                    )

            if probability is not None:
                if probability >= 70:
                    st.warning(
                        "High estimated risk. Consider a retention review."
                    )
                elif probability >= 40:
                    st.info(
                        "Moderate estimated risk. Review customer engagement."
                    )
                else:
                    st.success("Low estimated risk according to this model.")

            st.caption(
                "Predictions are estimates, not guarantees. Validate "
                "model input preprocessing and performance before real use."
            )

        except Exception as exc:
            st.error(
                "Prediction failed. The saved model may expect a different "
                "feature encoding or preprocessing pipeline."
            )
            st.code(str(exc))

# ==================================================
# MODEL INSIGHTS PAGE
# ==================================================
elif page == "Model Insights":
    heading(
        "Model Insights",
        "Inspect model metadata and available evaluation results."
    )

    st.markdown("#### Model information")

    st.write("**Model class:**", type(model).__name__)

    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None:
        st.write("**Expected input features:**", len(model_features))
        st.dataframe(
            pd.DataFrame({"Model input features": list(model_features)}),
            use_container_width=True,
            height=280
        )
    else:
        st.info(
            "The saved model does not expose feature_names_in_. "
            "Confirm the training preprocessing before deployment."
        )

    comparison_path = BASE / "model_comparison.csv"

    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        st.markdown("#### Saved model comparison")
        st.dataframe(comparison, use_container_width=True)

        numeric_cols = comparison.select_dtypes(
            include=np.number
        ).columns.tolist()

        if numeric_cols:
            metric_col = st.selectbox(
                "Select evaluation metric",
                numeric_cols
            )
            fig = px.bar(
                comparison,
                x=comparison.columns[0],
                y=metric_col,
                color=comparison.columns[0],
                title=f"Model comparison — {metric_col}"
            )
            st.plotly_chart(style_chart(fig, 350), use_container_width=True)
    else:
        st.info(
            "model_comparison.csv is not available. Add evaluation results "
            "from your training script to display comparisons."
        )

# ==================================================
# FOOTER
# ==================================================
st.divider()
st.caption(
    "ChurnIQ • Customer Analytics & Machine Learning • Built with Streamlit"
)

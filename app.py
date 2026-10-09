
import pandas as pd
import streamlit as st
from pathlib import Path

from model import load_or_train_model, DATASET_PATH

 
# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Risk Alert Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #e5edf8;
}

.block-container {
    padding: 1.5rem 2.5rem 2rem 2.5rem;
    max-width: 1500px;
}

header[data-testid="stHeader"] {
    background: transparent;
}

h1, h2, h3, p, label {
    color: #e5edf8;
}

.info-card {
    background: #111a2b;
    border: 1px solid #263750;
    border-radius: 14px;
    padding: 18px 20px;
    min-height: 100px;
}

.info-label {
    color: #91a5bf;
    font-size: 13px;
    margin-bottom: 9px;
}

.info-value {
    color: #f1f7ff;
    font-size: 24px;
    font-weight: 700;
}

.section-card {
    background: #101a2b;
    border: 1px solid #253650;
    border-radius: 14px;
    padding: 15px 20px;
    margin-top: 12px;
    margin-bottom: 12px;
}

.section-heading {
    font-size: 18px;
    font-weight: 700;
    color: #eaf3ff;
}

.section-description {
    font-size: 12px;
    color: #8fa6c2;
    margin-top: 5px;
}

div[data-testid="stNumberInput"] input {
    background: #17243a;
    color: #f1f7ff;
    border: 1px solid #30445f;
    border-radius: 8px;
}

div[data-testid="stNumberInput"] label {
    color: #c3d4e8;
    font-size: 13px;
}

div[data-testid="stNumberInput"] button {
    background: #1b2b43;
    color: #dbeafe;
}

.stFormSubmitButton > button {
    background: linear-gradient(90deg, #0891b2, #2563eb);
    color: white;
    border: none;
    border-radius: 9px;
    padding: 12px 20px;
    font-weight: 700;
    min-height: 46px;
}

.stFormSubmitButton > button:hover {
    background: linear-gradient(90deg, #0e7490, #1d4ed8);
    color: white;
    border: none;
}

div[data-testid="stMetric"] {
    background: #111a2b;
    border: 1px solid #263750;
    border-radius: 12px;
    padding: 15px;
}

div[data-testid="stProgress"] > div > div {
    background: #06b6d4;
}

div[data-testid="stExpander"] {
    background: #101a2b;
    border: 1px solid #263750;
    border-radius: 12px;
}

.footer {
    color: #71849e;
    text-align: center;
    font-size: 12px;
    padding: 25px 0 5px 0;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# LOAD MODEL AND DATA
# ==================================================

@st.cache_resource
def get_model():
    return load_or_train_model()


@st.cache_data
def get_data():
    return pd.read_csv(DATASET_PATH)


try:
    model, features = get_model()
    df = get_data()

except Exception as exc:
    st.error(f"Could not load model or dataset: {exc}")
    st.stop()


# ==================================================
# HEADER BANNER
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
BANNER_PATH = BASE_DIR / "risk_alert_banner.png"

if BANNER_PATH.exists():
    st.image(
        str(BANNER_PATH),
        width=850
    )
else:
    st.markdown("""
    <div style="
        background: linear-gradient(120deg, #0e2345, #12345e);
        border: 1px solid #17609a;
        border-radius: 18px;
        padding: 28px;
        margin-bottom: 20px;
    ">
        <div style="
            color: #36e0d0;
            font-size: 14px;
            font-weight: 700;
            letter-spacing: 1px;
        ">
            ML POWERED • CLASSIFICATION
        </div>
        <h1 style="
            color: #f1f7ff;
            font-size: 36px;
            margin-bottom: 5px;
        ">
            🛡️ Risk Alert Classifier
        </h1>
        <p style="
            color: #b8cbe2;
            font-size: 16px;
        ">
            Intelligent Customer Risk Prediction using Logistic Regression
        </p>
    </div>
    """, unsafe_allow_html=True)


# ==================================================
# FEATURE LABELS
# ==================================================

labels = {
    "customer_id": "Customer ID",
    "age": "Age",
    "annual_income_inr": "Annual Income (INR)",
    "credit_score": "Credit Score",
    "credit_utilization_ratio": "Credit Utilization Ratio",
    "missed_payments_12m": "Missed Payments (12 months)",
    "avg_late_payment_days": "Average Late Payment Days",
    "monthly_transaction_count": "Monthly Transaction Count",
    "monthly_spend_inr": "Monthly Spend (INR)",
    "cash_advance_count_6m": "Cash Advances (6 months)",
    "complaints_last_6m": "Complaints (6 months)",
    "failed_login_attempts_3m": "Failed Login Attempts (3 months)",
    "account_tenure_months": "Account Tenure (months)",
    "debt_balance_inr": "Debt Balance (INR)",
}


# ==================================================
# DASHBOARD SUMMARY CARDS
# ==================================================

st.write("")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="info-card">
        <div class="info-label">Total Customers</div>
        <div class="info-value">{len(df):,}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="info-card">
        <div class="info-label">Model Features</div>
        <div class="info-value">{len(features)}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="info-card">
        <div class="info-label">ML Algorithm</div>
        <div class="info-value" style="font-size:20px;">
            Logistic Regression
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==================================================
# CUSTOMER RISK DISTRIBUTION
# ==================================================

st.write("")
st.markdown("## 📊 Risk Analytics")

st.caption(
    "Overview of safe and risky customers in the dataset."
)

if "risk_status" in df.columns:

    with st.container(border=True):

        st.markdown("### Customer Risk Distribution")

        # Support numeric and common text labels.
        risk_values = df["risk_status"]

        numeric_risk = pd.to_numeric(
            risk_values,
            errors="coerce"
        )

        risk_labels = risk_values.astype(str).str.strip().str.lower()

        mapped_risk = risk_labels.map({
            "0": "Safe Customer",
            "safe": "Safe Customer",
            "low risk": "Safe Customer",
            "1": "Risky Customer",
            "risky": "Risky Customer",
            "high risk": "Risky Customer"
        })

        mapped_risk = mapped_risk.fillna(
            numeric_risk.map({
                0: "Safe Customer",
                1: "Risky Customer"
            })
        )

        risk_counts = (
            mapped_risk
            .value_counts()
            .reindex(
                ["Safe Customer", "Risky Customer"],
                fill_value=0
            )
        )

        if risk_counts.sum() > 0:

            risk_chart = risk_counts.rename(
                "Customers"
            ).to_frame()

            st.bar_chart(
                risk_chart,
                color="#06b6d4",
                height=300
            )

            total = int(risk_counts.sum())

            safe_pct = (
                risk_counts["Safe Customer"] / total * 100
            )

            risky_pct = (
                risk_counts["Risky Customer"] / total * 100
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Safe Customers",
                f"{safe_pct:.1f}%"
            )

            col2.metric(
                "Risky Customers",
                f"{risky_pct:.1f}%"
            )

            st.caption(
                f"Based on {total:,} records with recognized risk labels."
            )

        else:
            st.info(
                "No recognized safe/risky labels found in risk_status."
            )

else:
    st.info(
        "The dataset does not contain a risk_status column."
    )


st.write("")
st.divider()


# ==================================================
# CUSTOMER INPUT FORM
# ==================================================

st.markdown("## 👤 Customer Risk Assessment")

st.caption(
    "Enter the customer's information below to predict the risk class."
)


sections = {
    "Customer Profile": {
        "description": "Basic customer and account information.",
        "features": [
            "customer_id",
            "age",
            "annual_income_inr",
            "account_tenure_months"
        ]
    },

    "Credit & Payment Details": {
        "description": "Credit history, debt and payment patterns.",
        "features": [
            "credit_score",
            "credit_utilization_ratio",
            "missed_payments_12m",
            "avg_late_payment_days",
            "debt_balance_inr"
        ]
    },

    "Transaction & Activity": {
        "description": "Transaction activity and recent customer alerts.",
        "features": [
            "monthly_transaction_count",
            "monthly_spend_inr",
            "cash_advance_count_6m",
            "complaints_last_6m",
            "failed_login_attempts_3m"
        ]
    }
}


integer_features = {
    "customer_id",
    "age",
    "missed_payments_12m",
    "monthly_transaction_count",
    "cash_advance_count_6m",
    "complaints_last_6m",
    "failed_login_attempts_3m",
    "account_tenure_months",
    "debt_balance_inr"
}


def get_default(feature):

    if feature not in df.columns:
        return 0

    median = pd.to_numeric(
        df[feature],
        errors="coerce"
    ).median()

    if pd.isna(median):
        return 0

    return median


values = {}


with st.form("risk_prediction_form"):

    used_features = set()

    for section_name, section_info in sections.items():

        section_features = [
            f for f in section_info["features"]
            if f in features
        ]

        if not section_features:
            continue

        used_features.update(section_features)

        st.markdown(f"""
        <div class="section-card">
            <div class="section-heading">{section_name}</div>
            <div class="section-description">
                {section_info["description"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        for i, feature in enumerate(section_features):

            target_col = col1 if i % 2 == 0 else col2

            label = labels.get(
                feature,
                feature.replace("_", " ").title()
            )

            default = get_default(feature)

            with target_col:

                is_integer = (
                    feature in integer_features
                    or (
                        feature in df.columns
                        and pd.api.types.is_integer_dtype(
                            df[feature].dtype
                        )
                    )
                )

                if is_integer:

                    values[feature] = st.number_input(
                        label,
                        value=int(default),
                        step=1,
                        key=f"input_{feature}"
                    )

                else:

                    values[feature] = st.number_input(
                        label,
                        value=float(default),
                        step=0.1,
                        format=(
                            "%.3f"
                            if feature == "credit_utilization_ratio"
                            else "%.2f"
                        ),
                        key=f"input_{feature}"
                    )


    # Additional model features

    other_features = [
        f for f in features
        if f not in used_features
    ]

    if other_features:

        st.markdown("""
        <div class="section-card">
            <div class="section-heading">Additional Information</div>
            <div class="section-description">
                Other features required by the model.
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        for i, feature in enumerate(other_features):

            target_col = col1 if i % 2 == 0 else col2

            default = get_default(feature)

            label = labels.get(
                feature,
                feature.replace("_", " ").title()
            )

            with target_col:

                is_integer = (
                    feature in integer_features
                    or (
                        feature in df.columns
                        and pd.api.types.is_integer_dtype(
                            df[feature].dtype
                        )
                    )
                )

                if is_integer:

                    values[feature] = st.number_input(
                        label,
                        value=int(default),
                        step=1,
                        key=f"input_{feature}"
                    )

                else:

                    values[feature] = st.number_input(
                        label,
                        value=float(default),
                        step=0.1,
                        key=f"input_{feature}"
                    )


    st.write("")

    submitted = st.form_submit_button(
        "🔍  Predict Customer Risk",
        type="primary",
        use_container_width=True
    )


# ==================================================
# PREDICTION SUMMARY
# ==================================================

if submitted:

    input_df = pd.DataFrame(
        [[values[f] for f in features]],
        columns=features
    )

    try:

        prediction = model.predict(input_df)[0]

        probabilities = model.predict_proba(input_df)[0]

        classes = list(model.classes_)

        # Find probability of risky class (1).
        if 1 in classes:
            risky_probability = float(
                probabilities[classes.index(1)]
            )
        elif "1" in classes:
            risky_probability = float(
                probabilities[classes.index("1")]
            )
        else:
            risky_probability = 0.0

        is_risky = str(prediction).strip().lower() in (
            "1", "1.0", "risky", "high risk"
        )

        st.write("")
        st.divider()

        st.markdown("## 📋 Prediction Summary")

        result_col, probability_col = st.columns([1.2, 1])


        # Customer status card

        with result_col:

            with st.container(border=True):

                st.caption("CUSTOMER RISK STATUS")

                if is_risky:

                    st.error(
                        "⚠️ Risky Customer",
                        icon="⚠️"
                    )

                    st.write(
                        "The model classified this customer as risky."
                    )

                else:

                    st.success(
                        "✅ Safe Customer",
                        icon="✅"
                    )

                    st.write(
                        "The model classified this customer as safe."
                    )


        # Risk probability card

        with probability_col:

            with st.container(border=True):

                st.caption("ESTIMATED RISK PROBABILITY")

                st.metric(
                    label="Risk Probability",
                    value=f"{risky_probability:.1%}"
                )

                st.progress(
                    max(0.0, min(1.0, risky_probability))
                )

                st.caption(
                    "Model-estimated probability of the risky class."
                )


        st.info(
            "This prediction is an estimate for educational and "
            "demonstration purposes. It should not be used alone "
            "to make financial or customer eligibility decisions."
        )


    except Exception as exc:

        st.error(f"Prediction failed: {exc}")


# ==================================================
# ABOUT THE MODEL
# ==================================================

with st.expander("ℹ️ About the Model"):

    st.markdown("""
    ### Risk Alert Classifier

    This application predicts customer risk using a
    Logistic Regression classification model.

    **Machine Learning Pipeline**
    - KNN Imputer
    - Standard Scaler
    - Logistic Regression

    **Purpose**

    The model uses the customer's available numeric features
    to estimate the probability of belonging to the risky class.

    **Important**

    This is an educational and demonstration project.
    Evaluate the model using held-out test data before considering
    any real-world application.
    """)


# ==================================================
# FOOTER
# ==================================================

st.markdown("""
<div class="footer">
    🛡️ Risk Alert Classifier | Machine Learning Project
</div>
""", unsafe_allow_html=True)

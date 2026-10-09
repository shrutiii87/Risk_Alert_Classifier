import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from model import (
    CATEGORICAL_FEATURES,
    DATASET_PATH,
    FEATURE_LABELS,
    INTEGER_FEATURES,
    TARGET,
    evaluate_model,
    load_or_train_model,
)

st.set_page_config(
    page_title="Risk Alert Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================== CSS ==========================================
st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #e5edf8;
}
.block-container {
    max-width: 1500px;
    padding: 0.65rem 1rem 0.8rem;
}
[data-testid="stHeader"] {
    background: #0b1120;
}
h1, h2, h3, h4, p, label {
    color: #e5edf8;
}
[data-testid="stVerticalBlock"] {
    gap: 0.55rem;
}
[data-testid="stMetric"] {
    background: #111c2e;
    border: 1px solid #263a54;
    border-radius: 12px;
    padding: 12px 15px;
    min-height: 90px;
}
[data-testid="stMetricLabel"] {
    font-size: 13px;
}
[data-testid="stMetricValue"] {
    font-size: 25px;
}
[data-testid="stTabs"] {
    margin-top: 0;
}
[data-testid="stTabs"] [role="tablist"] {
    gap: 5px;
    border-bottom: 1px solid #263a54;
}
[data-testid="stTabs"] button {
    padding: 9px 10px;
}
[data-testid="stForm"] {
    background: #0e1829;
    border: 1px solid #263a54;
    border-radius: 12px;
    padding: 14px;
}
.stButton button,
.stFormSubmitButton button,
.stDownloadButton button {
    border-radius: 8px;
    font-weight: 650;
}
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid #263a54;
    border-radius: 9px;
}
hr {
    margin: 0.35rem 0;
    border-color: #263a54;
}
.footer {
    text-align: center;
    color: #71849e;
    font-size: 12px;
    padding: 8px 0 0;
}
div[data-testid="stAlert"] {
    margin: 0.25rem 0;
}
@media (max-width: 900px) {
    .block-container {
        padding: 0.5rem 0.6rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 21px;
    }
}
</style>
""", unsafe_allow_html=True)


# ============================== SVG BANNER ===================================
RISK_BANNER_SVG = r"""
<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 1800 480" width="100%"
     role="img" aria-labelledby="bannerTitle">

<title id="bannerTitle">Risk Alert Classifier — Animated ML Banner</title>

<defs>
    <linearGradient id="riskBg" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#0B1830"/>
        <stop offset="100%" stop-color="#163D68"/>
    </linearGradient>

    <linearGradient id="riskShield" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#27E6FF"/>
        <stop offset="100%" stop-color="#087BE5"/>
    </linearGradient>

    <clipPath id="riskPanel">
        <rect width="1800" height="480" rx="28"/>
    </clipPath>

    <clipPath id="riskRadarClip">
        <circle r="108"/>
    </clipPath>

    <style>
        .riskFont {
            font-family: Inter, "Segoe UI", Arial, sans-serif;
        }
        .riskEyebrow {
            font-size: 19px;
            font-weight: 800;
            letter-spacing: 2px;
            fill: #39E8D4;
        }
        .riskHeading {
            font-size: 56px;
            font-weight: 760;
            letter-spacing: -1px;
            fill: #F3F7FF;
        }
        .riskSubtitle {
            font-size: 24px;
            fill: #B8CBE2;
        }
        .riskLabel {
            font-size: 15px;
            font-weight: 700;
            letter-spacing: 1.5px;
            fill: #81A9CC;
        }
        .riskValue {
            font-size: 19px;
            font-weight: 700;
            fill: #E7F3FF;
        }
        .riskSmall {
            font-size: 16px;
            fill: #A8BED5;
        }
        .riskFloat {
            transform-box: fill-box;
            transform-origin: center;
            animation: riskFloat 3s ease-in-out infinite;
        }
        .riskPulse {
            transform-box: fill-box;
            transform-origin: center;
            animation: riskPulse 2s ease-in-out infinite;
        }
        .riskBlink {
            animation: riskBlink 1.7s ease-in-out infinite;
        }
        .riskScan {
            animation: riskScan 3.5s ease-in-out infinite;
        }
        .riskDraw {
            stroke-dasharray: 40;
            stroke-dashoffset: 40;
            animation: riskDraw 3s ease-in-out infinite;
        }
        .riskRing {
            transform-box: fill-box;
            transform-origin: center;
            animation: riskRing 3s ease-out infinite;
        }
        @keyframes riskFloat {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-7px); }
        }
        @keyframes riskPulse {
            0%, 100% { opacity: .55; transform: scale(.9); }
            50% { opacity: 1; transform: scale(1.12); }
        }
        @keyframes riskBlink {
            0%, 100% { opacity: .25; }
            50% { opacity: 1; }
        }
        @keyframes riskDraw {
            0%, 10% { stroke-dashoffset: 40; }
            45%, 75% { stroke-dashoffset: 0; }
            100% { stroke-dashoffset: 40; }
        }
        @keyframes riskScan {
            0% { transform: translateY(-105px); opacity: 0; }
            15% { opacity: .7; }
            85% { opacity: .7; }
            100% { transform: translateY(105px); opacity: 0; }
        }
        @keyframes riskRing {
            0% { transform: scale(.65); opacity: .8; }
            100% { transform: scale(1.3); opacity: 0; }
        }
        @media (prefers-reduced-motion: reduce) {
            * { animation: none !important; }
        }
    </style>
</defs>

<g clip-path="url(#riskPanel)">
    <rect width="1800" height="480" fill="url(#riskBg)"/>

    <g stroke="#6BBFFF" stroke-width="1" opacity=".055">
        <path d="M0 80H1800 M0 140H1800 M0 200H1800
                 M0 260H1800 M0 320H1800 M0 380H1800 M0 440H1800"/>
        <path d="M100 0V480 M200 0V480 M300 0V480 M400 0V480
                 M500 0V480 M600 0V480 M1200 0V480 M1300 0V480
                 M1400 0V480 M1500 0V480 M1600 0V480 M1700 0V480"/>
    </g>

    <g class="riskFont">
        <text x="48" y="67" class="riskEyebrow">
            ML POWERED <tspan fill="#779BBE">•</tspan> CLASSIFICATION
        </text>

        <g transform="translate(49 108)">
            <path d="M22 0L42 7V26Q40 45 22 56Q4 45 2 26V7Z"
                  fill="#B7DCF5" stroke="#FFFFFF" stroke-width="2"/>
            <path d="M22 0V56Q4 45 2 26V7Z" fill="url(#riskShield)"/>
            <path d="M11 26L19 34L32 17"
                  fill="none" stroke="#FFFFFF" stroke-width="2.8"
                  stroke-linecap="round" stroke-linejoin="round"
                  class="riskDraw"/>
        </g>

        <text x="110" y="151" class="riskHeading">Risk Alert Classifier</text>
        <text x="49" y="207" class="riskSubtitle">
            Intelligent Customer Risk Prediction using Logistic Regression
        </text>

        <rect x="49" y="235" width="125" height="4" rx="2" fill="#35E8D4">
            <animate attributeName="width" values="45;160;45"
                     dur="3s" repeatCount="indefinite"/>
        </rect>

        <g transform="translate(49 285)">
            <rect width="240" height="112" rx="12"
                  fill="#102C4A" stroke="#2C628B"/>
            <circle cx="24" cy="26" r="5" fill="#35E8D4" class="riskPulse"/>
            <text x="40" y="31" class="riskLabel">MODEL</text>
            <text x="20" y="66" class="riskValue">Logistic Regression</text>
            <text x="20" y="91" class="riskSmall">Binary classification</text>
        </g>

        <g transform="translate(305 285)">
            <rect width="240" height="112" rx="12"
                  fill="#102C4A" stroke="#2C628B"/>
            <circle cx="24" cy="26" r="5" fill="#44B6FF" class="riskPulse"/>
            <text x="40" y="31" class="riskLabel">PREDICTION</text>
            <text x="20" y="66" class="riskValue">Low Risk / High Risk</text>
            <text x="20" y="91" class="riskSmall">Customer risk assessment</text>
        </g>

        <g transform="translate(561 285)">
            <rect width="240" height="112" rx="12"
                  fill="#102C4A" stroke="#2C628B"/>
            <circle cx="24" cy="26" r="5" fill="#F7B955" class="riskPulse"/>
            <text x="40" y="31" class="riskLabel">PIPELINE</text>
            <text x="20" y="66" class="riskValue">Impute • Train • Predict</text>
            <text x="20" y="91" class="riskSmall">Data preparation included</text>
        </g>

        <text x="49" y="435" class="riskSmall">SUPERVISED LEARNING</text>
        <circle cx="244" cy="430" r="3" fill="#35E8D4"/>
        <text x="258" y="435" class="riskSmall">CUSTOMER RISK ANALYTICS</text>
    </g>

    <g class="riskFont">
        <rect x="1195" y="36" width="560" height="405" rx="20"
              fill="#0A1930" stroke="#2C6590" stroke-width="1.2"/>

        <text x="1225" y="72" class="riskLabel">LIVE RISK MONITOR</text>
        <circle cx="1715" cy="66" r="4" fill="#35E8D4" class="riskPulse"/>
        <text x="1699" y="91" font-size="11"
              text-anchor="end" fill="#66DCCF">SCANNING</text>

        <g transform="translate(1475 237)">
            <circle r="115" fill="#0D2944"
                    stroke="#347DA5" stroke-width="1.4"/>

            <g clip-path="url(#riskRadarClip)">
                <g fill="none" stroke="#54B8D5" stroke-width="1" opacity=".65">
                    <circle r="27"/>
                    <circle r="54"/>
                    <circle r="81"/>
                    <circle r="108"/>
                </g>

                <g stroke="#54B8D5" stroke-width="1" opacity=".55">
                    <path d="M-108 0H108 M0-108V108"/>
                    <path d="M-76.4-76.4L76.4 76.4
                             M-76.4 76.4L76.4-76.4" opacity=".3"/>
                </g>

                <g class="riskScan">
                    <line x1="-108" y1="0" x2="108" y2="0"
                          stroke="#37E8D4" stroke-width="1.5" opacity=".65"/>
                </g>

                <circle cx="-48" cy="-42" r="4" fill="#3DFFE0"/>
                <circle cx="57" cy="-20" r="4" fill="#FF657D" class="riskPulse"/>
                <circle cx="37" cy="66" r="4" fill="#47BFFF"/>
                <circle cx="-68" cy="40" r="3" fill="#47BFFF" class="riskBlink"/>

                <circle cx="-48" cy="-42" r="8" fill="none"
                        stroke="#3DFFE0" class="riskRing"/>
                <circle cx="57" cy="-20" r="8" fill="none"
                        stroke="#FF657D" class="riskRing"
                        style="animation-delay:1s"/>
            </g>

            <circle r="123" fill="none" stroke="#45C5DE"
                    stroke-width="1" stroke-dasharray="3 9" opacity=".7"/>

            <g class="riskFloat">
                <path d="M0-31L24-22L22 3Q18 21 0 31Q-18 21-22 3L-24-22Z"
                      fill="#D8ECFF" stroke="#FFFFFF" stroke-width="1.5"/>
                <path d="M0-31L24-22L22 3Q18 21 0 31Z" fill="#9CCCF0"/>
                <path d="M0-31V31Q-18 21-22 3L-24-22Z"
                      fill="url(#riskShield)"/>
                <path d="M-11 0L-3 8L12-10"
                      fill="none" stroke="#FFFFFF" stroke-width="3"
                      stroke-linecap="round" stroke-linejoin="round"
                      class="riskDraw"/>
            </g>
        </g>

        <circle cx="1230" cy="404" r="4" fill="#35E8D4"/>
        <text x="1243" y="409" font-size="14" fill="#B9CBE2">NORMAL</text>
        <circle cx="1355" cy="404" r="4" fill="#47BFFF"/>
        <text x="1368" y="409" font-size="14" fill="#B9CBE2">MONITOR</text>
        <circle cx="1500" cy="404" r="4" fill="#FF657D"/>
        <text x="1513" y="409" font-size="14" fill="#B9CBE2">HIGH-RISK SIGNAL</text>
    </g>

    <g transform="translate(1240 455)">
        <path d="M0 0H190" stroke="#285B7D" stroke-width="2"/>
        <circle cx="0" cy="0" r="3" fill="#35E8D4"/>
        <circle cy="0" r="3" fill="#35E8D4">
            <animate attributeName="cx" values="0;190;0"
                     dur="4s" repeatCount="indefinite"/>
        </circle>
    </g>

    <rect x="1" y="1" width="1798" height="478" rx="28"
          fill="none" stroke="#2878B5" stroke-width="1.3"/>
</g>
</svg>
"""

st.components.v1.html(
    RISK_BANNER_SVG,
    height=405,
    scrolling=False,
)


# ============================== LOAD DATA ====================================
@st.cache_data(show_spinner=False)
def get_dataset():
    return pd.read_csv(DATASET_PATH)


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_or_train_model()


@st.cache_data(show_spinner="Evaluating model...")
def get_evaluation():
    return evaluate_model()


try:
    df = get_dataset()
    model, features = get_model()
except Exception as exc:
    st.error(f"❌ Could not load the dataset or model: {exc}")
    st.info(
        "Keep app.py, model.py, and the dataset CSV in the correct locations."
    )
    st.stop()


# ============================== KPI CARDS ====================================
total_rows = len(df)
target_values = pd.to_numeric(df[TARGET], errors="coerce")
high_count = int((target_values == 1).sum())
low_count = int((target_values == 0).sum())
high_pct = high_count / total_rows * 100 if total_rows else 0
missing_cells = int(df[features].isna().sum().sum())

k1, k2, k3, k4 = st.columns(4, gap="small")
k1.metric("📁 Dataset Records", f"{total_rows:,}")
k2.metric("🧩 Model Input Features", f"{len(features):,}")
k3.metric("⚠️ High Risk Records", f"{high_count:,}", f"{high_pct:.1f}% of dataset")
k4.metric("🔎 Missing Feature Values", f"{missing_cells:,}")

st.markdown("---")

overview_tab, predict_tab, evaluate_tab, about_tab = st.tabs([
    "📊 Overview",
    "👤 Predict Customer Risk",
    "🧪 Model Evaluation",
    "ℹ️ About",
])


# ============================== OVERVIEW =====================================
with overview_tab:
    left, right = st.columns(2, gap="medium")

    with left:
        st.subheader("📊 Customer Risk Distribution")
        distribution = pd.DataFrame(
            {"Customers": [low_count, high_count]},
            index=["Low Risk", "High Risk"],
        )
        st.bar_chart(distribution, height=260, color="#22c5d9")

    with right:
        st.subheader("🩺 Dataset Health")
        health_df = pd.DataFrame({
            "Missing Values": df[features].isna().sum(),
            "Unique Values": df[features].nunique(dropna=True),
        })
        st.dataframe(
            health_df,
            use_container_width=True,
            height=260,
        )

    st.subheader("🤖 Model Snapshot")
    a, b, c = st.columns(3, gap="small")
    a.metric("Algorithm", "Logistic Regression")
    b.metric("Task", "Binary Classification")
    c.metric("Target", "Low Risk / High Risk")

    st.caption(
        "Dataset distribution describes the records; it does not measure model accuracy."
    )


# ============================== PREDICTION ===================================
with predict_tab:
    st.subheader("👤 Customer Risk Assessment")
    st.caption(
        "Enter customer details and select Predict to get a model-generated risk estimate."
    )
    st.info(
        "🎓 Educational demonstration only. Do not use this result alone "
        "for consequential financial decisions."
    )

    values = {}

    with st.form("risk_prediction_form"):
        col1, col2, col3 = st.columns(3, gap="medium")

        sections = [
            ("👤 Customer Profile", [
                "age",
                "gender",
                "region",
                "employment_type",
                "annual_income_inr",
                "account_tenure_months",
            ], col1),
            ("💳 Credit & Payments", [
                "credit_score",
                "credit_utilization_ratio",
                "missed_payments_12m",
                "avg_late_payment_days",
                "debt_balance_inr",
            ], col2),
            ("📊 Transactions & Activity", [
                "monthly_transaction_count",
                "monthly_spend_inr",
                "cash_advance_count_6m",
                "complaints_last_6m",
                "failed_login_attempts_3m",
            ], col3),
        ]

        for section_title, section_features, column in sections:
            with column:
                st.markdown(f"#### {section_title}")

                for feature in section_features:
                    if feature not in features:
                        continue

                    label = FEATURE_LABELS.get(
                        feature,
                        feature.replace("_", " ").title(),
                    )
                    series = df[feature]

                    if (
                        feature in CATEGORICAL_FEATURES
                        or not pd.api.types.is_numeric_dtype(series)
                    ):
                        options = sorted(
                            series.dropna().astype(str).unique().tolist()
                        )
                        if not options:
                            options = ["Unknown"]

                        modes = series.dropna().astype(str).mode()
                        default = (
                            str(modes.iloc[0])
                            if not modes.empty
                            else options[0]
                        )
                        index = options.index(default) if default in options else 0

                        values[feature] = st.selectbox(
                            label,
                            options,
                            index=index,
                            key=f"input_{feature}",
                        )

                    else:
                        clean = pd.to_numeric(
                            series, errors="coerce"
                        ).dropna()

                        minimum = float(clean.min()) if not clean.empty else 0.0
                        maximum = float(clean.max()) if not clean.empty else 100.0
                        default = float(clean.median()) if not clean.empty else 0.0

                        if feature == "credit_utilization_ratio":
                            minimum = max(0.0, minimum)
                            maximum = min(1.0, maximum)

                        if minimum >= maximum:
                            maximum = minimum + 1.0

                        default = min(max(default, minimum), maximum)

                        if feature in INTEGER_FEATURES:
                            values[feature] = st.number_input(
                                label,
                                min_value=int(np.floor(minimum)),
                                max_value=int(np.ceil(maximum)),
                                value=int(round(default)),
                                step=1,
                                key=f"input_{feature}",
                            )
                        else:
                            values[feature] = st.number_input(
                                label,
                                min_value=float(minimum),
                                max_value=float(maximum),
                                value=float(default),
                                step=(
                                    0.01
                                    if feature == "credit_utilization_ratio"
                                    else 1.0
                                ),
                                format=(
                                    "%.3f"
                                    if feature == "credit_utilization_ratio"
                                    else "%.2f"
                                ),
                                key=f"input_{feature}",
                            )

        submitted = st.form_submit_button(
            "🔍 Predict Customer Risk",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        input_df = pd.DataFrame(
            [[values.get(feature, np.nan) for feature in features]],
            columns=features,
        )

        try:
            prediction = int(model.predict(input_df)[0])
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)

            if 1 not in classes:
                st.error(
                    "The loaded model does not contain the High Risk class (1)."
                )
            else:
                high_probability = float(
                    probabilities[classes.index(1)]
                )

                result_col, probability_col = st.columns(2, gap="medium")

                with result_col:
                    st.markdown("#### 🎯 Prediction Result")

                    if prediction == 1:
                        st.error("⚠️ High Risk")
                        st.write(
                            "The model classified this customer as High Risk."
                        )
                    else:
                        st.success("✅ Low Risk")
                        st.write(
                            "The model classified this customer as Low Risk."
                        )

                with probability_col:
                    st.markdown("#### 📈 High Risk Probability")
                    st.metric(
                        "Estimated probability",
                        f"{high_probability:.1%}",
                    )
                    st.progress(
                        float(np.clip(high_probability, 0, 1))
                    )

                export_df = input_df.copy()
                export_df["predicted_risk_status"] = prediction
                export_df["predicted_risk_label"] = (
                    "High Risk" if prediction == 1 else "Low Risk"
                )
                export_df["high_risk_probability"] = high_probability

                st.download_button(
                    "⬇️ Download Prediction CSV",
                    data=export_df.to_csv(index=False).encode("utf-8"),
                    file_name="risk_prediction.csv",
                    mime="text/csv",
                )

        except Exception as exc:
            st.error(f"❌ Prediction failed: {exc}")


# ============================== MODEL EVALUATION ==============================
with evaluate_tab:
    st.subheader("🧪 Held-Out Test-Set Evaluation")
    st.caption(
        "Metrics and confusion matrix are calculated from the evaluation results "
        "returned by model.py."
    )

    try:
        metrics = get_evaluation()

        # Metric cards
        m1, m2, m3, m4, m5 = st.columns(5, gap="small")
        m1.metric("🎯 Accuracy", f"{metrics['accuracy']:.1%}")
        m2.metric("🔎 High Risk Precision", f"{metrics['precision']:.1%}")
        m3.metric("⚠️ High Risk Recall", f"{metrics['recall']:.1%}")
        m4.metric("⚖️ High Risk F1", f"{metrics['f1']:.1%}")
        m5.metric("📈 ROC-AUC", f"{metrics['roc_auc']:.3f}")

        st.markdown("---")

        cm = np.asarray(metrics["confusion_matrix"], dtype=int)

        if cm.shape != (2, 2):
            st.error(
                f"Expected a 2×2 confusion matrix for binary classification; "
                f"received shape {cm.shape}."
            )
            st.stop()

        # Heatmap and table aligned side by side
        heatmap_col, details_col = st.columns(
            [1.15, 1],
            gap="medium",
        )

        with heatmap_col:
            st.markdown("### 🧮 Confusion Matrix")

            fig, ax = plt.subplots(figsize=(7, 5))

            # Dark navy background to match the Streamlit dashboard
            fig.patch.set_facecolor("#0b1120")
            ax.set_facecolor("#0b1120")

            image = ax.imshow(
                cm,
                interpolation="nearest",
                cmap="Blues",
                aspect="equal",
            )

            colorbar = fig.colorbar(
                image,
                ax=ax,
                fraction=0.046,
                pad=0.04,
            )
            colorbar.ax.tick_params(colors="#dce7f5", labelsize=9)
            colorbar.outline.set_edgecolor("#263a54")

            class_names = ["Low Risk", "High Risk"]

            ax.set(
                xticks=np.arange(2),
                yticks=np.arange(2),
                xticklabels=class_names,
                yticklabels=class_names,
                xlabel="Predicted",
                ylabel="Actual",
                title="Confusion Matrix - Logistic Regression",
            )

            ax.xaxis.label.set_color("#dce7f5")
            ax.yaxis.label.set_color("#dce7f5")
            ax.title.set_color("#e5edf8")

            ax.tick_params(
                axis="both",
                colors="#dce7f5",
                labelsize=10,
            )

            # Put actual values in all four cells
            threshold = cm.max() / 2.0 if cm.size else 0

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    text_color = (
                        "#ffffff"
                        if cm[i, j] > threshold
                        else "#172033"
                    )

                    ax.text(
                        j,
                        i,
                        f"{cm[i, j]:,}",
                        ha="center",
                        va="center",
                        fontsize=15,
                        fontweight="bold",
                        color=text_color,
                    )

            # Separate the cells with subtle borders
            ax.set_xticks(
                np.arange(-0.5, 2, 1),
                minor=True,
            )
            ax.set_yticks(
                np.arange(-0.5, 2, 1),
                minor=True,
            )
            ax.grid(
                which="minor",
                color="#263a54",
                linestyle="-",
                linewidth=1.5,
            )
            ax.tick_params(
                which="minor",
                bottom=False,
                left=False,
            )

            for spine in ax.spines.values():
                spine.set_color("#263a54")

            fig.tight_layout(pad=1.5)

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.caption(
                "Rows = actual labels. Columns = predicted labels."
            )

        with details_col:
            st.markdown("### 📋 Confusion Matrix Values")

            cm_df = pd.DataFrame(
                cm,
                index=["Actual Low Risk", "Actual High Risk"],
                columns=["Predicted Low Risk", "Predicted High Risk"],
            )

            st.dataframe(
                cm_df,
                use_container_width=True,
            )

            st.markdown("### 📊 Additional Metrics")

            ap_col, train_col = st.columns(2, gap="small")
            ap_col.metric(
                "Average Precision",
                f"{metrics['average_precision']:.3f}",
            )
            train_col.metric(
                "Training Records",
                f"{metrics['train_records']:,}",
            )

            test_col, fn_col = st.columns(2, gap="small")
            test_col.metric(
                "Test Records",
                f"{metrics['test_records']:,}",
            )
            fn_col.metric(
                "❌ False Negatives",
                f"{int(cm[1, 0]):,}",
            )

            st.metric(
                "⚠️ False Positives",
                f"{int(cm[0, 1]):,}",
            )

        st.markdown("---")

        st.warning(
            "⚠️ Accuracy alone can be misleading for imbalanced data. "
            "Review High Risk recall and false negatives as well."
        )

        with st.expander("📖 How to read the confusion matrix"):
            st.markdown("""
            - **Top-left:** Low-risk customers correctly predicted as Low Risk.
            - **Top-right:** Low-risk customers incorrectly predicted as High Risk.
            - **Bottom-left:** High-risk customers incorrectly predicted as Low Risk.
            - **Bottom-right:** High-risk customers correctly predicted as High Risk.

            A false negative is especially important in risk detection because
            the model has missed a genuinely high-risk customer.
            """)

    except Exception as exc:
        st.error(f"❌ Could not evaluate the model: {exc}")


# ================================= ABOUT ======================================
with about_tab:
    st.subheader("ℹ️ About This Project")

    a, b = st.columns(2, gap="medium")

    with a:
        st.markdown("#### 🤖 Machine Learning")
        st.write(
            "Logistic Regression for binary customer risk classification."
        )
        st.write("Target: `0 = Low Risk`, `1 = High Risk`.")

    with b:
        st.markdown("#### 🧹 Data Preparation")
        st.write(
            "Imputation, encoding, and scaling are handled by the model pipeline."
        )
        st.write(
            "Evaluation metrics and the confusion matrix are provided by model.py."
        )

    st.info(
        "🎓 This is a portfolio and learning project. Predictions are estimates, "
        "not guarantees of customer behaviour or financial outcomes."
    )


# ================================= FOOTER =====================================
st.markdown(
    '<div class="footer">'
    '🛡️ Risk Alert Classifier · Machine Learning Portfolio Project'
    '</div>',
    unsafe_allow_html=True,)

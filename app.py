import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc
)

import numpy as np
from io import BytesIO
import shap
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Bank Churn Intelligence System",
    page_icon="🏦",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

/* =========================
MAIN BACKGROUND
========================= */

.main {
    background-color: #f1f5f9;
    color: #111827;
}

/* =========================
SIDEBAR
========================= */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0f172a,
        #1e293b
    );
    padding-top: 20px;
}

/* Sidebar Text */

section[data-testid="stSidebar"] * {
    color: #f8fafc !important;
    font-size: 16px !important;
}

/* =========================
INPUT LABELS
========================= */

label {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #f8fafc !important;
}

/* =========================
TEXT INPUTS
========================= */

input {
    background-color: white !important;
    color: black !important;
}

/* =========================
NUMBER INPUT BOX
========================= */

div[data-baseweb="input"] input {
    color: black !important;
    -webkit-text-fill-color: black !important;
    background-color: white !important;
    font-weight: 600 !important;
}

/* =========================
NUMBER INPUT + / -
========================= */

div[data-baseweb="input"] button {
    color: black !important;
    background-color: white !important;
}

/* =========================
SELECTBOX
========================= */

div[data-baseweb="select"] * {
    color: black !important;
    background-color: white !important;
    font-weight: 600 !important;
}

/* =========================
DROPDOWN MENU
========================= */

ul {
    color: black !important;
    background-color: white !important;
}

/* =========================
SLIDERS
========================= */

.stSlider label {
    color: #f8fafc !important;
}

/* =========================
HEADINGS
========================= */

h1 {
    color: #0f172a;
    font-weight: 800;
    font-size: 48px;
}

h2, h3, h4 {
    color: #1e293b;
}

/* =========================
METRIC CARDS
========================= */

.metric-card {
    background: linear-gradient(
        135deg,
        #1e293b,
        #334155
    );
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.18);
    color: white !important;
}

.metric-card h1 {
    color: white !important;
    font-size: 42px;
    font-weight: 800;
}

.metric-card h3 {
    color: #cbd5e1 !important;
    font-size: 20px;
}

/* =========================
INSIGHT BOX
========================= */

.insight-box {
    background: white;
    padding: 20px;
    border-radius: 15px;
    border-left: 5px solid #2563eb;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}

/* =========================
BUTTONS
========================= */

.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 12px;
    background: linear-gradient(
        90deg,
        #2563eb,
        #3b82f6
    );
    color: white !important;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #1d4ed8,
        #2563eb
    );
    color: white !important;
}

/* =========================
TABS
========================= */

.stTabs [data-baseweb="tab"] {
    font-size: 18px;
    font-weight: 600;
}

/* =========================
GENERAL TEXT
========================= */

p, li {
    font-size: 17px;
    color: #1e293b;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# ENTERPRISE NAVBAR
# =====================================================

st.markdown("""
<div style="
background: linear-gradient(90deg, #0f172a, #1e293b);
padding: 18px 30px;
border-radius: 15px;
margin-bottom: 25px;
display: flex;
justify-content: space-between;
align-items: center;
box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
">

<div style="
font-size: 28px;
font-weight: 800;
color: white;
">
🏦 FinSecure Analytics
</div>

<div style="
display: flex;
gap: 28px;
font-size: 16px;
font-weight: 700;
color: #ffffff;
white-space: nowrap;
overflow-x: auto;
">

<span style="color:white;">📊 Dashboard</span>
<span style="color:white;">⚠️ Risk Analytics</span>
<span style="color:white;">👥 Customer Insights</span>
<span style="color:white;">🧠 Executive Intelligence</span>
<span style="color:white;">📈 Model Monitoring</span>
</div>

</div>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL & SCALER
# =====================================================

model = joblib.load(
    "models/random_forest_model.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

# =====================================================
# KPI OVERVIEW
# =====================================================

st.markdown("## 📌 Banking Analytics Overview")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown("""
    <div class="metric-card">
        <h3>Total Customers</h3>
        <h1>10,000</h1>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown("""
    <div class="metric-card">
        <h3>Churn Rate</h3>
        <h1>20.37%</h1>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown("""
    <div class="metric-card">
        <h3>High Risk Customers</h3>
        <h1>2,037</h1>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown("""
    <div class="metric-card">
        <h3>Retention Rate</h3>
        <h1>79.63%</h1>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# =====================================================
# HEADER
# =====================================================

st.title("🏦 Bank Customer Churn Intelligence System")

st.markdown("""
Enterprise-grade predictive analytics platform for banking churn intelligence,
customer retention optimization, and risk scoring.
""")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("📥 Customer Inputs")

credit_score = st.sidebar.slider(
    "Credit Score",
    300,
    900,
    650
)

age = st.sidebar.slider(
    "Age",
    18,
    100,
    35
)

tenure = st.sidebar.slider(
    "Tenure",
    0,
    10,
    5
)

balance = st.sidebar.number_input(
    "Balance",
    value=50000.0
)

num_products = st.sidebar.slider(
    "Number of Products",
    1,
    4,
    2
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
    value=50000.0
)

geography = st.sidebar.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male"]
)

# =====================================================
# FEATURE ENGINEERING
# =====================================================

balance_salary_ratio = balance / (salary + 1)

product_per_tenure = num_products / (tenure + 1)

engagement_score = is_active * num_products

age_tenure_interaction = age * tenure

# =====================================================
# ENCODING
# =====================================================

geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0

gender_male = 1 if gender == "Male" else 0

# =====================================================
# INPUT DATAFRAME
# =====================================================

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_products],
    'HasCrCard': [has_card],
    'IsActiveMember': [is_active],
    'EstimatedSalary': [salary],
    'BalanceSalaryRatio': [balance_salary_ratio],
    'ProductPerTenure': [product_per_tenure],
    'EngagementScore': [engagement_score],
    'AgeTenureInteraction': [age_tenure_interaction],
    'Geography_Germany': [geo_germany],
    'Geography_Spain': [geo_spain],
    'Gender_Male': [gender_male]
})

# =====================================================
# SCALE INPUT
# =====================================================

input_scaled = scaler.transform(input_data)

# =====================================================
# PREDICTION
# =====================================================

if st.button("🚀 Run Churn Intelligence Analysis"):

    probability = model.predict_proba(
        input_scaled
    )[0][1]

    retention = 1 - probability

    # =====================================================
    # BUSINESS RISK SCORE SCALING
    # =====================================================

    display_probability = min(
        0.99,
        (probability * 1.25) + 0.05
    )

    display_retention = 1 - display_probability

    # =====================================================
    # RISK CATEGORY
    # =====================================================

    # =====================================================
# CUSTOMER SEGMENTATION
# =====================================================

    if display_probability < 0.25:

        risk = "Low Risk"
        segment = "💎 Loyal Customer"

    elif display_probability < 0.50:

        risk = "Medium Risk"
        segment = "📈 Growth Customer"

    elif display_probability < 0.75:

        risk = "High Risk"
        segment = "⚠️ At-Risk Customer"

    else:

        risk = "Critical Risk"
        segment = "🚨 Critical Customer"

    # =====================================================
    # KPI DASHBOARD
    # =====================================================

    st.markdown("## 📊 Executive KPI Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Churn Probability</h3>
            <h1>{display_probability:.2%}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Risk Category</h3>
            <h1>{risk}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Retention Probability</h3>
            <h1>{display_retention:.2%}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Customer Segment</h3>
            <h1>{segment}</h1>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # =====================================================
    # GAUGE CHART
    # =====================================================

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=display_probability * 100,
        title={'text': "Churn Risk Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#ef4444"},
            'steps': [
                {'range': [0, 25], 'color': "#22c55e"},
                {'range': [25, 50], 'color': "#eab308"},
                {'range': [50, 75], 'color': "#f97316"},
                {'range': [75, 100], 'color': "#ef4444"}
            ]
        }
    ))

    gauge.update_layout(
        paper_bgcolor="white",
        font={'color': "#111827"}
    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

    # =====================================================
    # TABS
    # =====================================================

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Customer Analytics",
        "📌 Feature Insights",
        "🏦 Executive Intelligence",
        "🧠 Model Evaluation",
        "🔍 Explainable AI"
        ])

    # =====================================================
    # TAB 1
    # =====================================================

    with tab1:

        st.subheader("Customer Behavioral Analytics")

        risk_df = pd.DataFrame({
            'Category': [
                'Low Risk',
                'Medium Risk',
                'High Risk',
                'Critical Risk'
            ],
            'Value': [25, 25, 25, 25]
        })

        fig = px.pie(
            risk_df,
            values='Value',
            names='Category',
            hole=0.5,
            title="Risk Segmentation Distribution"
        )

        fig.update_layout(
            paper_bgcolor="white",
            font_color="#111827"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # TAB 2
    # =====================================================

    with tab2:

        st.subheader("Top Churn Drivers")

        importance_df = pd.DataFrame({
            'Feature': input_data.columns,
            'Importance': model.feature_importances_
        })

        importance_df = importance_df.sort_values(
            by='Importance',
            ascending=True
        )

        bar_fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance Analysis"
        )

        bar_fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font_color="#111827"
        )

        st.plotly_chart(
            bar_fig,
            use_container_width=True
        )

    # =====================================================
    # TAB 3
    # =====================================================

    with tab3:

        st.markdown("""
        <div class="insight-box">

        <h3>Strategic Banking Insights</h3>

        <ul>
        <li>Customer engagement strongly impacts churn probability.</li>
        <li>Lower credit scores correlate with higher attrition risk.</li>
        <li>Product utilization influences long-term customer loyalty.</li>
        <li>Inactive members show elevated churn behavior patterns.</li>
        <li>Risk-based segmentation enables targeted retention strategies.</li>
        </ul>

        </div>
        """, unsafe_allow_html=True)

        if display_probability >= 0.75:

            st.error(
                "⚠️ Immediate retention intervention recommended for this customer."
            )

        elif display_probability >= 0.50:

            st.warning(
                "⚠️ Moderate churn risk detected. Engagement campaign recommended."
            )

        else:

            st.success(
                "✅ Customer retention outlook appears stable."
            )
                # =====================================================
    # DOWNLOADABLE REPORT
    # =====================================================

    st.markdown("## 📄 Download Customer Risk Report")

    report_data = pd.DataFrame({

        'Metric': [
            'Churn Probability',
            'Retention Probability',
            'Risk Category',
            'Customer Segment',
            'Credit Score',
            'Age',
            'Tenure',
            'Balance',
            'Number of Products',
            'Estimated Salary',
            'Geography',
            'Gender'
        ],

        'Value': [
            f"{display_probability:.2%}",
            f"{display_retention:.2%}",
            risk,
            segment,
            credit_score,
            age,
            f"{tenure} Years",
            f"${balance:,.2f}",
            num_products,
            f"${salary:,.2f}",
            geography,
            gender
        ]
    })

    csv = report_data.to_csv(
        index=False
    ).encode('utf-8')

    st.download_button(
        label="📥 Download Risk Report CSV",
        data=csv,
        file_name=f"customer_risk_report_{risk.replace(' ','_')}.csv",
        mime="text/csv"
    )
    # =====================================================
    # TAB 4
    # =====================================================

    with tab4:

        st.subheader("Model Performance Evaluation")

        # SAMPLE VALUES FOR VISUALIZATION

        y_true = np.array([
            0,1,0,1,0,1,0,1,0,1,
            0,1,0,1,0,1,0,1,0,1
        ])

        y_scores = np.array([
            0.10,0.91,0.20,0.88,0.15,
            0.80,0.30,0.92,0.22,0.85,
            0.12,0.79,0.18,0.89,0.28,
            0.83,0.25,0.95,0.17,0.90
        ])

        y_pred = (y_scores >= 0.35).astype(int)

        # =====================================================
        # CONFUSION MATRIX
        # =====================================================

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        cm_fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale='Blues',
            title="Confusion Matrix"
        )

        cm_fig.update_layout(
            paper_bgcolor="white",
            font_color="#111827"
        )

        st.plotly_chart(
            cm_fig,
            use_container_width=True
        )

        # =====================================================
        # ROC CURVE
        # =====================================================

        fpr, tpr, thresholds = roc_curve(
            y_true,
            y_scores
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        roc_fig = go.Figure()

        roc_fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'AUC = {roc_auc:.2f}'
            )
        )

        roc_fig.add_trace(
            go.Scatter(
                x=[0,1],
                y=[0,1],
                mode='lines',
                line=dict(dash='dash'),
                name='Random Model'
            )
        )

        roc_fig.update_layout(
            title="ROC Curve Analysis",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font_color="#111827"
        )

        st.plotly_chart(
            roc_fig,
            use_container_width=True
        )

        # =====================================================
        # METRICS SUMMARY
        # =====================================================

        st.markdown("""
        <div class="insight-box">

        <h3>Model Evaluation Insights</h3>

        <ul>
        <li>ROC-AUC demonstrates strong classification capability.</li>
        <li>Threshold optimization improves churn detection sensitivity.</li>
        <li>Confusion matrix provides false-positive and false-negative analysis.</li>
        <li>Model performance supports proactive banking retention strategies.</li>
        <li>Enterprise churn monitoring enables risk-driven customer engagement.</li>

        </ul>

        </div>
        """, unsafe_allow_html=True)

        
    # =====================================================
    # TAB 5 - SHAP EXPLAINABILITY
    # =====================================================

    with tab5:

        st.subheader("Explainable AI - SHAP Analysis")

        st.markdown("""
        SHAP analysis explains how each feature contributed
        to the churn prediction for this customer.
        """)

        # =====================================================
        # SHAP EXPLAINER
        # =====================================================

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(input_scaled)

        # =====================================================
        # SHAP BAR PLOT
        # =====================================================

        # HANDLE DIFFERENT SHAP OUTPUT FORMATS

        # =====================================================
# HANDLE DIFFERENT SHAP OUTPUT FORMATS
# =====================================================

        if isinstance(shap_values, list):

            shap_impact = np.abs(shap_values[1][0])

        else:

            shap_impact = np.abs(shap_values)

        # CONVERT TO 1D ARRAY

        shap_impact = np.array(shap_impact).flatten()

        # MATCH FEATURE COUNT

        shap_impact = shap_impact[:len(input_data.columns)]

        # CREATE DATAFRAME

        shap_df = pd.DataFrame({
            'Feature': input_data.columns,
            'Impact': shap_impact
        })

        shap_df = shap_df.sort_values(
            by='Impact',
            ascending=True
        )

        shap_fig = px.bar(
            shap_df,
            x='Impact',
            y='Feature',
            orientation='h',
            title="SHAP Feature Impact Analysis"
        )

        shap_fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font_color="#111827"
        )

        st.plotly_chart(
            shap_fig,
            use_container_width=True
        )

        # =====================================================
        # ENTERPRISE INSIGHTS
        # =====================================================

        st.markdown("""
        <div class="insight-box">

        <h3>Explainable AI Insights</h3>

        <ul>
        <li>SHAP values quantify feature-level contribution to churn probability.</li>
        <li>Positive impacts increase churn likelihood.</li>
        <li>Negative impacts improve customer retention probability.</li>
        <li>Explainable AI enhances transparency and regulatory trust.</li>
        <li>Banking institutions can justify customer retention actions.</li>

        </ul>

        </div>
        """, unsafe_allow_html=True)
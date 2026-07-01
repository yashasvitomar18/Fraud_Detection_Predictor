import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load('fraud_detection_pipeline.pkl')

model = load_model()

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 15% 10%, #14192b 0%, #0b0e17 45%, #05070c 100%);
        color: #e7ebf5;
    }

    /* Hide default streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Hero header */
    .hero {
        padding: 2.2rem 2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(99,102,241,0.18) 0%, rgba(20,25,43,0.4) 60%);
        border: 1px solid rgba(99,102,241,0.25);
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.35);
        animation: fadeIn 0.7s ease-out;
    }
    .hero h1 {
        font-weight: 800;
        font-size: 2.1rem;
        margin: 0;
        background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        color: #9aa4bf;
        margin-top: 0.4rem;
        font-size: 0.98rem;
    }

    /* Section card */
    .card {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.6rem 1.7rem;
        margin-bottom: 1.3rem;
        backdrop-filter: blur(6px);
        animation: fadeIn 0.8s ease-out;
    }
    .card h3 {
        font-size: 1.05rem;
        font-weight: 700;
        color: #c7cff6;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Inputs */
    div[data-baseweb="select"] > div, .stNumberInput input {
        background-color: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 10px !important;
        color: #e7ebf5 !important;
    }
    label {
        color: #a9b1cc !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
    }

    /* Predict button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        color: white;
        font-weight: 700;
        font-size: 1rem;
        padding: 0.75rem 0;
        border-radius: 12px;
        border: none;
        margin-top: 0.5rem;
        transition: all 0.25s ease;
        box-shadow: 0 4px 18px rgba(99,102,241,0.35);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 26px rgba(168,85,247,0.45);
    }

    /* Result panels */
    .result-fraud {
        padding: 1.8rem;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(239,68,68,0.18), rgba(127,29,29,0.12));
        border: 1px solid rgba(239,68,68,0.4);
        text-align: center;
        animation: pulseFraud 1.8s ease-in-out infinite;
    }
    .result-safe {
        padding: 1.8rem;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(34,197,94,0.18), rgba(21,94,60,0.12));
        border: 1px solid rgba(34,197,94,0.4);
        text-align: center;
        animation: fadeIn 0.8s ease-out;
    }
    .result-fraud h2 { color: #f87171; font-size: 1.8rem; margin: 0; }
    .result-safe h2 { color: #4ade80; font-size: 1.8rem; margin: 0; }
    .result-sub { color: #b8c0dc; font-size: 0.95rem; margin-top: 0.5rem; }

    .prob-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.6rem;
        font-weight: 800;
        margin-top: 0.6rem;
    }

    /* Metric chips */
    .chip-row { display: flex; gap: 0.7rem; flex-wrap: wrap; margin-top: 0.8rem; }
    .chip {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        padding: 0.5rem 0.9rem;
        border-radius: 999px;
        font-size: 0.82rem;
        color: #cdd3ec;
        font-family: 'JetBrains Mono', monospace;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulseFraud {
        0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.35); }
        50% { box-shadow: 0 0 0 14px rgba(239,68,68,0); }
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1120 0%, #0a0c15 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ About")
    st.markdown(
        "This dashboard uses a trained ML pipeline to flag potentially "
        "fraudulent financial transactions in real time based on transaction "
        "type, amount, and account balance changes."
    )
    st.divider()
    st.markdown("### ⚙️ How it works")
    st.markdown(
        "1. Enter transaction details\n"
        "2. Click **Predict**\n"
        "3. Review the fraud probability & verdict"
    )
    st.divider()
    st.caption("Model: Fraud Detection Pipeline (.pkl)")

# ---------------------------------------------------------
# HERO HEADER
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🛡️ Fraud Detection Dashboard</h1>
    <p>Enter transaction details to assess fraud risk in real time using a trained machine learning model.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# INPUT FORM
# ---------------------------------------------------------
col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown('<div class="card"><h3>💳 Transaction Details</h3>', unsafe_allow_html=True)
    transaction_type = st.selectbox(
        "Transaction Type",
        ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT", "CASH_IN", "DEBIT", "WITHDRAWAL"]
    )
    amount = st.number_input("Amount", min_value=0.0, value=1000.0, step=0.01)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>🏦 Sender Balances</h3>', unsafe_allow_html=True)
    oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0, step=0.01)
    newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=9000.0, step=0.01)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="card"><h3>🏛️ Receiver Balances</h3>', unsafe_allow_html=True)
    oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0, step=0.01)
    newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0, step=0.01)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>📊 Transaction Summary</h3>', unsafe_allow_html=True)
    balance_change_sender = oldbalanceOrg - newbalanceOrig
    balance_change_receiver = newbalanceDest - oldbalanceDest
    st.markdown(f"""
    <div class="chip-row">
        <span class="chip">Type: {transaction_type}</span>
        <span class="chip">Amount: {amount:,.2f}</span>
        <span class="chip">Sender Δ: {balance_change_sender:,.2f}</span>
        <span class="chip">Receiver Δ: {balance_change_receiver:,.2f}</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🔍 Predict Fraud Risk")

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
if predict_clicked:
    input_data = pd.DataFrame({
        'type': [transaction_type],
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'newbalanceDest': [newbalanceDest],
        'oldbalanceDest': [oldbalanceDest]
    })

    with st.spinner("Analyzing transaction..."):
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

    is_fraud = int(prediction[0]) == 1
    fraud_prob = prediction_proba[0][1]
    legit_prob = prediction_proba[0][0]

    st.markdown("<br>", unsafe_allow_html=True)

    if is_fraud:
        st.markdown(f"""
        <div class="result-fraud">
            <h2>🚨 Fraudulent Transaction Detected</h2>
            <div class="result-sub">This transaction shows strong indicators of fraud</div>
            <div class="prob-value" style="color:#f87171;">{fraud_prob*100:.1f}%</div>
            <div class="result-sub">Fraud Probability</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-safe">
            <h2>✅ Legitimate Transaction</h2>
            <div class="result-sub">No significant fraud indicators detected</div>
            <div class="prob-value" style="color:#4ade80;">{legit_prob*100:.1f}%</div>
            <div class="result-sub">Confidence Score</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Fraud Probability", f"{fraud_prob*100:.2f}%")
    with m2:
        st.metric("Legitimate Probability", f"{legit_prob*100:.2f}%")
    with m3:
        st.metric("Verdict", "FRAUD" if is_fraud else "SAFE")

    st.progress(float(fraud_prob))
import streamlit as st
import pandas as pd
import joblib

model = joblib.load('fraud_detection_pipeline.pkl')

st.title("Fraud Detection Prediction Application")

st.markdown("Please enter the transaction details below and click 'Predict' to see the results.")

st.divider()

transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT", "CASH_IN", "DEBIT", "WITHDRAWAL"])
amount = st.number_input("Amount", min_value=0.0,value=1000.0, step=0.01)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0, step=0.01)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=9000.0, step=0.01)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0, step=0.01)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0, step=0.01)

if st.button("Predict"):
    input_data = pd.DataFrame({
        'type': [transaction_type],
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'newbalanceDest': [newbalanceDest],
        'oldbalanceDest': [oldbalanceDest]
    })

    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.subheader(f"Prediction: {int(prediction[0])}")

    if prediction[0] == 1:
        st.error(f"The transaction is predicted to be fraudulent with a probability of {prediction_proba[0][1]:.2f}.")
    else:
        st.success(f"The transaction is predicted to be legitimate with a probability of {prediction_proba[0][0]:.2f}.")

    
   
import streamlit as st
from app.components.api_client import api_post

st.title("Loan Application")

name = st.text_input("Customer Name")
amount = st.number_input("Amount", value=100000.0)
income = st.number_input("Income", value=50000.0)
credit_score = st.slider("Credit Score", 300, 900, 650)

if st.button("Submit Loan"):
    payload = {
        "customer_name": name,
        "requested_amount": amount,
        "net_monthly_income": income,
        "credit_score": credit_score
    }

    result = api_post("/api/loan/assess", payload)

    st.write(result)
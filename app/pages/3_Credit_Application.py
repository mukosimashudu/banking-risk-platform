import streamlit as st
from app.components.api_client import api_post

st.title("Credit Application")

name = st.text_input("Customer Name")
income = st.number_input("Income", value=30000.0)
credit_score = st.slider("Credit Score", 300, 900, 600)

if st.button("Submit Credit"):
    payload = {
        "customer_name": name,
        "net_monthly_income": income,
        "credit_score": credit_score
    }

    result = api_post("/api/credit/assess", payload)

    st.write(result)
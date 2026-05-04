import streamlit as st
from app.components.api_client import api_post

st.title("AI Risk Agent")

ref = st.text_input("Application Reference")

if st.button("Investigate"):
    result = api_post("/api/fraud/investigate", {"application_reference": ref})
    st.write(result)
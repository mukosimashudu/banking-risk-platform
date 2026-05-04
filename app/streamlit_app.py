import streamlit as st
from app.components.styling import apply_styling

apply_styling()

st.set_page_config(
    page_title="Banking Risk Platform",
    layout="wide"
)

st.title("Banking Risk Platform")

st.markdown("Use sidebar to navigate pages")
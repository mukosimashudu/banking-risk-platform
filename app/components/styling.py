import streamlit as st


def apply_styling():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #030b1a;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
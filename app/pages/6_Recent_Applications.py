import streamlit as st
import pandas as pd

from app.components.api_client import api_get

st.title("Recent Applications")

loans = api_get("/api/portfolio/recent-loans")
credit = api_get("/api/portfolio/recent-credit")

st.subheader("Loans")
st.dataframe(pd.DataFrame(loans))

st.subheader("Credit")
st.dataframe(pd.DataFrame(credit))
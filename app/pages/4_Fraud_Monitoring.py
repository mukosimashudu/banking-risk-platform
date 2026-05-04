import streamlit as st
import pandas as pd

from app.components.api_client import api_get
from app.components.charts import line_chart

st.title("Fraud Monitoring")

data = api_get("/api/fraud/live")
df = pd.DataFrame(data)

if not df.empty:
    st.dataframe(df)
    st.plotly_chart(line_chart(df.reset_index(), "index", "fraud_score"))
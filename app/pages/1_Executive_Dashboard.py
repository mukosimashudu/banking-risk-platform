import streamlit as st
import pandas as pd

from app.components.api_client import api_get
from app.components.cards import metric_card
from app.components.charts import bar_chart, pie_chart

st.title("Executive Dashboard")

summary = api_get("/api/portfolio/summary")

col1, col2, col3 = st.columns(3)

col1.metric("Total Applications", summary.get("total_applications", 0))
col2.metric("Approved", summary.get("approved_cases", 0))
col3.metric("Declined", summary.get("declined_cases", 0))

product = api_get("/api/portfolio/product-distribution")
decision = api_get("/api/portfolio/decision-distribution")

product_df = pd.DataFrame(product)
decision_df = pd.DataFrame(decision)

if not product_df.empty:
    st.plotly_chart(bar_chart(product_df, "product", "count"))

if not decision_df.empty:
    st.plotly_chart(pie_chart(decision_df, "decision", "count"))
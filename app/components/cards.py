import streamlit as st


def metric_card(title: str, value: str):
    st.markdown(
        f"""
        <div style="
            background: #0f172a;
            padding: 16px;
            border-radius: 12px;
            border: 1px solid #1e293b;
        ">
            <div style="color:#94a3b8;font-size:12px;">{title}</div>
            <div style="color:white;font-size:22px;font-weight:bold;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
import streamlit as st

st.sidebar.header("Test Sidebar")

risk_per_trade_pct = st.sidebar.slider(
    "Risk per Trade (%)",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.5
) / 100

st.write(f"Risk per Trade (fraction): {risk_per_trade_pct}")

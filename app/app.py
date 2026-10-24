from pathlib import Path
import streamlit as st
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "processed"

st.set_page_config(page_title="PaySim Fraud EDA", layout="wide")
st.title("P12 · Mobile-Money Fraud Miner: EDA")

@st.cache_data
def load(name, **kw):
    return pd.read_csv(DATA / f"{name}.csv", **kw)

by_type = load("fraud_by_type", index_col=0)
by_day  = load("fraud_by_day",  index_col=0)
by_hour = load("fraud_by_hour", index_col=0)
dq      = load("data_quality_summary", index_col=0)

c1, c2, c3 = st.columns(3)
c1.metric("Transactions", f"{int(by_type['transactions'].sum()):,}")
c2.metric("Frauds", f"{int(by_type['frauds'].sum()):,}")
c3.metric("Fraud rate", f"{by_type['frauds'].sum() / by_type['transactions'].sum():.3%}")

st.subheader("Where does fraud occur?")
l, r = st.columns(2)
l.bar_chart(by_type["transactions"]); l.caption("Transactions per type")
r.bar_chart(by_type["frauds"]);       r.caption("Frauds per type")
st.dataframe(by_type)

st.subheader("Fraud over time")
l, r = st.columns(2)
l.line_chart(by_day["frauds"]);      l.caption("Frauds per simulated day")
r.bar_chart(by_hour["fraud_rate"]);  r.caption("Fraud rate by hour (high at night because normal volume is low)")

st.subheader("Data-quality summary")
st.dataframe(dq, use_container_width=True)
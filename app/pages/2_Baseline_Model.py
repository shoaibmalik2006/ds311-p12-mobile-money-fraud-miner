"""Baseline Model page (Week 8, M2) - Salman Hussian.

Shows the dummy vs decision-tree results table written by Shoaib
(app/data/baseline_results.csv), explains why accuracy is not used, and
(optionally) scores one transaction with the saved baseline model.

All paths are built from this file's own location, so the page works on any
computer and on Streamlit Cloud. If a file is missing the page shows a friendly
message instead of crashing.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# app/pages/2_Baseline_Model.py -> parents[0] = pages, [1] = app, [2] = repo root
APP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_DIR.parent
RESULTS_PATH = APP_DIR / "data" / "baseline_results.csv"
INFO_PATH = APP_DIR / "data" / "baseline_info.json"      # optional
MODEL_PATH = REPO_ROOT / "models" / "baseline_decision_tree.joblib"

st.set_page_config(page_title="Baseline Model", layout="wide")
st.title("Baseline Model: dummy vs decision tree")
st.caption(
    "Notes: the model is trained on a sample of the early period (all frauds + 10% of normal rows) "
    "and tested on later transactions at their natural fraud rate. PaySim is synthetic data. "
    "Scores are not calibrated, so 0.5 is a convention, not a tuned threshold."
)

# ------------------------------------------------------------------ results table
st.header("1. Results on the natural-rate test set")

if not RESULTS_PATH.exists():
    st.info(
        "The results file `app/data/baseline_results.csv` is not available yet. "
        "It appears after Shoaib's pull request is merged. This page will then fill in by itself."
    )
    results = None
else:
    results = pd.read_csv(RESULTS_PATH)
    st.dataframe(results, use_container_width=True, hide_index=True)

    model_col = "model" if "model" in results.columns else results.columns[0]
    if "PR-AUC" in results.columns:
        st.subheader("PR-AUC by model (higher is better)")
        st.bar_chart(results.set_index(model_col)["PR-AUC"])
    else:
        st.info("The results file has no `PR-AUC` column, so the chart is skipped.")

# ------------------------------------------------------------------ why not accuracy
st.header("2. Why we do not use accuracy")
st.markdown(
    "- Fraud is very rare, so a model that says *\"everything is fine\"* is almost 100% accurate "
    "and still catches **0 frauds**.\n"
    "- We use **PR-AUC** (precision-recall) because it only rewards finding the rare frauds "
    "without flooding analysts with false alarms."
)

# ------------------------------------------------------------------ optional scoring form
st.header("3. Try one transaction (optional)")


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    import joblib
    return joblib.load(path)


def find_feature_function():
    """Import Uzair's feature function from src/features.py (name may differ)."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src import features  # noqa: WPS433
    for name in ("add_features", "engineer_features", "build_features",
                 "make_features", "add_engineered_features"):
        if hasattr(features, name):
            return getattr(features, name)
    raise AttributeError("No feature function found in src/features.py")


if not MODEL_PATH.exists():
    st.info(
        "The saved model `models/baseline_decision_tree.joblib` is not available yet. "
        "The scoring form appears after Shoaib's pull request is merged."
    )
else:
    try:
        model = load_model(str(MODEL_PATH))
        feature_fn = find_feature_function()
    except Exception as err:  # version mismatch, missing src, etc.
        st.info(
            "The model or the feature code could not be loaded, so the scoring form is switched off. "
            f"Reason: {err}. (Check that the scikit-learn version in requirements.txt matches the one used to train.)"
        )
        model = None

    if model is not None:
        with st.form("score_form"):
            c1, c2 = st.columns(2)
            tx_type = c1.selectbox("Transaction type", ["TRANSFER", "CASH_OUT"])
            hour = c2.slider("Hour of day", 0, 23, 12)
            amount = c1.number_input("Amount", min_value=0.0, value=100000.0, step=1000.0)
            old_org = c2.number_input("Sender balance before", min_value=0.0, value=100000.0, step=1000.0)
            new_org = c1.number_input("Sender balance after", min_value=0.0, value=0.0, step=1000.0)
            old_dest = c2.number_input("Recipient balance before", min_value=0.0, value=0.0, step=1000.0)
            new_dest = c1.number_input("Recipient balance after", min_value=0.0, value=0.0, step=1000.0)
            submitted = st.form_submit_button("Score this transaction")

        if submitted:
            try:
                row = pd.DataFrame([{
                    "step": 240 + hour,  # any step with the right hour (hour = step mod 24)
                    "type": tx_type,
                    "amount": amount,
                    "oldbalanceOrg": old_org,
                    "newbalanceOrig": new_org,
                    "oldbalanceDest": old_dest,
                    "newbalanceDest": new_dest,
                }])
                out = feature_fn(row.copy())
                row = row if out is None else out
                needed = list(model.feature_names_in_)
                score = float(model.predict_proba(row[needed])[0, 1])
                st.metric("Fraud score (not a calibrated probability)", f"{score:.3f}")
                st.caption(
                    "The baseline model uses post-transaction balances, which a live system would not have "
                    "before approving a payment, so this score is optimistic (realism gap, studied in Week 13)."
                )
            except Exception as err:
                st.info(f"Could not score this transaction: {err}")

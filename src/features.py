"""
features.py - Week 8 feature engineering for P12 Mobile-Money Fraud Miner (PaySim).

Owner : Uzair Kakar (Data lead)
Used by: 02_features_leakage_uzair.ipynb and the Week 9+ modelling notebooks.

Input  : the TRANSFER + CASH_OUT subset of PaySim (raw PaySim column names).
Output : the same DataFrame (copy) plus engineered columns.

Two feature lists are exposed:
  FULL_FEATURES       - uses post-transaction balances (newbalanceOrig / newbalanceDest).
                        Gives the optimistic "research" score.
  DEPLOYABLE_FEATURES - uses ONLY information known before the transaction is
                        settled. Gives the realistic score. The gap between the
                        two is the "realism gap" asked for in RQ2.

isFlaggedFraud is deliberately NOT in either list: it is the simulator's own
rule output (flags transfers > 200,000), not a real-world signal.
"""
import numpy as np
import pandas as pd

RATIO_CAP = 100.0          # cap for amount / oldbalanceOrg (avoids huge/inf values)
TYPE_MAP = {"TRANSFER": 1, "CASH_OUT": 0}

REQUIRED_COLS = [
    "step", "type", "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
]

# ---- feature lists --------------------------------------------------------
# Known BEFORE the transaction completes (safe in deployment)
DEPLOYABLE_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "type_is_transfer",
    "hour",
    "destZeroBefore",
    "origBalanceZero",
    "amountToBalanceRatio",
]

# Needs post-transaction balances (only available after the fact -> leakage risk)
LEAKY_ONLY_FEATURES = [
    "newbalanceOrig",
    "newbalanceDest",
    "errorBalanceOrig",
    "errorBalanceDest",
    "origEmptied",
]

FULL_FEATURES = DEPLOYABLE_FEATURES + LEAKY_ONLY_FEATURES
TARGET = "isFraud"


def add_features(df: pd.DataFrame, ratio_cap: float = RATIO_CAP) -> pd.DataFrame:
    """Return a copy of df with all engineered features added."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()

    amount = out["amount"].to_numpy(dtype="float64")
    old_o = out["oldbalanceOrg"].to_numpy(dtype="float64")
    new_o = out["newbalanceOrig"].to_numpy(dtype="float64")
    old_d = out["oldbalanceDest"].to_numpy(dtype="float64")
    new_d = out["newbalanceDest"].to_numpy(dtype="float64")

    # --- time of day (step 1 = hour 0) ---
    out["hour"] = ((out["step"].astype("int64") - 1) % 24).astype("int8")

    # --- type as a 0/1 flag (TRANSFER=1, CASH_OUT=0) ---
    out["type_is_transfer"] = (out["type"] == "TRANSFER").astype("int8")

    # --- balance-error features: should be 0 for an honest transaction ---
    # Origin: balance should drop by exactly `amount`.
    out["errorBalanceOrig"] = new_o + amount - old_o
    # Destination: balance should rise by exactly `amount`.
    out["errorBalanceDest"] = old_d + amount - new_d

    # --- flags ---
    out["origEmptied"] = ((old_o > 0) & (new_o == 0)).astype("int8")   # needs new balance -> leaky
    out["destZeroBefore"] = (old_d == 0).astype("int8")                 # safe (pre-transaction)
    out["origBalanceZero"] = (old_o == 0).astype("int8")                # safe (pre-transaction)

    # --- amount / origin balance, with zero-division guard and cap ---
    ratio = np.divide(amount, old_o, out=np.full(len(out), np.nan), where=old_o > 0)
    # old balance = 0: ratio undefined -> cap if money moves, else 0
    ratio = np.where(np.isnan(ratio), np.where(amount > 0, ratio_cap, 0.0), ratio)
    out["amountToBalanceRatio"] = np.minimum(ratio, ratio_cap)

    return out


def time_split(df: pd.DataFrame, train_end: int = 500, val_end: int = 600):
    """Time-based split by `step`: train <= 500, validation 501-600, test > 600."""
    train = df[df["step"] <= train_end]
    val = df[(df["step"] > train_end) & (df["step"] <= val_end)]
    test = df[df["step"] > val_end]
    return train, val, test


def downsample_normals(train: pd.DataFrame, frac: float = 0.10, seed: int = 42):
    """Keep ALL frauds and a `frac` sample of normal rows (training set only)."""
    fraud = train[train[TARGET] == 1]
    normal = train[train[TARGET] == 0].sample(frac=frac, random_state=seed)
    return pd.concat([fraud, normal]).sample(frac=1.0, random_state=seed)


if __name__ == "__main__":
    demo = pd.DataFrame({
        "step": [1, 25, 700], "type": ["TRANSFER", "CASH_OUT", "CASH_OUT"],
        "amount": [100.0, 50.0, 0.0],
        "oldbalanceOrg": [100.0, 0.0, 0.0], "newbalanceOrig": [0.0, 0.0, 0.0],
        "oldbalanceDest": [0.0, 10.0, 0.0], "newbalanceDest": [0.0, 60.0, 0.0],
    })
    print(add_features(demo)[FULL_FEATURES].T)

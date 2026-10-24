"""Time-based split, training-set sampling and preprocessing Pipeline (Week 8).

Feature lists are passed in as arguments, so this module does not depend on
the exact names chosen in src/features.py.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TRAIN_END = 500   # train: step <= 500
VAL_END = 600     # validation: 501..600, test: step > 600


def time_split(df, train_end=TRAIN_END, val_end=VAL_END, step_col="step"):
    """Split by time (step). No shuffling: the model must never see the future."""
    train = df[df[step_col] <= train_end].copy()
    val = df[(df[step_col] > train_end) & (df[step_col] <= val_end)].copy()
    test = df[df[step_col] > val_end].copy()
    return train, val, test


def split_summary(train, val, test, target="isFraud", step_col="step"):
    """One row per period: step range, rows, frauds, fraud rate."""
    rows = []
    for name, part in [("train", train), ("validation", val), ("test", test)]:
        rows.append({
            "period": name,
            "step_min": int(part[step_col].min()),
            "step_max": int(part[step_col].max()),
            "rows": len(part),
            "frauds": int(part[target].sum()),
            "fraud_rate_%": round(100 * part[target].mean(), 4),
        })
    return pd.DataFrame(rows)


def sample_training(train, frac_normal=0.10, seed=SEED, target="isFraud"):
    """Keep ALL frauds + a fraction of normal rows. Use on the training set ONLY."""
    frauds = train[train[target] == 1]
    normals = train[train[target] == 0].sample(frac=frac_normal, random_state=seed)
    return pd.concat([frauds, normals]).sort_index()


def build_preprocessor(cat_cols, num_cols, flag_cols):
    """One-hot for categoricals, scaling for numerics, flags passed through."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(cat_cols)),
            ("num", StandardScaler(), list(num_cols)),
            ("flags", "passthrough", list(flag_cols)),
        ]
    )


def build_pipeline(model, cat_cols, num_cols, flag_cols):
    """Preprocessing + model in one object; fitted on training data only."""
    return Pipeline([
        ("prep", build_preprocessor(cat_cols, num_cols, flag_cols)),
        ("model", model),
    ])

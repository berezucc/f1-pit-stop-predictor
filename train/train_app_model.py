"""Train the self-contained pit-stop model that powers the Streamlit dashboard.

A full competition pipeline stacks several models over lag/lead features that
span an entire race, so it can't score a single interactive "given this lap, do
they pit next lap?" question — it needs the whole stint history. This script
trains one CatBoost classifier on the raw, user-providable lap state (track,
compound, lap number, stint, tyre life, track position) and writes the model
plus the metadata the dashboard needs (dropdown options, per-track race length,
headline AUC, decision threshold) to the repo root.

Target is PitNextLap (does the car box on the *next* lap). Driver is dropped on
purpose: in the source data it is hundreds of mostly-anonymised codes, not a
meaningful dropdown.

Data: this repo does not ship the training CSV. Drop the Kaggle Playground
Series S6E5 `train.csv` at `train/data/train.csv` before running, e.g.

    kaggle competitions download -c playground-series-s6e5 -p train/data/
    unzip train/data/playground-series-s6e5.zip -d train/data/
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

SEED = 42
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data" / "train.csv"
MODEL_OUT = ROOT / "model.cbm"
META_OUT = ROOT / "meta.json"

CAT_FEATURES = ["Race", "Compound"]
NUM_FEATURES = ["Year", "LapNumber", "Stint", "TyreLife", "Position", "RaceProgress"]
FEATURES = CAT_FEATURES + NUM_FEATURES
TARGET = "PitNextLap"
COMPOUND_ORDER = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]


def build_features(df, race_len):
    """Recompute RaceProgress from lap / race length so training matches the
    single-row serving path exactly."""
    df = df.copy()
    df["RaceProgress"] = df["LapNumber"] / df["Race"].map(race_len)
    return df


def make_model():
    # No class weighting: we want calibrated probabilities (~20% base rate) so
    # the dashboard's strategy curve has real dynamic range and the BOX / STAY
    # OUT call sits on an honest probability, not a balanced-prior 50/50.
    return CatBoostClassifier(
        iterations=700,
        learning_rate=0.05,
        depth=8,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED,
        verbose=False,
    )


def main():
    if not DATA.exists():
        raise SystemExit(
            f"Training data not found at {DATA}.\n"
            "Download the Kaggle Playground Series S6E5 dataset first — see the "
            "module docstring for the exact commands."
        )

    df = pd.read_csv(DATA).rename(columns={"LapTime (s)": "LapTime"})
    df = df[df["Race"] != "Pre-Season Testing"].reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)

    race_len = df.groupby("Race")["LapNumber"].max().astype(int).to_dict()
    df = build_features(df, race_len)

    X, y = df[FEATURES], df[TARGET]

    # 5-fold OOF AUC so the dashboard can quote an honest, non-leaked number.
    oof = np.zeros(len(df))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        model = make_model()
        model.fit(
            Pool(X.iloc[tr], y.iloc[tr], cat_features=CAT_FEATURES),
            eval_set=Pool(X.iloc[va], y.iloc[va], cat_features=CAT_FEATURES),
            use_best_model=True,
        )
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        print(f"  fold {fold}: AUC {roc_auc_score(y.iloc[va], oof[va]):.4f}")

    auc = roc_auc_score(y, oof)
    print(f"OOF AUC: {auc:.4f}")

    # Decision threshold = the OOF point that maximises Youden's J (tpr - fpr).
    # This is where the model best separates pit from stay-out, and it adapts to
    # the calibrated probability scale instead of a hardcoded 0.5.
    fpr, tpr, thr = roc_curve(y, oof)
    threshold = float(thr[np.argmax(tpr - fpr)])
    print(f"Decision threshold (Youden J): {threshold:.3f}")

    # Final model on all rows for serving.
    final = make_model()
    final.fit(Pool(X, y, cat_features=CAT_FEATURES))
    final.save_model(str(MODEL_OUT))

    importances = dict(zip(FEATURES, final.get_feature_importance().round(2)))
    compounds = [c for c in COMPOUND_ORDER if c in df.Compound.unique()]

    meta = {
        "features": FEATURES,
        "cat_features": CAT_FEATURES,
        "num_features": NUM_FEATURES,
        "target": TARGET,
        "oof_auc": round(float(auc), 4),
        "threshold": round(threshold, 3),
        "base_rate": round(float(y.mean()), 4),
        "n_rows": int(len(df)),
        "years": sorted(int(v) for v in df.Year.unique()),
        "races": sorted(df.Race.unique().tolist()),
        "compounds": compounds,
        "race_len": race_len,
        "tyrelife_max": int(df.TyreLife.max()),
        "stint_max": int(df.Stint.max()),
        "position_max": int(df.Position.max()),
        "importances": {k: float(v) for k, v in importances.items()},
        "pit_by_compound": df.groupby("Compound")[TARGET].mean().round(4).to_dict(),
    }
    META_OUT.write_text(json.dumps(meta, indent=2))
    print(f"Saved {MODEL_OUT} and {META_OUT}")

    assert MODEL_OUT.exists() and META_OUT.exists()
    assert 0.5 < auc < 1.0


if __name__ == "__main__":
    main()

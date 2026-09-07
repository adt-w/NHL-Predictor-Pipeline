"""
STEP 4 — Train the ensemble and evaluate honestly.

This wires the exact model your resume describes: a soft-voting ensemble of
Logistic Regression + Gradient Boosting classifiers (scikit-learn). The
plumbing is here so you can spend your time on the parts that matter:
feature engineering (Step 3) and tuning (marked TUNE below).

Run:  python -m src.train
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(config.PROCESSED / "dataset.csv")


def time_aware_split(df: pd.DataFrame):
    """Train on earlier seasons, test on the latest one. NEVER shuffle across
    time for a forecasting task — a random split lets the model peek at the
    future and inflates your score."""
    latest = df["season"].max()
    train = df[df["season"] < latest]
    test = df[df["season"] == latest]
    return train, test, latest


def build_ensemble() -> VotingClassifier:
    logreg = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0)),   # TUNE: C
    ])
    gboost = GradientBoostingClassifier(
        n_estimators=200,          # TUNE
        learning_rate=0.05,        # TUNE
        max_depth=3,               # TUNE
    )
    # soft voting averages predicted probabilities -> calibrated-ish output
    return VotingClassifier(
        estimators=[("logreg", logreg), ("gboost", gboost)],
        voting="soft",
        weights=[1, 1],            # TUNE: try [2, 1] etc.
    )


def report(name: str, y_true, y_prob) -> None:
    y_pred = (y_prob >= 0.5).astype(int)
    print(f"\n{name}")
    print(f"  accuracy : {accuracy_score(y_true, y_pred):.3f}")
    print(f"  roc_auc  : {roc_auc_score(y_true, y_prob):.3f}")
    print(f"  log_loss : {log_loss(y_true, y_prob):.3f}")
    print(f"  brier    : {brier_score_loss(y_true, y_prob):.3f}")


def main() -> None:
    df = load_dataset()
    feature_cols = [c for c in df.columns
                    if c not in ("game_id", "season", "home_win")]
    train, test, latest = time_aware_split(df)
    print(f"Train: {len(train)} games | Test (season {latest}): {len(test)} games")

    X_tr, y_tr = train[feature_cols], train["home_win"]
    X_te, y_te = test[feature_cols], test["home_win"]

    # --- Baseline: always pick the home team ------------------------------
    # NHL home teams win ~55% of the time. Your model must beat THIS to matter.
    home_baseline = accuracy_score(y_te, np.ones(len(y_te)))
    print(f"\nHome-ice baseline accuracy: {home_baseline:.3f}")

    # --- Ensemble ---------------------------------------------------------
    model = build_ensemble()
    model.fit(X_tr, y_tr)
    prob_te = model.predict_proba(X_te)[:, 1]
    report("Ensemble (test)", y_te, prob_te)

    # TUNE: also print each member alone (model.named_estimators_) to see which
    #       one carries the ensemble, then adjust weights / hyperparameters.

    config.MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": feature_cols},
                config.MODELS / "ensemble.joblib")
    print(f"\nSaved -> {config.MODELS / 'ensemble.joblib'}")


if __name__ == "__main__":
    main()

"""
Task 2: ECOD as implemented in PyOD vs ECOD as described in the paper.

PyOD (pyod/models/ecod.py) aggregates differently from the paper:
  - PyOD:  for each feature, take max(left, right, skewness-chosen) tail score,
           THEN sum over features.  -> score_i = sum_j max(L_ij, R_ij)
           (the skewness-chosen tail is always one of L or R, so it never wins
           the max: the skewness correction has no effect)
  - Paper, Eq. (4)-(6): sum each of the three scores over ALL features first,
           THEN take the maximum of the three totals.
           -> score_i = max( sum_j L_ij, sum_j R_ij, sum_j Auto_ij )

This script runs both, plus the paper's variants, under the same protocol
(60/40 split, 10 trials, ROC and AP), so that the only difference is the
aggregation rule.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from pyod.models.ecod import ECOD
from scipy.stats import rankdata, skew
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
TEST_SIZE = 0.4
SEEDS = range(10)

# file -> (paper name, paper ECOD ROC [Table 4], paper ECOD AP [Table 5])
DATASETS = {
    "4_breastw.npz":     ("Breastw (mat)",    0.994, 0.988),
    "6_cardio.npz":      ("Cardio (mat)",     0.897, 0.579),
    "18_Ionosphere.npz": ("Ionosphere (mat)", 0.831, 0.719),
    "29_Pima.npz":       ("Pima (mat)",       0.664, 0.541),
    "42_WBC.npz":        ("WBC (arff)",       0.990, 0.838),
}


# ---------------------------------------------------------------------------
# ECOD following the paper's equations
# ---------------------------------------------------------------------------
def tail_scores(X):
    """Per-feature negative log tail probabilities, Eq. (1)-(2).

    Returns L, R, Auto, each of shape (n_samples, n_features).
    """
    n = X.shape[0]
    # left-tail ECDF: share of observations <= x  (rank with ties -> max)
    F_left = rankdata(X, method="max", axis=0) / n
    # right-tail ECDF: share of observations >= x
    F_right = (n - rankdata(X, method="min", axis=0) + 1) / n

    L = -np.log(F_left)
    R = -np.log(F_right)

    # skewness per feature: gamma < 0 -> left tail, gamma >= 0 -> right tail
    gamma = np.nan_to_num(skew(X, axis=0))
    Auto = np.where(gamma < 0, L, R)
    return L, R, Auto


def paper_scores(X_train, X_test):
    """All paper variants for the test points.

    As in PyOD, ECDFs are estimated on train + test stacked together,
    so that both implementations see exactly the same data.
    """
    X = np.vstack([X_train, X_test])
    L, R, Auto = tail_scores(X)
    O_left, O_right, O_auto = L.sum(1), R.sum(1), Auto.sum(1)  # Eq. (4), (5)
    out = {
        "Paper Eq.(6)": np.maximum.reduce([O_left, O_right, O_auto]),  # Eq. (6)
        "Skewness only": O_auto,
        "ECOD-L": O_left,
        "ECOD-R": O_right,
        "ECOD-B": 0.5 * (O_left + O_right),
    }
    m = X_test.shape[0]
    return {k: v[-m:] for k, v in out.items()}


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
METHODS = ["PyOD", "Paper Eq.(6)", "Skewness only", "ECOD-L", "ECOD-R", "ECOD-B"]
roc_rows, ap_rows = [], []

for file_name, (paper_name, paper_roc, paper_ap) in DATASETS.items():
    data = np.load(DATA_DIR / file_name, allow_pickle=True)
    X, y = data["X"].astype(float), data["y"].astype(int).ravel()

    roc = {m: [] for m in METHODS}
    ap = {m: [] for m in METHODS}
    for seed in SEEDS:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed
        )
        scores = {"PyOD": ECOD(n_jobs=1).fit(X_tr).decision_function(X_te)}
        scores.update(paper_scores(X_tr, X_te))
        for m in METHODS:
            roc[m].append(roc_auc_score(y_te, scores[m]))
            ap[m].append(average_precision_score(y_te, scores[m]))

    roc_rows.append({"Dataset": paper_name, "Paper (Table 4)": paper_roc,
                     **{m: round(np.mean(roc[m]), 3) for m in METHODS}})
    ap_rows.append({"Dataset": paper_name, "Paper (Table 5)": paper_ap,
                    **{m: round(np.mean(ap[m]), 3) for m in METHODS}})
    print(f"Done: {paper_name}")

roc_df, ap_df = pd.DataFrame(roc_rows), pd.DataFrame(ap_rows)
for df in (roc_df, ap_df):
    avg = df.drop(columns="Dataset").mean().round(3)
    df.loc[len(df)] = {"Dataset": "AVG", **avg.to_dict()}

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
print("\nROC\n", roc_df.to_string(index=False))
print("\nAverage precision\n", ap_df.to_string(index=False))

roc_df.to_csv(Path(__file__).parent / "ecod_comparison_roc.csv", index=False)
ap_df.to_csv(Path(__file__).parent / "ecod_comparison_ap.csv", index=False)
print("\nSaved to ecod_comparison_roc.csv and ecod_comparison_ap.csv")

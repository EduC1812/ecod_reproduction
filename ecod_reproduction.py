"""
Task 2: Reproducing the main ECOD experiment (Li et al., 2022, Tables 4 and 5).

Protocol, as described in Section 4.1 of the paper:
  - 60% of each dataset for training, 40% for testing
  - 10 independent trials (different random splits), results averaged
  - Metrics: ROC AUC and average precision (AP) on the test set

Data: ADBench preprocessed versions of the benchmark datasets (.npz with keys X, y),
stored in the data/ folder next to this script.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from pyod.models.ecod import ECOD
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Parameters (all values used in the experiment)
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
TEST_SIZE = 0.4          # 60/40 train/test split, as in the paper
N_TRIALS = 10            # 10 independent trials, as in the paper
SEEDS = range(N_TRIALS)  # random_state = 0, 1, ..., 9 for each split
CONTAMINATION = 0.1      # PyOD default; only affects the 0/1 labels, not the scores used for ROC/AP
N_JOBS = 1               # single-process version, as in the paper

# file name -> (name in the paper's tables, paper ROC, paper AP) from Tables 4 and 5
DATASETS = {
    "4_breastw.npz":     ("Breastw (mat)",    0.994, 0.988),
    "6_cardio.npz":      ("Cardio (mat)",     0.897, 0.579),
    "18_Ionosphere.npz": ("Ionosphere (mat)", 0.831, 0.719),
    "29_Pima.npz":       ("Pima (mat)",       0.664, 0.541),
    "42_WBC.npz":        ("WBC (arff)",       0.990, 0.838),
}

# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
rows = []
for file_name, (paper_name, paper_roc, paper_ap) in DATASETS.items():
    data = np.load(DATA_DIR / file_name, allow_pickle=True)
    X, y = data["X"], data["y"].astype(int).ravel()

    rocs, aps = [], []
    for seed in SEEDS:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed
        )
        clf = ECOD(contamination=CONTAMINATION, n_jobs=N_JOBS)
        clf.fit(X_train)                        # unsupervised: labels are not used
        scores = clf.decision_function(X_test)  # higher = more outlying
        rocs.append(roc_auc_score(y_test, scores))
        aps.append(average_precision_score(y_test, scores))

    rows.append({
        "Dataset": paper_name,
        "n": X.shape[0],
        "d": X.shape[1],
        "% outliers": round(100 * y.mean(), 1),
        "ROC (ours)": round(np.mean(rocs), 3),
        "ROC sd": round(np.std(rocs), 3),
        "ROC (paper)": paper_roc,
        "ROC diff": round(np.mean(rocs) - paper_roc, 3),
        "AP (ours)": round(np.mean(aps), 3),
        "AP sd": round(np.std(aps), 3),
        "AP (paper)": paper_ap,
        "AP diff": round(np.mean(aps) - paper_ap, 3),
    })
    print(f"Done: {paper_name}")

results = pd.DataFrame(rows)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
print("\n", results.to_string(index=False))

results.to_csv(Path(__file__).parent / "ecod_results.csv", index=False)
print("\nSaved to ecod_results.csv")

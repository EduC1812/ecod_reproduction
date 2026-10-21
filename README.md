# Reproducing ECOD (Li et al., 2022)

Code for Task 2 of the Data Wrangling assignment (Applied Data Science, Utrecht University):
reproduction of the main experiment of

> Li, Z., Zhao, Y., Hu, X., Botta, N., Ionescu, C. and Chen, G. H. (2022).
> ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions.
> *IEEE Transactions on Knowledge and Data Engineering*. arXiv:2201.00382.

## Files

| File | What it does |
|---|---|
| `ecod_reproduction.py` | Runs PyOD's ECOD with the paper's protocol and compares ROC/AP with Tables 4 and 5 of the paper. |
| `ecod_comparison.py` | Runs PyOD's ECOD next to an implementation that follows Eqs. (1)–(6) of the paper, plus the ablation variants (left tail, right tail, average of both, skewness-based score). |

## Protocol and parameters

| Setting | Value |
|---|---|
| Train/test split | 60/40 (`test_size = 0.4`) |
| Trials | 10 (`random_state = 0, …, 9`), results averaged |
| Metrics | ROC AUC and average precision on the test set |
| `contamination` | 0.1 (PyOD default; does not affect ROC or AP) |
| `n_jobs` | 1 |
| Feature scaling | none |
| ECDF estimation | training and test data together (as in PyOD) |

## Data

The datasets are the preprocessed versions from ADBench
(https://github.com/Minqi824/ADBench/tree/main/adbench/datasets/Classical).
Download these files into `data/`:

- `4_breastw.npz`
- `6_cardio.npz`
- `18_Ionosphere.npz`
- `29_Pima.npz`
- `42_WBC.npz`

## How to run

Tested with Python 3.12.

```
pip install -r requirements.txt
python ecod_reproduction.py
python ecod_comparison.py
```

Results are printed and saved as CSV files in the same folder.

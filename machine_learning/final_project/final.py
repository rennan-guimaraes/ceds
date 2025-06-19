#!/usr/bin/env python3
"""German Credit Classification - Full Pipeline in Python
Author: Rennan Guimarães
Date: 2024-09-24

Requirements (install via pip if missing):
    pyreadr
    pandas
    numpy
    scikit-learn
    imbalanced-learn
    feature_engine
    matplotlib
    seaborn
    lightgbm
    xgboost
    scipy
    ydata-profiling

Usage:
    Place `german.rds` in the same directory as this script, then run:
        python german_credit_classification.py

Outputs:
    - german_credit_eda_report.html : exploratory data analysis report
    - metrics_cv.csv : cross‑validation metric summary
    - metrics_test.csv : test‑set metric summary
    - auc_boxplot.png : variability of ROC‑AUC across folds
    - roc_curves.png : ROC curves for best models on test set
    - pr_curves.png : Precision-Recall curves for best models on test set
    - final_classification_model.pkl : pickle of final LightGBM model
"""

# ---------------------------------------------------------------------
# Imports and global config
# ---------------------------------------------------------------------
import warnings, pathlib, sys, time

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pyreadr
import ydata_profiling
from scipy.stats import ks_2samp

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from feature_engine.outliers import Winsorizer

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from copy import deepcopy
import joblib

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
METRICS = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "roc_auc": roc_auc_score,
    "ks": lambda y_true, y_prob: ks_2samp(y_prob[y_true == 1], y_prob[y_true == 0])[0],
}


def evaluate_cv(model, X, y, cv):
    """Return DataFrame with metric values for each CV split."""
    rows = []
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        rows.append(
            {
                m: f(y_val, y_pred if m not in ["roc_auc", "ks"] else y_prob)
                for m, f in METRICS.items()
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 0. Load data
# ---------------------------------------------------------------------
DATA_PATH = pathlib.Path(__file__).parent / "german.rds"
if not DATA_PATH.exists():
    sys.exit("ERROR: german.rds not found. Please place the file next to this script.")

result = pyreadr.read_r(DATA_PATH)
german = result[None]
german.columns = german.columns.str.replace("[^0-9a-zA-Z]+", "_", regex=True).str.strip(
    "_"
)

# Clean categorical column values for XGBoost compatibility
for col in german.select_dtypes(exclude=["number"]).columns:
    if col != "Good_loan":
        german[col] = german[col].astype(str).str.replace(r"[\[\]<]", "_", regex=True)

german["Good_loan"] = german["Good_loan"].astype("category").cat.codes

# ---------------------------------------------------------------------
# 0.5. Exploratory Data Analysis
# ---------------------------------------------------------------------
print("=== Gerando relatório de Análise Exploratória (EDA) ===")
profile = ydata_profiling.ProfileReport(
    german, title="German Credit Data Profiling", minimal=True
)
profile.to_file("german_credit_eda_report.html")
print("--> Relatório salvo em german_credit_eda_report.html\n")

X = german.drop(columns=["Good_loan"])
y = german["Good_loan"]

# ---------------------------------------------------------------------
# 1. Train‑test split
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=17, stratify=y
)

num_cols = X_train.select_dtypes(include=["number"]).columns
cat_cols = X_train.select_dtypes(exclude=["number"]).columns

# ---------------------------------------------------------------------
# 2. Pre‑processing pipeline
# ---------------------------------------------------------------------
winsor = Winsorizer(capping_method="quantiles", tail="both", fold=0.01)

num_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("winsor", winsor),
        ("scaler", StandardScaler()),
    ]
)

cat_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocessor = ColumnTransformer(
    [("num", num_pipeline, num_cols), ("cat", cat_pipeline, cat_cols)]
)
preprocessor.set_output(transform="pandas")

# ---------------------------------------------------------------------
# 3. Model definitions and search spaces
# ---------------------------------------------------------------------
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=17)

searches = {}

# -- KNN
knn_pipe = ImbPipeline(
    steps=[
        ("prep", preprocessor),
        ("smote", SMOTE(random_state=17)),
        ("model", KNeighborsClassifier()),
    ]
)
param_knn = {
    "model__n_neighbors": range(3, 31),
    "model__weights": ["uniform", "distance"],
    "model__p": [1, 2],
}
searches["KNN"] = RandomizedSearchCV(
    knn_pipe, param_knn, n_iter=20, cv=cv, scoring="roc_auc", n_jobs=-1, random_state=17
)

# -- Decision Tree
dt_pipe = ImbPipeline(
    steps=[
        ("prep", preprocessor),
        ("smote", SMOTE(random_state=17)),
        ("model", DecisionTreeClassifier(random_state=17)),
    ]
)
param_dt = {
    "model__max_depth": range(1, 31),
    "model__min_samples_split": range(2, 21),
    "model__ccp_alpha": np.linspace(0.0, 0.02, 20),
}
searches["Decision Tree"] = RandomizedSearchCV(
    dt_pipe, param_dt, n_iter=20, cv=cv, scoring="roc_auc", n_jobs=-1, random_state=17
)

# -- Random Forest
rf_pipe = ImbPipeline(
    steps=[
        ("prep", preprocessor),
        ("smote", SMOTE(random_state=17)),
        ("model", RandomForestClassifier(random_state=17)),
    ]
)
param_rf = {
    "model__n_estimators": range(500, 2001, 250),
    "model__max_depth": [None] + list(range(5, 31, 5)),
    "model__min_samples_split": range(2, 21),
    "model__max_features": ["sqrt", "log2"],
}
searches["Random Forest"] = RandomizedSearchCV(
    rf_pipe, param_rf, n_iter=20, cv=cv, scoring="roc_auc", n_jobs=-1, random_state=17
)

# -- LightGBM
lgbm_pipe = ImbPipeline(
    steps=[
        ("prep", preprocessor),
        ("smote", SMOTE(random_state=17)),
        (
            "model",
            LGBMClassifier(objective="binary", random_state=17, verbosity=-1),
        ),
    ]
)
param_lgbm = {
    "model__n_estimators": range(500, 2001, 250),
    "model__max_depth": [-1] + list(range(3, 16)),
    "model__learning_rate": np.linspace(0.01, 0.3, 30),
    "model__subsample": np.linspace(0.5, 1.0, 20),
    "model__colsample_bytree": np.linspace(0.5, 1.0, 20),
}
searches["LightGBM"] = RandomizedSearchCV(
    lgbm_pipe,
    param_lgbm,
    n_iter=20,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    random_state=17,
)

# -- XGBoost
xgb_pipe = ImbPipeline(
    steps=[
        ("prep", preprocessor),
        ("smote", SMOTE(random_state=17)),
        (
            "model",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=17,
            ),
        ),
    ]
)
param_xgb = {
    "model__n_estimators": range(500, 2001, 250),
    "model__max_depth": range(3, 16),
    "model__learning_rate": np.linspace(0.01, 0.3, 30),
    "model__subsample": np.linspace(0.5, 1.0, 20),
    "model__colsample_bytree": np.linspace(0.5, 1.0, 20),
    "model__min_child_weight": range(1, 11),
}
searches["XGBoost"] = RandomizedSearchCV(
    xgb_pipe, param_xgb, n_iter=20, cv=cv, scoring="roc_auc", n_jobs=-1, random_state=17
)

# ---------------------------------------------------------------------
# 4. Hyperparameter tuning with CV
# ---------------------------------------------------------------------
cv_results = {}
tuning_times = {}
for name, search in searches.items():
    print(f"=== Treinando {name} ===")
    start_time = time.time()
    search.fit(X_train, y_train)
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    tuning_times[name] = duration

    cv_scores = evaluate_cv(search.best_estimator_, X_train, y_train, cv)
    cv_results[name] = cv_scores
    print(cv_scores.mean().round(3))
    print(f"--> Tempo de tuning: {duration}s\n")

# Save CV metrics
cv_summary = (
    pd.concat(cv_results, names=["Model"])
    .groupby(level=0)
    .agg(["mean", "std"])
    .round(3)
)
cv_summary.to_csv("metrics_cv.csv")

# Boxplot AUC
auc_long = (
    pd.concat(cv_results, names=["Model"]).reset_index().loc[:, ["Model", "roc_auc"]]
)
plt.figure(figsize=(8, 4))
sns.boxplot(data=auc_long, x="Model", y="roc_auc")
plt.title("Variabilidade da AUC entre folds")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("auc_boxplot.png", dpi=300)

# ---------------------------------------------------------------------
# 5. Evaluation on test set
# ---------------------------------------------------------------------
best_models = {name: s.best_estimator_ for name, s in searches.items()}
test_rows = []
fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
fig_pr, ax_pr = plt.subplots(figsize=(10, 8))

for idx, (name, mdl) in enumerate(best_models.items(), start=1):
    model = deepcopy(mdl)
    model.fit(X_train, y_train)

    start_pred_time = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    end_pred_time = time.time()

    test_rows.append(
        {
            "Model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "ks": ks_2samp(y_prob[y_test == 1], y_prob[y_test == 0])[0],
            "tuning_time_s": tuning_times[name],
            "predict_time_s": round(end_pred_time - start_pred_time, 4),
        }
    )

    RocCurveDisplay.from_predictions(y_test, y_prob, name=name, ax=ax_roc)
    PrecisionRecallDisplay.from_predictions(y_test, y_prob, name=name, ax=ax_pr)

ax_roc.set_title("ROC Curves — Test Set")
ax_roc.legend()
fig_roc.savefig("roc_curves.png", dpi=300)

ax_pr.set_title("Precision-Recall Curves — Test Set")
ax_pr.legend()
fig_pr.savefig("pr_curves.png", dpi=300)

plt.close(fig_roc)
plt.close(fig_pr)

test_metrics = pd.DataFrame(test_rows).round(3)
test_metrics.to_csv("metrics_test.csv", index=False)
print("\nTest‑set metrics:")
print(test_metrics)

# ---------------------------------------------------------------------
# 6. Select best model (highest ROC AUC) and train on full data
# ---------------------------------------------------------------------
best_name = test_metrics.loc[test_metrics["roc_auc"].idxmax(), "Model"]
print(f"\n>>> Melhor modelo selecionado: {best_name}")

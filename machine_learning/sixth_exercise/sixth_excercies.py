import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold

# ---------------------------------------------------------------------
# 1. Carregamento ------------------------------------------------------
csv_path = Path("forestfires.csv")
if not csv_path.exists():
    raise FileNotFoundError(
        "Coloque o dataset CSV (forestfires.csv) na mesma pasta deste script "
        "ou ajuste o caminho em 'csv_path'."
    )

df = pd.read_csv(csv_path)

# A variável-alvo
y_raw = df["area"].values
# Aplicamos log1p porque 'area' é extremamente assimétrica
y = np.log1p(y_raw)

# Atributos
X = df.drop(columns=["area"])

# ---------------------------------------------------------------------
# 2. Pré-processamento + CV (versão sem estratificação)
from sklearn.model_selection import KFold

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = ["month", "day"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

cv = KFold(n_splits=5, shuffle=True, random_state=42)  # ✅ agora aceita y contínuo


# Escolha da métrica -- aqui usaremos RMSE (menor é melhor)
rmse = make_scorer(
    lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)), greater_is_better=False
)

# ---------------------------------------------------------------------
# 3. Modelos + grades de hiperparâmetros ------------------------------

pipelines = {
    "KNN": Pipeline(
        steps=[
            ("prep", preprocess),
            ("reg", KNeighborsRegressor()),
        ]
    ),
    "DecisionTree": Pipeline(
        steps=[
            ("prep", preprocess),
            ("reg", DecisionTreeRegressor(random_state=42)),
        ]
    ),
    "SVR": Pipeline(
        steps=[
            ("prep", preprocess),
            ("reg", SVR()),
        ]
    ),
    "MLP": Pipeline(
        steps=[
            ("prep", preprocess),
            ("reg", MLPRegressor(max_iter=1000, random_state=42, early_stopping=True)),
        ]
    ),
}

param_grids = {
    "KNN": {
        "reg__n_neighbors": [3, 5, 7, 11],
        "reg__weights": ["uniform", "distance"],
        "reg__p": [1, 2],  # Manhattan ou Euclidiana
    },
    "DecisionTree": {
        "reg__max_depth": [None, 4, 6, 8, 12],
        "reg__min_samples_leaf": [1, 3, 5, 10],
        "reg__criterion": ["squared_error", "friedman_mse"],
    },
    "SVR": {
        "reg__kernel": ["rbf", "poly", "sigmoid"],
        "reg__C": [1, 10, 100],
        "reg__gamma": ["scale", "auto"],
        "reg__epsilon": [0.1, 0.5, 1.0],
    },
    "MLP": {
        "reg__hidden_layer_sizes": [(50,), (100,), (50, 50)],
        "reg__activation": ["relu", "tanh"],
        "reg__alpha": [1e-4, 1e-3],
        "reg__learning_rate_init": [0.001, 0.01],
    },
}

# ---------------------------------------------------------------------
# 4. Busca em grade + CV ----------------------------------------------
results = []

for name, pipe in pipelines.items():
    print(f"\n🔍 {name} – grid-search...")
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grids[name],
        scoring=rmse,  # negativo (quanto mais próximo de 0, melhor)
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X, y)
    best_rmse = -grid.best_score_  # invertendo sinal
    best_r2 = grid.cv_results_[
        "mean_test_score"
    ]  # usamos RMSE, mas podemos recalcular R^2
    # Recalcula R^2 para o melhor estimador
    r2_val = r2_score(y, grid.best_estimator_.predict(X))
    results.append(
        {
            "Modelo": name,
            "Melhor_RMSE_CV": round(best_rmse, 3),
            "R2_no_treino": round(r2_val, 3),
            "Best_Params": grid.best_params_,
        }
    )
    print(f"→ RMSE (CV): {best_rmse:.3f}")

# ---------------------------------------------------------------------
# 5. Resumo ------------------------------------------------------------
print("\n===== Resumo =====")
res_df = pd.DataFrame(results).set_index("Modelo")
print(res_df)

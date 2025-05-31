# Exercício 1
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

# ---------------------------------------------------------------------
# 1. Carregar o dataset
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# ---------------------------------------------------------------------
# 2. Definir CV estratificada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

# ---------------------------------------------------------------------
# 3. SVM
svm_pipe = Pipeline([("scaler", StandardScaler()), ("clf", SVC())])

svm_param_grid = {
    "clf__kernel": ["linear", "rbf", "poly"],
    "clf__C": [0.1, 1, 10],
    "clf__gamma": ["scale", "auto"],
}

svm_grid = GridSearchCV(svm_pipe, svm_param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
svm_grid.fit(X, y)

results.append(
    {
        "Modelo": "SVM",
        "Melhor Acurácia CV": svm_grid.best_score_,
        "Melhores Params": svm_grid.best_params_,
    }
)

# ---------------------------------------------------------------------
# 4. Perceptron
perc_pipe = Pipeline(
    [("scaler", StandardScaler()), ("clf", Perceptron(max_iter=1000, tol=1e-3))]
)

perc_param_grid = {
    "clf__penalty": [None, "l2", "l1", "elasticnet"],
    "clf__alpha": [1e-4, 1e-3, 1e-2],
    "clf__eta0": [0.1, 1.0, 10],
    "clf__fit_intercept": [True, False],
}

perc_grid = GridSearchCV(
    perc_pipe, perc_param_grid, cv=cv, scoring="accuracy", n_jobs=-1
)
perc_grid.fit(X, y)

results.append(
    {
        "Modelo": "Perceptron",
        "Melhor Acurácia CV": perc_grid.best_score_,
        "Melhores Params": perc_grid.best_params_,
    }
)

# ---------------------------------------------------------------------
# 5. Multi-layer Perceptron
mlp_pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(max_iter=500, random_state=42, early_stopping=True)),
    ]
)

mlp_param_grid = {
    "clf__hidden_layer_sizes": [(50,), (100,), (50, 50)],
    "clf__activation": ["relu", "tanh"],
    "clf__alpha": [1e-4, 1e-3],
    "clf__learning_rate_init": [0.001, 0.01],
}

mlp_grid = GridSearchCV(mlp_pipe, mlp_param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
mlp_grid.fit(X, y)

results.append(
    {
        "Modelo": "MLP",
        "Melhor Acurácia CV": mlp_grid.best_score_,
        "Melhores Params": mlp_grid.best_params_,
    }
)

# ---------------------------------------------------------------------
# 6. Exibir resultados
df_results = pd.DataFrame(results)

print(df_results)

# ---------------------------------------------------------------------
# Exercício 2
# ---------- Dataset ----------------------------------------------------------
# Mapas: Sim=1 / Não=0 ; Grandes=1 / Pequenas=0
pacientes = {
    "João": ([1, 1, 0, 1], 1),
    "Pedro": ([0, 0, 1, 0], 0),
    "Maria": ([1, 1, 0, 0], 0),
    "José": ([1, 0, 1, 1], 1),
    "Ana": ([1, 0, 0, 1], 0),
    "Leila": ([0, 0, 1, 1], 1),
}

X = np.array([v[0] for v in pacientes.values()], dtype=int)
d = np.array([v[1] for v in pacientes.values()], dtype=int)

# ---------- Parâmetros -------------------------------------------------------
eta = 0.5
threshold = -0.5
max_epochs = 100

w = np.zeros(X.shape[1])
b = 0.0

# ---------- Treinamento com verbose -----------------------------------------
for epoch in range(max_epochs):
    print(f"\n===== Época {epoch} =====")
    errors = 0
    for idx, (xi, target) in enumerate(zip(X, d)):
        v = np.dot(w, xi) + b
        y = 1 if v >= threshold else 0
        erro = target - y

        print(f"Amostra {idx+1}: x={xi}, d={target}, v={v:.2f}, y={y}, erro={erro}")

        if erro != 0:
            # Antes da atualização
            old_w = w.copy()
            old_b = b
            # Regra de ajuste
            w += eta * erro * xi
            b += eta * erro
            errors += 1
            print(f"  -> Ajuste! w: {old_w} -> {w},  b: {old_b:.2f} -> {b:.2f}")
        else:
            print("  -> Nenhum ajuste necessário.")

    print(f"Erros nesta época: {errors}")
    if errors == 0:
        print("Rede convergiu!")
        break

print("\nPesos finais:", w)
print("Bias final:", b)

# ---------- Testes -----------------------------------------------------------
testes = {
    "Luis": np.array([0, 0, 0, 1]),
    "Laura": np.array([1, 1, 1, 1]),
}


def classificar(xi):
    v = np.dot(w, xi) + b
    return 1 if v >= threshold else 0


print("\n=== Testes finais ===")
for nome, xi in testes.items():
    pred = classificar(xi)
    print(f"{nome}: {'Doente' if pred else 'Saudável'}")

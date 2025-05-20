import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import warnings

# ---------------------
# Carregar dataset e silenciar warnings
# ---------------------
wine_dataset = load_wine()
wine_features = pd.DataFrame(wine_dataset.data, columns=wine_dataset.feature_names)
wine_target = pd.Series(wine_dataset.target, name="class")
warnings.filterwarnings("ignore")

# ---------------------
# 1a
# ---------------------

for feature_name in wine_features.columns:
    # Cálculo dos limites para outliers com base no slide da aula 2 (Q3-Q1)
    Q1 = wine_features[feature_name].quantile(0.25)
    Q3 = wine_features[feature_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = wine_features[
        (wine_features[feature_name] < lower_bound)
        | (wine_features[feature_name] > upper_bound)
    ][feature_name]

    # Criar figura com dois subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])

    # Boxplot no subplot superior
    sns.boxplot(
        x=feature_name, data=wine_features, palette="Set3", ax=ax1, legend=False
    )
    ax1.set_title(f"Boxplot e Análise de Outliers - {feature_name}")
    ax1.set_ylabel("Valor")

    # Estatísticas no subplot inferior
    ax2.axis("off")
    stats_text = f"""
    Média: {wine_features[feature_name].mean():.3f}
    Desvio Padrão: {wine_features[feature_name].std():.3f}
    Número de outliers: {len(outliers)}
    """
    if len(outliers) > 0:
        stats_text += f"\nValores dos outliers:\n{outliers.values}"

    ax2.text(0.1, 0.5, stats_text, fontsize=10, va="center")

    plt.tight_layout()
    plt.show()


# ---------------------
# 1b
# ---------------------
min_max_scaler = MinMaxScaler()
normalized_features = pd.DataFrame(
    min_max_scaler.fit_transform(wine_features), columns=wine_dataset.feature_names
)
print("\nPrimeiras linhas após MinMax Scaling:\n", normalized_features.head())

# ---------------------
# 1c
# ---------------------
pca_transformer = PCA(n_components=2, random_state=42)
pca_features = pca_transformer.fit_transform(normalized_features)

print(
    f"Variância explicada pelos 2 componentes: {pca_transformer.explained_variance_ratio_.sum():.2%}"
)

plt.figure(figsize=(8, 6))
color_palette = sns.color_palette("Set1", n_colors=len(np.unique(wine_target)))
sns.scatterplot(
    x=pca_features[:, 0],
    y=pca_features[:, 1],
    hue=wine_target,
    palette=color_palette,
    s=60,
    alpha=0.8,
    edgecolor="k",
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Wine PCA – 2 componentes")
plt.legend(title="Classe")
plt.tight_layout()
plt.show()

# ---------------------
# 1d
# ---------------------
correlation_matrix = wine_features.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix,
    annot=False,
    cmap="coolwarm",
    square=True,
    cbar_kws={"shrink": 0.8},
)
plt.title("Matriz de Correlação – Dataset Wine")
plt.tight_layout()
plt.show()

# Pares de features com correlação >= 0.75
correlation_pairs = (
    correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    .stack()
    .reset_index()
)
correlation_pairs.columns = ["feature_1", "feature_2", "correlation"]
high_correlation_pairs = correlation_pairs.loc[
    correlation_pairs["correlation"].abs() >= 0.75
]
print("\nPares de features com correlação ≥ 0.75:\n", high_correlation_pairs)

# ---------------------
# 1e
# ---------------------
class_distribution = wine_target.value_counts().sort_index()
print("\nDistribuição das classes:\n", class_distribution)
plt.figure(figsize=(6, 4))
sns.barplot(
    x=class_distribution.index, y=class_distribution.values, palette=color_palette
)
plt.xlabel("Classe")
plt.ylabel("Número de instâncias")
plt.title("Distribuição das Classes – Dataset Wine")
for idx, val in enumerate(class_distribution.values):
    plt.text(idx, val + 1, str(val), ha="center")
plt.tight_layout()
plt.show()

class_percentage = class_distribution.values / len(wine_target) * 100
print("\nPercentual de classes:\n", class_percentage)

# ---------------------
# 2
# ---------------------
features_with_missing = wine_features.copy()

# Remover 5% dos valores do dataset de forma aleatória
random_generator = np.random.default_rng(42)
missing_mask = random_generator.random(features_with_missing.shape) < 0.05
features_with_missing = features_with_missing.mask(missing_mask)

missing_percentage = features_with_missing.isna().mean().mean() * 100
print(f"\nPercentual total de valores ausentes introduzidos: {missing_percentage:.2f}%")

# Imputar usando mediana
median_imputer = SimpleImputer(strategy="median")
imputed_features = pd.DataFrame(
    median_imputer.fit_transform(features_with_missing),
    columns=wine_dataset.feature_names,
)


## Referências:
# Slides
# Código de referência passado em aula
# Chat gpt para melhorar eficiencia do código, como por exemplo os loops e retirar dúvidas de entendimento da documentação
# Links consultados:
# - [https://matplotlib.org/stable/api/\_as\_gen/matplotlib.pyplot.boxplot.html](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html)
# - [https://seaborn.pydata.org/generated/seaborn.boxplot.html](https://seaborn.pydata.org/generated/seaborn.boxplot.html)
# - [https://numpy.org/doc/2.1/reference/generated/numpy.quantile.html](https://numpy.org/doc/2.1/reference/generated/numpy.quantile.html)
# - [https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.quantile.html](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.quantile.html)
# - [https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
# - [https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
# - [https://seaborn.pydata.org/generated/seaborn.scatterplot.html](https://seaborn.pydata.org/generated/seaborn.scatterplot.html)
# - [https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html)
# - [https://seaborn.pydata.org/generated/seaborn.heatmap.html](https://seaborn.pydata.org/generated/seaborn.heatmap.html)
# - [https://pandas.pydata.org/docs/reference/api/pandas.Series.value\_counts.html](https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html)
# - [https://seaborn.pydata.org/generated/seaborn.barplot.html](https://seaborn.pydata.org/generated/seaborn.barplot.html)
# - [https://numpy.org/doc/2.2/reference/random/generator.html](https://numpy.org/doc/2.2/reference/random/generator.html)
# - [https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)

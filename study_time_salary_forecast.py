# Problema de Negócio:
# Usando dados históricos é possível prever o salário de alguém com base no
# tempo dedicado aos estudos em horas por mês?

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from utils.figures import difference_plot

# %%
# Carregando os dados
data = pd.read_csv("data/dataset.csv")
data.head()

# %%
# Análise exploratória
data.describe()
# %%
data.info(memory_usage="deep")
# %%
data.isnull().any()
# %%
data["salario"].value_counts()
# %%
data["horas_estudo_mes"].value_counts()
# %%
data.shape
# %%
sns.histplot(data=data, x="horas_estudo_mes", kde=True)

# %%
# Correlação dos dados
data.corr()

# %%
# Preparação dos dados
X = data["horas_estudo_mes"].values.reshape(-1, 1)  # Matriz com uma coluna
y = data["salario"].values  # Vetor
# %%
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(X, y, "o", color="blue")
ax.set_xlabel("Horas de Estudo por Mês")
ax.set_ylabel("Salário")
ax.grid(True)
ax.legend(["Salário x Horas de Estudo"])
plt.show()
# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"x_train shape: {X_train.shape}")
print(f"x_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
# %%
# Treinamento do modelo
model = LinearRegression()
model.fit(X_train, y_train)
# %%
# Avaliação do modelo
y_pred = model.predict(X_test)
# %%
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(X_test, y_test, "o", color="blue")
ax.plot(X_test, y_pred, color="red")
ax.set_xlabel("Horas de Estudo por Mês")
ax.set_ylabel("Salário")
ax.grid(True)
ax.legend(["Real", "Previsto"])
plt.show()
plt.savefig("data/salario_real_vs_previsto.png")
# %%
print(f"Model score (R^2): {round(model.score(X_test, y_test), 4)}")
# %%
print(f"Coeficiente angular: {model.coef_[0]}")
# %%
difference_plot(
    real_data=y_test,
    predicted_data=y_pred,
    xy_labels={"x": "Salário Real", "y": "Salário Previsto"},
    legends={
        "ax0": ["Real", "Previsto"],
        "ax1": ["Diferença entre Real e Previsto"],
    },
    return_fig=True,
    save=True,
    filename="data/salario_real_vs_previsto2",
    formats=["png", "pdf"],
)
# %%
model.intercept_

# %%
# Deploy do modelo
horas_estudo = np.array([48]).reshape(-1, 1)

salario_previsto = round(model.predict(horas_estudo)[0], 4)
print(
    f"Salário previsto para {horas_estudo[0][0]} horas de estudo: {salario_previsto}"
)
novo_salario = round(model.intercept_ + model.coef_[0] * horas_estudo[0][0], 4)
print(
    f"Salário previsto para {horas_estudo[0][0]} horas de estudo: {novo_salario}"
)

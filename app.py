import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier
from imblearn.over_sampling import SMOTE

# CONFIG DA PÁGINA
st.set_page_config(page_title="Análise de Crédito", layout="wide")

st.title("💳 Análise e Previsão de Aprovação de Empréstimos")

# CARREGAR DADOS
@st.cache_data
def load_data():
    df = pd.read_csv('emprestimos_banco.csv')
    return df

df = load_data()

# SIDEBAR
st.sidebar.header("Navegação")
opcao = st.sidebar.radio("Escolha uma opção", [
    "Visão Geral",
    "Análise Exploratória",
    "Modelo de Machine Learning",
    "Previsão"
])

# =========================
# 📊 VISÃO GERAL
# =========================
if opcao == "Visão Geral":
    st.subheader("📊 Dados Brutos")
    st.dataframe(df)

    st.subheader("📌 Estatísticas por Aprovação")
    st.write(df.groupby('aprovado')[['idade','renda_mensal','score_credito','tempo_emprego','dependentes']].mean())

    st.subheader("📈 Distribuição (%)")
    porcentagem = df['aprovado'].value_counts(normalize=True) * 100
    st.write(porcentagem)

# =========================
# 📈 ANÁLISE
# =========================
elif opcao == "Análise Exploratória":

    st.subheader("Idade vs Renda")
    fig, ax = plt.subplots()
    sns.scatterplot(x='idade', y='renda_mensal', hue='aprovado', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Score vs Renda")
    fig, ax = plt.subplots()
    sns.scatterplot(x='score_credito', y='renda_mensal', hue='aprovado', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Idade vs Score")
    fig, ax = plt.subplots()
    sns.scatterplot(x='idade', y='score_credito', hue='aprovado', data=df, ax=ax)
    st.pyplot(fig)

# =========================
# 🤖 MODELO
# =========================
elif opcao == "Modelo de Machine Learning":

    X = df[['idade','renda_mensal','divida_atual','score_credito','tempo_emprego','dependentes']]
    y = df['aprovado']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    smote = SMOTE(k_neighbors=2, random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    modelo = LogisticRegression()
    modelo.fit(X_train_res, y_train_res)

    previsao = modelo.predict(X_test)

    st.subheader("📊 Resultado do Modelo")

    st.write(f"Acurácia: {modelo.score(X_test, y_test):.2%}")
    st.text(classification_report(y_test, previsao))

    # Modelo burro
    burro = DummyClassifier(strategy='most_frequent')
    burro.fit(X_train, y_train)

    st.subheader("🤖 Comparação com modelo burro")
    st.write(f"Modelo burro: {burro.score(X_test, y_test):.2%}")

# =========================
# 🧪 PREVISÃO
# =========================
elif opcao == "Previsão":

    st.subheader("🧪 Teste um cliente")

    idade = st.slider("Idade", 18, 80, 30)
    renda = st.number_input("Renda Mensal", 1000, 50000, 3000)
    divida = st.number_input("Dívida Atual", 0, 50000, 1000)
    score = st.slider("Score de Crédito", 0, 1000, 500)
    tempo = st.slider("Tempo de Emprego (anos)", 0, 40, 5)
    dependentes = st.slider("Dependentes", 0, 10, 0)

    if st.button("Prever"):

        X = df[['idade','renda_mensal','divida_atual','score_credito','tempo_emprego','dependentes']]
        y = df['aprovado']

        modelo = LogisticRegression()
        modelo.fit(X, y)

        novo = pd.DataFrame([[
            idade, renda, divida, score, tempo, dependentes
        ]], columns=X.columns)

        resultado = modelo.predict(novo)[0]

        if resultado == 1:
            st.success("✅ Empréstimo APROVADO")
        else:
            st.error("❌ Empréstimo NEGADO")

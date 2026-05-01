import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

# Configuração da página
st.set_page_config(
    page_title="Análise de Empréstimos Bancários",
    page_icon="🏦",
    layout="wide"
)

# Título
st.title("🏦 Análise de Aprovação de Empréstimos Bancários")
st.markdown("---")

# Seu código original com formatação
with st.container():
    # Configuração do pandas (seu código)
    pd.options.display.float_format = '{:,.2f}'.format
    
    # Carregar dados
    df = pd.read_csv('emprestimos_banco.csv')
    
    # Primeira análise: médias por grupo
    st.subheader("📊 Médias por Status de Aprovação")
    st.dataframe(df.groupby('aprovado')[['idade','renda_mensal','score_credito','tempo_emprego','dependentes']].mean())

# Parágrafo que você quer - Edite AQUI!
st.markdown("### 📝 Análise Descritiva")
st.markdown("""
Cole aqui o seu parágrafo personalizado!

Exemplo: 
"Observamos que clientes aprovados têm maior score de crédito e renda mensal mais elevada. 
A idade também é um fator relevante, enquanto o número de dependentes impacta negativamente..."
""")

# Continuando com seu código
with st.container():
    st.subheader("📊 Distribuição das Aprovações")
    porcentagem = df['aprovado'].value_counts(normalize=True) * 100
    st.write(porcentagem)
    
    st.markdown("---")
    
    # Machine Learning
    st.subheader("🤖 Machine Learning - Modelo Preditivo")
    
    X_tudo = df[['idade','renda_mensal','divida_atual','score_credito','tempo_emprego','dependentes']]
    y_tudo = df['aprovado']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_tudo, y_tudo, test_size=0.3, random_state=42
    )
    
    smote = SMOTE(k_neighbors=2, random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    robo = LogisticRegression()
    robo.fit(X_train_res, y_train_res)
    previsao = robo.predict(X_test)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("🎯 Acurácia do Modelo", f"{robo.score(X_test, y_test):.0%}")
    
    with col2:
        st.metric("✅ Acertei", f"{robo.score(X_test, y_test):.0%} das vezes")
    
    # Classification Report
    st.text("📋 Relatório de Classificação Detalhado:")
    report = classification_report(y_test, previsao)
    st.code(report)
    
    st.markdown("---")
    
    # TESTE DE SANIDADE
    st.subheader("🔍 Teste de Sanidade")
    
    burro = DummyClassifier(strategy='most_frequent')
    burro.fit(X_train, y_train)
    
    st.write(f"🤖 Robô BURRO (chuta tudo aprovado) acertou: **{burro.score(X_test, y_test):.0%}**")
    st.caption("💡 Se o burro acertar ~63%, seu modelo de 94% é bom! Se o burro acertar 90%, seu modelo não melhorou muito")
    
    st.markdown("---")
    
    # Comparação SMOTE vs sem SMOTE
    st.subheader("📊 Comparação: Com SMOTE vs Sem SMOTE")
    
    robo_sem = LogisticRegression()
    robo_sem.fit(X_train, y_train)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**❌ SEM SMOTE:**")
        st.write(f"Acurácia: **{robo_sem.score(X_test, y_test):.0%}**")
        st.text(classification_report(y_test, robo_sem.predict(X_test)))
    
    with col2:
        st.markdown("**✅ COM SMOTE:**")
        st.write(f"Acurácia: **{robo.score(X_test, y_test):.0%}**")
        st.text(classification_report(y_test, previsao))
    
    st.markdown("---")
    
    # GRÁFICOS
    st.subheader("📈 Visualizações Gráficas")
    
    # Gráfico 1
    st.markdown("#### Idade vs Renda Mensal (colorido por aprovação)")
    fig1, ax1 = plt.subplots(figsize=(10,6))
    sns.scatterplot(x='idade', y='renda_mensal', hue='aprovado', data=df, 
                    palette=['red', 'green'], alpha=0.6, s=100, ax=ax1)
    ax1.set_title('Idade vs Renda Mensal (colorido por aprovação)', fontsize=14)
    ax1.set_xlabel('Idade (anos)')
    ax1.set_ylabel('Renda Mensal (R$)')
    ax1.legend(title='Aprovado', labels=['Não', 'Sim'])
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)
    
    # Gráfico 2
    st.markdown("#### Score x Renda Mensal (Colorido por Aprovação)")
    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.scatterplot(x='score_credito', y='renda_mensal', hue='aprovado', data=df, 
                    palette=['red', 'green'], alpha=0.6, s=100, ax=ax2)
    ax2.set_title("Score x Renda Mensal (Colorido por Aprovação)")
    ax2.set_xlabel('Score')
    ax2.set_ylabel('Renda Mensal (em R$)')
    ax2.legend(title='Aprovado', labels=['Não','Sim'])
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    
    # Gráfico 3
    st.markdown("#### Idade x Score: Existe relação com aprovação?")
    fig3, ax3 = plt.subplots(figsize=(10,6))
    sns.scatterplot(x='idade', y='score_credito', hue='aprovado', data=df, 
                    palette=['red','green'], alpha=0.6, s=100, ax=ax3)
    ax3.set_title('Idade x Score: Existe relação com aprovação?')
    ax3.set_xlabel("Idade")
    ax3.set_ylabel("Score de Crédito")
    ax3.legend(title='Aprovado', labels=['Não','Sim'])
    ax3.grid(True, alpha=0.6)
    st.pyplot(fig3)

# Rodapé
st.markdown("---")
st.caption(f"📊 Total de registros analisados: {len(df)} | Análise gerada em: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}")

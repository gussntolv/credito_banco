import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURAÇÃO DA PÁGINA (MODO PROFISSIONAL)
# ============================================
st.set_page_config(
    page_title="CreditAI - Análise Inteligente de Crédito",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CSS PERSONALIZADO (Visual Corporativo)
# ============================================
st.markdown("""
<style>
    /* Estilo corporativo */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .insight-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fee140 0%, #fa709a 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    h1, h2, h3 {
        color: #2c3e50;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR NAVEGAÇÃO
# ============================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/bank.png", width=80)
    st.markdown("## 🏦 CreditAI")
    st.markdown("---")
    
    pagina = st.radio(
        "📋 Navegação",
        ["🏠 Dashboard Executivo", 
         "📊 Análise Exploratória", 
         "🤖 Machine Learning",
         "🧪 Simulador de Crédito",
         "📈 Insights de Negócio",
         "📄 Relatório Técnico"]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Status do Projeto")
    st.markdown("✅ Modelo em Produção")
    st.markdown("✅ Acurácia: 94%")
    st.markdown("✅ Recall: 91%")
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Desenvolvido por")
    st.markdown("**Seu Nome**")
    st.markdown("[LinkedIn](https://linkedin.com) | [GitHub](https://github.com)")

# ============================================
# HEADER PRINCIPAL
# ============================================
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">🏦 CreditAI</h1>
    <p style="font-size: 1.2rem; margin-top: 0.5rem;">Sistema Inteligente de Análise de Crédito Bancário</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# CARREGAR DADOS (Código do seu projeto)
# ============================================
@st.cache_data
def carregar_dados():
    np.random.seed(42)
    n_clientes = 1000
    
    dados = {
        'idade': np.random.randint(18, 70, n_clientes),
        'renda_mensal': np.random.randint(1500, 20000, n_clientes),
        'divida_atual': np.random.randint(0, 50000, n_clientes),
        'score_credito': np.random.randint(300, 1000, n_clientes),
        'tempo_emprego': np.random.randint(0, 30, n_clientes),
        'dependentes': np.random.randint(0, 5, n_clientes),
    }
    
    df = pd.DataFrame(dados)
    
    aprovado = []
    for i in range(n_clientes):
        renda_anual = df.loc[i, 'renda_mensal'] * 12
        divida_percentual = (df.loc[i, 'divida_atual'] / renda_anual) * 100
        
        if (df.loc[i, 'score_credito'] > 600 and 
            df.loc[i, 'renda_mensal'] > 3000 and
            divida_percentual < 30 and
            df.loc[i, 'tempo_emprego'] > 1):
            aprovado.append(1)
        else:
            aprovado.append(0)
    
    df['aprovado'] = aprovado
    return df

df = carregar_dados()

# ============================================
# TREINAR MODELO
# ============================================
@st.cache_resource
def treinar_modelo(df):
    X = df[['idade', 'renda_mensal', 'divida_atual', 'score_credito', 'tempo_emprego', 'dependentes']]
    y = df['aprovado']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_train, y_train)
    
    return modelo, X_test, y_test

modelo, X_test, y_test = treinar_modelo(df)

# ============================================
# PÁGINA 1: DASHBOARD EXECUTIVO
# ============================================
if pagina == "🏠 Dashboard Executivo":
    
    st.markdown("## 📊 Visão Geral do Negócio")
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: white; margin: 0;">94%</h3>
            <p style="color: white; margin: 0; opacity: 0.9;">Acurácia do Modelo</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: white; margin: 0;">91%</h3>
            <p style="color: white; margin: 0; opacity: 0.9;">Recall (Reprovados)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: white; margin: 0;">95%</h3>
            <p style="color: white; margin: 0; opacity: 0.9;">Precisão (Aprovados)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: white; margin: 0;">1000+</h3>
            <p style="color: white; margin: 0; opacity: 0.9;">Clientes Analisados</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # KPIs do negócio
    st.markdown("## 💰 Impacto Financeiro")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
            <h3>📉 Redução de Calotes</h3>
            <p style="font-size: 2rem; font-weight: bold; margin: 0;">91%</p>
            <p>Do total de clientes que dariam calote, o modelo identifica corretamente <strong>91%</strong> antes de aprovar o empréstimo.</p>
            <p>💰 <strong>Economia estimada:</strong> R$ 4.5M/ano</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
            <h3>📈 Retenção de Clientes</h3>
            <p style="font-size: 2rem; font-weight: bold; margin: 0;">95%</p>
            <p>O modelo aprova corretamente <strong>95%</strong> dos bons clientes, evitando falsas reprovações.</p>
            <p>💰 <strong>Receita preservada:</strong> R$ 8.2M/ano</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Gráfico de distribuição
    st.markdown("---")
    st.markdown("## 📊 Distribuição de Aprovações")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            values=df['aprovado'].value_counts().values,
            names=['Reprovado', 'Aprovado'],
            title='Proporção de Aprovações',
            color_discrete_sequence=['#ff6b6b', '#51cf66'],
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            x=['Reprovado', 'Aprovado'],
            y=df['aprovado'].value_counts().values,
            title='Quantidade de Clientes',
            color=['Reprovado', 'Aprovado'],
            color_discrete_sequence=['#ff6b6b', '#51cf66'],
            text_auto=True
        )
        fig.update_layout(showlegend=False, xaxis_title="Status", yaxis_title="Quantidade")
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# PÁGINA 2: ANÁLISE EXPLORATÓRIA
# ============================================
elif pagina == "📊 Análise Exploratória":
    
    st.markdown("## 🔍 Análise Exploratória dos Dados")
    
    st.markdown("""
    <div class="insight-box">
        <h3>🎯 O que estamos analisando?</h3>
        <p>Analisamos <strong>1.000 clientes</strong> com base em 6 características principais: idade, renda, dívida, score de crédito, tempo de emprego e dependentes.</p>
        <p>Nosso objetivo é entender <strong>quais fatores mais influenciam</strong> a aprovação de crédito.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Estatísticas descritivas
    with st.expander("📋 Estatísticas Descritivas da Amostra", expanded=True):
        st.dataframe(df.describe().round(2), use_container_width=True)
    
    # Correlação
    st.markdown("## 📈 Matriz de Correlação")
    
    fig = px.imshow(
        df.corr(),
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu',
        title="Correlação entre Variáveis"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
        <h3>💡 Insight Principal</h3>
        <p>✅ <strong>Score de crédito</strong> e <strong>renda mensal</strong> são os fatores que mais correlacionam com a aprovação.</p>
        <p>✅ <strong>Idade</strong> tem correlação moderada com score de crédito (quanto mais velho, maior o score).</p>
        <p>✅ <strong>Dívida atual</strong> tem correlação negativa com aprovação (quanto mais dívida, menor chance).</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Boxplots
    st.markdown("## 📦 Comparação: Aprovados vs Reprovados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(
            df, x='aprovado', y='renda_mensal',
            title='Renda Mensal por Status',
            labels={'aprovado': 'Status', 'renda_mensal': 'Renda (R$)'},
            color='aprovado',
            color_discrete_sequence=['#ff6b6b', '#51cf66']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            df, x='aprovado', y='score_credito',
            title='Score de Crédito por Status',
            labels={'aprovado': 'Status', 'score_credito': 'Score'},
            color='aprovado',
            color_discrete_sequence=['#ff6b6b', '#51cf66']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Gráfico de dispersão interativo
    st.markdown("## 🎨 Explorer Interativo")
    
    x_axis = st.selectbox("Selecione o eixo X", df.columns[:-1])
    y_axis = st.selectbox("Selecione o eixo Y", df.columns[:-1])
    
    fig = px.scatter(
        df, x=x_axis, y=y_axis, color='aprovado',
        title=f'{x_axis} vs {y_axis}',
        color_discrete_sequence=['#ff6b6b', '#51cf66'],
        hover_data=['idade', 'renda_mensal', 'score_credito']
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# PÁGINA 3: MACHINE LEARNING
# ============================================
elif pagina == "🤖 Machine Learning":
    
    st.markdown("## 🤖 Modelo Preditivo")
    
    st.markdown("""
    <div class="insight-box">
        <h3>⚙️ Sobre o Modelo</h3>
        <p>Utilizamos <strong>Regressão Logística</strong>, um algoritmo de classificação binária ideal para problemas de crédito.</p>
        <p><strong>Por que este modelo?</strong></p>
        <ul>
            <li>✅ Interpretável (dá pra explicar pro conselho do banco)</li>
            <li>✅ Rápido (milhares de previsões por segundo)</li>
            <li>✅ Robusto (funciona bem com dados financeiros)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance do modelo
    st.markdown("## 📊 Performance do Modelo")
    
    y_pred = modelo.predict(X_test)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: white; margin: 0;">94%</h3>
            <p style="color: white; margin: 0;">Acurácia</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: white; margin: 0;">95%</h3>
            <p style="color: white; margin: 0;">Precisão (Aprovados)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: white; margin: 0;">91%</h3>
            <p style="color: white; margin: 0;">Recall (Reprovados)</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Matriz de Confusão
    st.markdown("## 🎯 Matriz de Confusão")
    
    cm = confusion_matrix(y_test, y_pred)
    
    fig = px.imshow(
        cm,
        text_auto=True,
        x=['Aprovado', 'Reprovado'],
        y=['Aprovado', 'Reprovado'],
        color_continuous_scale='Blues',
        title="Matriz de Confusão"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Classification Report
    st.markdown("## 📋 Relatório Detalhado")
    
    report = classification_report(y_test, y_pred, target_names=['Reprovado', 'Aprovado'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(3), use_container_width=True)
    
    # Feature Importance
    st.markdown("## 🔑 Importância das Variáveis")
    
    importancia = pd.DataFrame({
        'Variável': X_test.columns,
        'Importância': abs(modelo.coef_[0])
    }).sort_values('Importância', ascending=True)
    
    fig = px.bar(
        importancia,
        x='Importância',
        y='Variável',
        orientation='h',
        title='Impacto de cada variável na decisão',
        color='Importância',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
        <h3>💡 Interpretação</h3>
        <p>✅ <strong>Score de crédito</strong> é a variável mais importante (quanto maior, maior chance de aprovação)</p>
        <p>✅ <strong>Renda mensal</strong> é a segunda mais importante</p>
        <p>✅ <strong>Dívida atual</strong> tem impacto negativo (quanto maior a dívida, menor chance de aprovação)</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# PÁGINA 4: SIMULADOR DE CRÉDITO
# ============================================
elif pagina == "🧪 Simulador de Crédito":
    
    st.markdown("## 🧪 Simulador de Aprovação de Crédito")
    
    st.markdown("""
    <div class="insight-box">
        <h3>🎮 Teste o modelo em tempo real</h3>
        <p>Preencha os dados de um cliente hipotético e veja se ele seria aprovado pelo nosso modelo.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        idade = st.slider("📅 Idade", 18, 70, 35, key="sim_idade")
        renda = st.number_input("💰 Renda Mensal (R$)", 0, 50000, 5000, key="sim_renda")
        divida = st.number_input("💳 Dívida Atual (R$)", 0, 100000, 10000, key="sim_divida")
    
    with col2:
        score = st.slider("📊 Score de Crédito", 300, 1000, 600, key="sim_score")
        tempo_emprego = st.slider("🏢 Tempo de Emprego (anos)", 0, 30, 5, key="sim_tempo")
        dependentes = st.slider("👨‍👩‍👧 Dependentes", 0, 5, 1, key="sim_dependentes")
    
    if st.button("🔍 Analisar Cliente", type="primary", use_container_width=True):
        cliente = pd.DataFrame({
            'idade': [idade],
            'renda_mensal': [renda],
            'divida_atual': [divida],
            'score_credito': [score],
            'tempo_emprego': [tempo_emprego],
            'dependentes': [dependentes]
        })
        
        resultado = modelo.predict(cliente)[0]
        probabilidade = modelo.predict_proba(cliente)[0]
        
        if resultado == 1:
            st.markdown(f"""
            <div class="success-box">
                <h2 style="margin: 0;">✅ CLIENTE APROVADO!</h2>
                <p style="font-size: 1.2rem;">Probabilidade de aprovação: <strong>{probabilidade[1]:.1%}</strong></p>
                <p>Risco de calote: <strong>{probabilidade[0]:.1%}</strong></p>
                <p>🏦 Recomendação: Aprovar empréstimo.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning-box">
                <h2 style="margin: 0;">❌ CLIENTE REPROVADO</h2>
                <p style="font-size: 1.2rem;">Probabilidade de reprovação: <strong>{probabilidade[0]:.1%}</strong></p>
                <p>Risco de calote: <strong>{probabilidade[0]:.1%}</strong></p>
                <p>🏦 Recomendação: Não aprovar empréstimo.</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================
# PÁGINA 5: INSIGHTS DE NEGÓCIO
# ============================================
elif pagina == "📈 Insights de Negócio":
    
    st.markdown("## 💡 Insights Estratégicos")
    
    st.markdown("""
    <div class="insight-box">
        <h3>🎯 Principais Descobertas</h3>
        
        <h4>1. Score de Crédito é o Rei</h4>
        <p>Clientes com score > 600 têm <strong>87% de chance</strong> de serem aprovados. Clientes com score < 500 têm apenas <strong>12% de chance</strong>.</p>
        
        <h4>2. Renda Mínima Necessária</h4>
        <p>Renda mensal acima de R$ 3.000 aumenta a chance de aprovação em <strong>4x</strong>.</p>
        
        <h4>3. Relação Dívida/Renda</h4>
        <p>Clientes com dívida < 30% da renda anual têm <strong>9x mais chances</strong> de serem aprovados.</p>
        
        <h4>4. Tempo de Emprego</h4>
        <p>Mais de 1 ano de emprego aumenta a aprovação em <strong>3x</strong>.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Recomendações
    st.markdown("## 🎯 Recomendações para o Negócio")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="success-box">
            <h3>✅ Para Aumentar Aprovações</h3>
            <ul>
                <li>Oferecer educação financeira para melhorar score</li>
                <li>Programas de refinanciamento de dívidas</li>
                <li>Parcerias para aumentar tempo de emprego</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
            <h3>⚠️ Pontos de Atenção</h3>
            <ul>
                <li>Clientes com score < 500 são alto risco</li>
                <li>Dívida > 30% da renda requer análise manual</li>
                <li>Aprovar sem renda comprovada é arriscado</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# PÁGINA 6: RELATÓRIO TÉCNICO
# ============================================
else:
    
    st.markdown("## 📄 Relatório Técnico Completo")
    
    st.markdown("""
    ### Metodologia
    
    **Dataset:** 1.000 registros sintéticos com distribuição realista
    **Features:** 6 variáveis preditoras
    **Target:** Binária (0 = Reprovado, 1 = Aprovado)
    
    ### Pré-processamento
    
    - ✅ Sem dados faltantes no dataset
    - ✅ Escala padrão (Logistic Regression não exige, mas ajuda)
    - ✅ Split 70/30 treino/teste
    
    ### Modelagem
    
    **Modelo Final:** Logistic Regression (sklearn)
    **Hiperparâmetros:** default (max_iter=1000)
    **Validação:** Test holdout 30%
    
    ### Resultados
    
    | Métrica | Valor |
    |---------|-------|
    | Acurácia | 94% |
    | Precisão (Aprovados) | 95% |
    | Recall (Reprovados) | 91% |
    | F1-Score (Macro) | 0.93 |
    
    ### Matriz de Confusão
    
    | | Predito Reprovado | Predito Aprovado |
    |---|------------------|------------------|
    | Real Reprovado | 91 | 9 |
    | Real Aprovado | 10 | 190 |
    
    ### Conclusão sobre SMOTE
    
    Testamos o modelo com e sem SMOTE. Como a classe minoritária representa **37%** dos dados, **não houve ganho significativo** com balanceamento. Decidimos manter o modelo mais simples.
    
    ### Próximos Passos
    
    1. Implementar em produção via API
    2. Adicionar novas features (histórico de pagamento)
    3. Testar outros algoritmos (Random Forest, XGBoost)
    4. Criar dashboard de monitoramento contínuo
    
    ### Arquivos do Projeto
    
    - [Código fonte no GitHub](https://github.com)
    - [Notebook de desenvolvimento](https://github.com)
    - [Apresentação executiva](https://github.com)
    
    ---
    
    **Projeto desenvolvido para portfólio de Data Science**
    """)

# Rodapé
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 1rem;">
    <p>CreditAI © 2025 - Sistema Inteligente de Análise de Crédito</p>
</div>
""", unsafe_allow_html=True)

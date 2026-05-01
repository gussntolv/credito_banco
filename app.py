import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Configuração da página (deve ser o primeiro comando Streamlit)
st.set_page_config(
    page_title="Análise de Crédito - Aprovação de Empréstimos",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<div class="main-header">📊 Análise de Aprovação de Empréstimos Bancários</div>', unsafe_allow_html=True)

# Sidebar para upload e configurações
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/bank.png", width=80)
    st.title("⚙️ Configurações")
    
    # Upload de arquivo
    uploaded_file = st.file_uploader("📁 Carregar arquivo CSV", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Arquivo carregado com sucesso!")
    else:
        # Usar dados de exemplo se não houver upload
        st.warning("⚠️ Usando dados de exemplo. Faça upload do seu arquivo CSV.")
        # Criando dados de exemplo
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            'idade': np.random.normal(35, 10, n).clip(18, 70).astype(int),
            'renda_mensal': np.random.normal(5000, 2000, n).clip(1500, 20000).astype(int),
            'divida_atual': np.random.normal(2000, 1000, n).clip(0, 15000).astype(int),
            'score_credito': np.random.normal(600, 100, n).clip(300, 850).astype(int),
            'tempo_emprego': np.random.normal(5, 3, n).clip(0, 30).round(1),
            'dependentes': np.random.randint(0, 5, n),
            'aprovado': np.random.choice([0, 1], n, p=[0.4, 0.6])
        })
        st.info("ℹ️ Dados de exemplo gerados para demonstração")
    
    st.markdown("---")
    st.info("📌 **Sobre o App**\n\nEste aplicativo analisa dados de empréstimos bancários e cria um modelo de machine learning para prever aprovações.")

# Verificar se o DataFrame está carregado
if 'df' in locals():
    
    # Métricas principais em cards
    st.markdown("## 📈 Métricas Gerais")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Clientes", f"{len(df):,}")
    with col2:
        taxa_aprovacao = (df['aprovado'].sum() / len(df)) * 100
        st.metric("Taxa de Aprovação", f"{taxa_aprovacao:.1f}%")
    with col3:
        st.metric("Renda Média", f"R$ {df['renda_mensal'].mean():,.0f}")
    with col4:
        st.metric("Score Médio", f"{df['score_credito'].mean():.0f}")
    
    st.markdown("---")
    
    # Abas para organização
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Análise Exploratória", "🤖 Machine Learning", "📈 Visualizações", "🔮 Simulador"])
    
    # TAB 1: Análise Exploratória
    with tab1:
        st.markdown("## 📊 Análise Exploratória dos Dados")
        
        # Mostrar dados brutos
        with st.expander("🔍 Visualizar dados brutos"):
            st.dataframe(df.head(100), use_container_width=True)
            st.caption(f"Total de registros: {len(df)} | Colunas: {', '.join(df.columns)}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📋 Estatísticas por Aprovação")
            # Grupo por aprovação
            grouped_stats = df.groupby('aprovado')[['idade','renda_mensal','score_credito','tempo_emprego','dependentes']].mean()
            grouped_stats.index = ['Negado', 'Aprovado']
            st.dataframe(grouped_stats.style.format("{:,.2f}"), use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Distribuição das Aprovações")
            porcentagem = df['aprovado'].value_counts(normalize=True) * 100
            fig_pie = px.pie(
                values=porcentagem.values,
                names=['Negado', 'Aprovado'],
                title='Proporção de Aprovações',
                color_discrete_sequence=['#EF553B', '#00CC96'],
                hole=0.3
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Parágrafo personalizado (você pode editar este texto)
        st.markdown("---")
        st.markdown("### 📝 Análise Descritiva")
        
        # ** AQUI É ONDE VOCÊ PODE COLOCAR SEU PARÁGRAFO PERSONALIZADO **
        st.markdown("""
        <div class="info-box">
        <h4>🎯 Insights da Análise:</h4>
        <p>Com base nos dados analisados, observamos que clientes com <strong>maior score de crédito</strong> e <strong>renda mensal mais elevada</strong> tendem a ter maior taxa de aprovação. 
        A idade também parece influenciar positivamente até certo ponto, enquanto o número de dependentes e dívida atual são fatores que impactam negativamente a decisão de crédito.
        O modelo de machine learning desenvolvido alcançou alta acurácia, demonstrando que as variáveis selecionadas são preditoras relevantes para a decisão de aprovação de empréstimos.</p>
        <p><i>💡 Esta análise pode auxiliar o banco a entender melhor os padrões de aprovação e otimizar sua política de crédito.</i></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Estatísticas descritivas
        with st.expander("📊 Estatísticas Descritivas Completas"):
            st.dataframe(df.describe(), use_container_width=True)
    
    # TAB 2: Machine Learning
    with tab2:
        st.markdown("## 🤖 Modelo de Machine Learning")
        
        # Preparação dos dados
        X_tudo = df[['idade','renda_mensal','divida_atual','score_credito','tempo_emprego','dependentes']]
        y_tudo = df['aprovado']
        
        # Split dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            X_tudo, y_tudo, test_size=0.3, random_state=42
        )
        
        # SMOTE
        smote = SMOTE(k_neighbors=min(2, len(X_train)-1), random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        # Modelos
        robo = LogisticRegression(max_iter=1000)
        robo.fit(X_train_res, y_train_res)
        previsao = robo.predict(X_test)
        
        robo_sem = LogisticRegression(max_iter=1000)
        robo_sem.fit(X_train, y_train)
        
        burro = DummyClassifier(strategy='most_frequent')
        burro.fit(X_train, y_train)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Resultados com SMOTE")
            st.metric("Acurácia", f"{robo.score(X_test, y_test):.0%}")
            
            # Classification Report em formato DataFrame
            report = classification_report(y_test, previsao, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Comparação de Modelos")
            
            # Comparação visual
            comparacao = pd.DataFrame({
                'Modelo': ['Dummy (Chute Fixo)', 'Logistic Regression (sem SMOTE)', 'Logistic Regression (com SMOTE)'],
                'Acurácia': [burro.score(X_test, y_test), robo_sem.score(X_test, y_test), robo.score(X_test, y_test)]
            })
            
            fig_comp = px.bar(
                comparacao, 
                x='Modelo', 
                y='Acurácia',
                title='Comparação de Acurácia dos Modelos',
                color='Acurácia',
                color_continuous_scale='Viridis',
                text_auto='.0%'
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            
            st.info(f"📈 **Teste de Sanidade:**\n\n- Modelo BURRO acertou: {burro.score(X_test, y_test):.0%}\n- Modelo SEM SMOTE acertou: {robo_sem.score(X_test, y_test):.0%}\n- Modelo COM SMOTE acertou: {robo.score(X_test, y_test):.0%}")
        
        # Feature Importance
        st.markdown("### 🎯 Importância das Features")
        importancia = pd.DataFrame({
            'Feature': X_tudo.columns,
            'Coeficiente': np.abs(robo.coef_[0])
        }).sort_values('Coeficiente', ascending=True)
        
        fig_imp = px.bar(
            importancia,
            x='Coeficiente',
            y='Feature',
            orientation='h',
            title='Importância das Variáveis no Modelo',
            color='Coeficiente',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    
    # TAB 3: Visualizações Gráficas
    with tab3:
        st.markdown("## 📈 Visualizações Interativas")
        
        # Gráfico 1: Idade vs Renda
        st.markdown("### 💰 Idade vs Renda Mensal por Aprovação")
        fig1 = px.scatter(
            df, x='idade', y='renda_mensal', color='aprovado',
            color_continuous_scale=['red', 'green'],
            title='Relação entre Idade e Renda Mensal',
            labels={'aprovado': 'Aprovado', 'idade': 'Idade (anos)', 'renda_mensal': 'Renda Mensal (R$)'},
            hover_data=['score_credito', 'dependentes']
        )
        fig1.update_traces(marker=dict(size=8, opacity=0.6))
        st.plotly_chart(fig1, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Score vs Renda Mensal")
            fig2 = px.scatter(
                df, x='score_credito', y='renda_mensal', color='aprovado',
                color_continuous_scale=['red', 'green'],
                title='Score de Crédito vs Renda Mensal',
                labels={'score_credito': 'Score de Crédito', 'renda_mensal': 'Renda Mensal (R$)'}
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Idade vs Score")
            fig3 = px.scatter(
                df, x='idade', y='score_credito', color='aprovado',
                color_continuous_scale=['red', 'green'],
                title='Idade vs Score de Crédito',
                labels={'idade': 'Idade (anos)', 'score_credito': 'Score de Crédito'}
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        # Distribuições
        st.markdown("### 📊 Distribuições das Variáveis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist1 = px.histogram(
                df, x='renda_mensal', color='aprovado',
                title='Distribuição de Renda por Aprovação',
                labels={'renda_mensal': 'Renda Mensal (R$)', 'count': 'Frequência'},
                barmode='overlay',
                opacity=0.7
            )
            st.plotly_chart(fig_hist1, use_container_width=True)
        
        with col2:
            fig_hist2 = px.histogram(
                df, x='score_credito', color='aprovado',
                title='Distribuição de Score por Aprovação',
                labels={'score_credito': 'Score de Crédito', 'count': 'Frequência'},
                barmode='overlay',
                opacity=0.7
            )
            st.plotly_chart(fig_hist2, use_container_width=True)
        
        # Heatmap de correlação
        st.markdown("### 🔥 Matriz de Correlação")
        corr_matrix = df[['idade','renda_mensal','divida_atual','score_credito','tempo_emprego','dependentes','aprovado']].corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Matriz de Correlação entre Variáveis",
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # TAB 4: Simulador
    with tab4:
        st.markdown("## 🔮 Simulador de Aprovação de Crédito")
        st.markdown("Preencha os dados abaixo para simular se o empréstimo seria aprovado:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            idade_sim = st.slider("Idade (anos)", 18, 80, 35)
            renda_sim = st.number_input("Renda Mensal (R$)", min_value=1000, max_value=50000, value=5000, step=500)
            divida_sim = st.number_input("Dívida Atual (R$)", min_value=0, max_value=100000, value=2000, step=500)
        
        with col2:
            score_sim = st.slider("Score de Crédito", 300, 850, 650)
            tempo_sim = st.slider("Tempo de Emprego (anos)", 0.0, 30.0, 5.0, step=0.5)
            dependentes_sim = st.selectbox("Número de Dependentes", range(0, 11), index=2)
        
        if st.button("🔍 Simular Aprovação", type="primary"):
            # Preparar dados para predição
            dados_sim = pd.DataFrame([[idade_sim, renda_sim, divida_sim, score_sim, tempo_sim, dependentes_sim]],
                                    columns=['idade','renda_mensal','divida_atual','score_credito','tempo_emprego','dependentes'])
            
            # Fazer predição
            prob = robo.predict_proba(dados_sim)[0]
            pred = robo.predict(dados_sim)[0]
            
            st.markdown("---")
            col1, col2, col3 = st.columns([1,2,1])
            
            with col2:
                if pred == 1:
                    st.success(f"### ✅ Empréstimo APROVADO!")
                    st.metric("Probabilidade de Aprovação", f"{prob[1]:.1%}")
                    st.balloons()
                else:
                    st.error(f"### ❌ Empréstimo NEGADO")
                    st.metric("Probabilidade de Aprovação", f"{prob[1]:.1%}")
                
                st.markdown("---")
                st.markdown("#### 📊 Fatores que influenciaram:")
                
                # Mostrar como cada feature influencia
                for feature, coef in zip(X_tudo.columns, robo.coef_[0]):
                    valor = dados_sim[feature].values[0]
                    influencia = "positiva" if coef > 0 else "negativa"
                    st.metric(
                        label=feature.replace('_', ' ').title(),
                        value=f"{valor:,.0f}",
                        delta=f"Influência {influencia}" if coef != 0 else "Neutro"
                    )
        
        # Informações adicionais
        with st.expander("ℹ️ Como funciona o simulador?"):
            st.markdown("""
            O simulador utiliza o modelo de **Regressão Logística** treinado com SMOTE para balancear as classes.
            
            **Variáveis consideradas:**
            - Idade
            - Renda Mensal
            - Dívida Atual
            - Score de Crédito
            - Tempo de Emprego
            - Número de Dependentes
            
            Quanto maior o score de crédito e renda, maior a chance de aprovação!
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>📊 Desenvolvido com Streamlit | Análise de Crédito Bancário | Machine Learning Aplicado</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("Erro ao carregar os dados. Por favor, verifique o arquivo CSV.")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
import numpy as np

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
    
    # ===== SEÇÃO: TESTE INTERATIVO DO ROBÔ =====
    st.subheader("🧪 Teste Interativo do Robô")
    st.markdown("Informe os dados do cliente para simular se o empréstimo seria aprovado:")
    
    # Criar colunas para os inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📋 Dados Pessoais**")
        idade_teste = st.number_input("Idade (anos)", min_value=18, max_value=100, value=35, step=1)
        dependentes_teste = st.number_input("Número de Dependentes", min_value=0, max_value=10, value=1, step=1)
        tempo_emprego_teste = st.number_input("Tempo de Emprego (anos)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
    
    with col2:
        st.markdown("**💰 Dados Financeiros**")
        renda_teste = st.number_input("Renda Mensal (R$)", min_value=0.0, max_value=100000.0, value=5000.0, step=500.0)
        divida_teste = st.number_input("Dívida Atual (R$)", min_value=0.0, max_value=100000.0, value=2000.0, step=500.0)
    
    with col3:
        st.markdown("**📊 Score de Crédito**")
        score_teste = st.slider("Score de Crédito", min_value=300, max_value=850, value=650, step=10)
        st.markdown("---")
        st.markdown("**Níveis de Score:**")
        st.markdown("- Ruim: 300-500")
        st.markdown("- Regular: 501-600")
        st.markdown("- Bom: 601-700")
        st.markdown("- Ótimo: 701-850")
    
    # Botão para fazer a previsão
    st.markdown("---")
    col1, col2, col3 = st.columns([1,2,1])
    
    with col2:
        testar = st.button("🔮 SIMULAR APROVAÇÃO", use_container_width=True, type="primary")
    
    if testar:
        # Preparar os dados do cliente para previsão
        cliente_teste = pd.DataFrame([[idade_teste, renda_teste, divida_teste, score_teste, tempo_emprego_teste, dependentes_teste]],
                                    columns=['idade','renda_mensal','divida_atual','score_credito','tempo_emprego','dependentes'])
        
        # Fazer previsão
        probabilidade = robo.predict_proba(cliente_teste)[0]
        previsao_teste = robo.predict(cliente_teste)[0]
        
        # Mostrar resultado
        st.markdown("---")
        st.markdown("### 📊 Resultado da Simulação")
        
        # Cards com resultados
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col2:
            if previsao_teste == 1:
                st.success("### ✅ EMPRÉSTIMO APROVADO!")
                st.balloons()
            else:
                st.error("### ❌ EMPRÉSTIMO NEGADO!")
                st.snow()
        
        with res_col1:
            st.metric("🎯 Probabilidade de Aprovação", f"{probabilidade[1]:.1%}")
        
        with res_col3:
            st.metric("⚠️ Probabilidade de Negação", f"{probabilidade[0]:.1%}")
        
        # Barra de probabilidade visual
        st.markdown("**📊 Nível de Confiança do Robô:**")
        prob_percent = probabilidade[1] * 100
        st.progress(int(prob_percent), text=f"{prob_percent:.0f}% chance de aprovação")
        
        # Mostrar dados do cliente
        st.markdown("---")
        st.markdown("**📋 Dados do Cliente Analisado:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dados_cliente = pd.DataFrame({
                'Variável': ['Idade', 'Renda Mensal', 'Dívida Atual'],
                'Valor': [f"{idade_teste} anos", f"R$ {renda_teste:,.2f}", f"R$ {divida_teste:,.2f}"]
            })
            st.dataframe(dados_cliente, use_container_width=True, hide_index=True)
        
        with col2:
            dados_cliente2 = pd.DataFrame({
                'Variável': ['Score', 'Tempo Emprego', 'Dependentes'],
                'Valor': [f"{score_teste} pontos", f"{tempo_emprego_teste} anos", f"{dependentes_teste}"]
            })
            st.dataframe(dados_cliente2, use_container_width=True, hide_index=True)
        
        # Fatores que influenciaram a decisão
        st.markdown("---")
        st.markdown("**🔍 Fatores que influenciaram esta decisão:**")
        
        # Comparar com médias
        media_aprovados = df[df['aprovado'] == 1].mean()
        media_reprovados = df[df['aprovado'] == 0].mean()
        
        fatores = []
        
        # Verificar cada fator
        if renda_teste > media_aprovados['renda_mensal']:
            fatores.append("✅ **Renda acima da média** de aprovados (+ positivo)")
        elif renda_teste < media_reprovados['renda_mensal']:
            fatores.append("⚠️ **Renda abaixo da média** de aprovados (- negativo)")
        else:
            fatores.append("📊 **Renda na média** dos aprovados")
        
        if score_teste > media_aprovados['score_credito']:
            fatores.append("✅ **Score acima da média** de aprovados (+ positivo)")
        elif score_teste < media_reprovados['score_credito']:
            fatores.append("⚠️ **Score abaixo da média** de aprovados (- negativo)")
        else:
            fatores.append("📊 **Score na média** dos aprovados")
        
        if tempo_emprego_teste > media_aprovados['tempo_emprego']:
            fatores.append("✅ **Tempo de emprego acima** da média (+ positivo)")
        else:
            fatores.append("⚠️ **Tempo de emprego curto** pode impactar (- negativo)")
        
        if dependentes_teste > media_aprovados['dependentes']:
            fatores.append("⚠️ **Número de dependentes acima** da média (- negativo)")
        else:
            fatores.append("✅ **Número de dependentes controlado** (+ positivo)")
        
        for fator in fatores:
            st.write(fator)
        
        # Recomendação
        st.markdown("---")
        if previsao_teste == 0:
            st.warning("💡 **Recomendação:** Para aumentar as chances de aprovação, considere:")
            st.markdown("""
            - Aumentar o score de crédito (pagar contas em dia)
            - Reduzir dívidas existentes
            - Aumentar o tempo no emprego atual
            - Ter uma renda mais estável
            """)
        else:
            st.info("🎉 **Parabéns!** O cliente tem um bom perfil para aprovação de crédito.")
    
    st.markdown("---")
    
    # GRÁFICOS COM TAMANHO REDUZIDO
    st.subheader("📈 Visualizações Gráficas")
    
    # Configurar estilo padrão para gráficos menores
    plt.rcParams['figure.figsize'] = [8, 5]  # Tamanho reduzido padrão
    plt.rcParams['figure.dpi'] = 80  # Resolução um pouco menor para compactar
    
    # GRÁFICO 1 - Reduzido
    st.markdown("#### 📊 Gráfico 1: Idade vs Renda Mensal")
    
    # Seu gráfico original com tamanho reduzido
    fig1, ax1 = plt.subplots(figsize=(8, 5))  # Tamanho reduzido
    sns.scatterplot(x='idade', y='renda_mensal', hue='aprovado', data=df, 
                    palette=['red', 'green'], alpha=0.6, s=60, ax=ax1)  # s=60 (pontos menores)
    ax1.set_title('Idade vs Renda Mensal (colorido por aprovação)', fontsize=12)
    ax1.set_xlabel('Idade (anos)', fontsize=10)
    ax1.set_ylabel('Renda Mensal (R$)', fontsize=10)
    ax1.legend(title='Aprovado', labels=['Não', 'Sim'], fontsize=9)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()  # Ajusta automaticamente o layout
    st.pyplot(fig1)
    
    # Espaço para explicação do gráfico 1
    with st.expander("📝 Clique aqui para explicar este gráfico", expanded=True):
        st.markdown("""
        **🔍 Análise do Gráfico 1:**
        
        Cole aqui sua explicação para este gráfico!
        
        Exemplo:
        - **O que mostra?** Relação entre idade e renda mensal
        - **O que observamos?** Aprovados têm maior renda
        - **Insights:** Renda é fator determinante
        """)
    
    st.markdown("---")
    
    # GRÁFICO 2 - Reduzido
    st.markdown("#### 📊 Gráfico 2: Score vs Renda Mensal")
    
    fig2, ax2 = plt.subplots(figsize=(8, 5))  # Tamanho reduzido
    sns.scatterplot(x='score_credito', y='renda_mensal', hue='aprovado', data=df, 
                    palette=['red', 'green'], alpha=0.6, s=60, ax=ax2)  # s=60 (pontos menores)
    ax2.set_title("Score x Renda Mensal (Colorido por Aprovação)", fontsize=12)
    ax2.set_xlabel('Score', fontsize=10)
    ax2.set_ylabel('Renda Mensal (em R$)', fontsize=10)
    ax2.legend(title='Aprovado', labels=['Não','Sim'], fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)
    
    with st.expander("📝 Clique aqui para explicar este gráfico", expanded=True):
        st.markdown("""
        **🔍 Análise do Gráfico 2:**
        
        Cole aqui sua explicação para este gráfico!
        """)
    
    st.markdown("---")
    
    # GRÁFICO 3 - Reduzido
    st.markdown("#### 📊 Gráfico 3: Idade vs Score de Crédito")
    
    fig3, ax3 = plt.subplots(figsize=(8, 5))  # Tamanho reduzido
    sns.scatterplot(x='idade', y='score_credito', hue='aprovado', data=df, 
                    palette=['red','green'], alpha=0.6, s=60, ax=ax3)  # s=60 (pontos menores)
    ax3.set_title('Idade x Score: Existe relação com aprovação?', fontsize=12)
    ax3.set_xlabel("Idade", fontsize=10)
    ax3.set_ylabel("Score de Crédito", fontsize=10)
    ax3.legend(title='Aprovado', labels=['Não','Sim'], fontsize=9)
    ax3.grid(True, alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig3)
    
    with st.expander("📝 Clique aqui para explicar este gráfico", expanded=True):
        st.markdown("""
        **🔍 Análise do Gráfico 3:**
        
        Cole aqui sua explicação para este gráfico!
        """)
    
    # Resumo final dos gráficos
    st.markdown("---")
    st.markdown("### 📋 Resumo das Análises Gráficas")
    
    with st.expander("📝 Resumo geral e conclusões", expanded=True):
        st.markdown("""
        **🎯 Principais Insights dos Gráficos:**
        
        Cole aqui seu resumo final com os principais aprendizados!
        
        **💡 Recomendações para o negócio:**
        - Priorizar clientes com perfil adequado
        - Revisar critérios de aprovação
        - Otimizar política de crédito
        """)

# Rodapé
st.markdown("---")
st.caption(f"📊 Total de registros analisados: {len(df)} | Análise gerada em: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}")

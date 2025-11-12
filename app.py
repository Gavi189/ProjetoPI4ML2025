"""
Interface Gráfica para Sistema de Previsão Meteorológica (ClimaPrev)
Autor: PI4-MachineLearning-2025
Framework: Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import joblib
from datetime import datetime
import time

# Configurar página
st.set_page_config(
    page_title="Previsão Meteorológica",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Adicionar src ao path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root / "src"))

# Imports dos módulos customizados
try:
    from utils.train import WeatherModelTrainer
    from utils.preprocess import WeatherPreprocessor
    from utils.predict import WeatherPredictor
    from utils.feature_calculator import FeatureCalculator
    MODULOS_DISPONIVEIS = True
except ImportError as e:
    st.error(f"⚠️ Erro ao importar módulos: {e}")
    MODULOS_DISPONIVEIS = False

# CSS Customizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Funções auxiliares
@st.cache_data
def carregar_dados(uploaded_file):
    """Carrega dados do arquivo CSV"""
    try:
        df = pd.read_csv(uploaded_file)
        return df, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def carregar_modelo(caminho):
    """Carrega modelo salvo"""
    try:
        return joblib.load(caminho), None
    except Exception as e:
        return None, str(e)

def criar_grafico_temporal(df, coluna, titulo):
    """Cria gráfico temporal interativo"""
    if 'datetime' in df.columns:
        fig = px.line(df, x='datetime', y=coluna, title=titulo)
    elif 'Data' in df.columns:
        fig = px.line(df, x='Data', y=coluna, title=titulo)
    else:
        fig = px.line(df, y=coluna, title=titulo)
    
    fig.update_layout(
        xaxis_title="Tempo",
        yaxis_title=coluna,
        hovermode='x unified'
    )
    return fig

def criar_grafico_correlacao(df):
    """Cria heatmap de correlação"""
    # Selecionar apenas colunas numéricas
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Limitar a 15 colunas para visualização
    if len(df_numeric.columns) > 15:
        df_numeric = df_numeric.iloc[:, :15]
    
    corr_matrix = df_numeric.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0
    ))
    
    fig.update_layout(
        title="Matriz de Correlação",
        width=800,
        height=800
    )
    
    return fig

# ============================================================================
# SIDEBAR - Navegação
# ============================================================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1163/1163661.png", width=100)
st.sidebar.title("🌦️ Menu Principal")

pagina = st.sidebar.radio(
    "Navegação",
    ["🏠 Home", "📊 Upload & EDA", "🤖 Treinar Modelo", "🔮 Fazer Previsão", "📈 Análise de Resultados"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**PI4 Machine Learning 2025**  
Sistema de Previsão Meteorológica (ClimaPrev) 
Desenvolvido com Streamlit
""")

# ============================================================================
# PÁGINA: HOME
# ============================================================================
if pagina == "🏠 Home":
    st.markdown('<h1 class="main-header">🌦️ ClimaPrev</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Bem-vindo ao ClimaPrev! 
    
    Este sistema utiliza **Machine Learning** para prever:
    - ☔ **Classificação:** Vai chover ou não?
    - 💧 **Regressão:** Qual a quantidade de chuva esperada?
    
    ### 🎯 Funcionalidades Principais:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("📊 **Upload de Dados**\nFaça upload dos seus dados meteorológicos em formato CSV")
    
    with col2:
        st.success("🤖 **Treinar Modelo**\nTreine modelos de ML com seus dados")
    
    with col3:
        st.warning("🔮 **Fazer Previsões**\nUse modelos treinados para prever chuva")
    
    st.markdown("---")
    
    st.markdown("### 📋 Como Usar:")
    st.markdown("""
    1. **Upload de Dados:** Navegue até "Upload & EDA" e faça upload do seu CSV
    2. **Exploração:** Visualize gráficos e estatísticas dos dados
    3. **Treinamento:** Vá para "Treinar Modelo" e clique em "Treinar"
    4. **Previsão:** Use "Fazer Previsão" para testar o modelo
    5. **Análise:** Veja métricas detalhadas em "Análise de Resultados"
    """)
    
    st.markdown("---")
    
    # Status do sistema
    st.markdown("### 🔧 Status do Sistema:")
    
    col1, col2, col3 = st.columns(3)
    
    # Verificar dados
    data_dir = project_root / 'src' / 'data'
    dados_processados = data_dir / 'dados_processados_ml.csv'
    
    with col1:
        if dados_processados.exists():
            st.success("✅ Dados Processados Disponíveis")
        else:
            st.error("❌ Dados não encontrados")
    
    # Verificar modelos
    modelo_class = data_dir /  'models' / 'modelo_classificacao.joblib'
    modelo_reg = data_dir / 'models' / 'modelo_regressao.joblib'
    
    with col2:
        if modelo_class.exists():
            st.success("✅ Modelo de Classificação OK")
        else:
            st.warning("⚠️ Modelo não treinado")
    
    with col3:
        if modelo_reg.exists():
            st.success("✅ Modelo de Regressão OK")
        else:
            st.warning("⚠️ Modelo não treinado")

# ============================================================================
# PÁGINA: UPLOAD & EDA
# ============================================================================
elif pagina == "📊 Upload & EDA":
    st.title("📊 Upload de Dados & Análise Exploratória")
    
    st.markdown("""
    Faça upload do seu arquivo CSV com dados meteorológicos para análise exploratória.
    
    **Formatos aceitos:** CSV com separador `;` ou `,`
    """)
    
    # Upload de arquivo
    uploaded_file = st.file_uploader(
        "Escolha um arquivo CSV",
        type=['csv'],
        help="Arquivo CSV com dados do INMET ou similar"
    )
    
    if uploaded_file is not None:
        # Carregar dados
        with st.spinner("Carregando dados..."):
            df, erro = carregar_dados(uploaded_file)
        
        if erro:
            st.error(f"❌ Erro ao carregar arquivo: {erro}")
        else:
            st.success(f"✅ Arquivo carregado com sucesso!")
            
            # Informações básicas
            st.markdown("### 📋 Informações do Dataset")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Registros", f"{len(df):,}")
            with col2:
                st.metric("Colunas", len(df.columns))
            with col3:
                st.metric("Valores Nulos", f"{df.isnull().sum().sum():,}")
            with col4:
                memoria_mb = df.memory_usage(deep=True).sum() / 1024**2
                st.metric("Tamanho", f"{memoria_mb:.2f} MB")
            
            # Abas para diferentes visualizações
            tab1, tab2, tab3, tab4 = st.tabs(["📋 Dados", "📊 Estatísticas", "📈 Gráficos", "🔗 Correlação"])
            
            with tab1:
                st.markdown("#### Primeiras 100 Linhas")
                st.dataframe(df.head(100), use_container_width=True)
                
                st.markdown("#### Tipos de Dados")
                tipo_info = pd.DataFrame({
                    'Coluna': df.columns,
                    'Tipo': df.dtypes.values,
                    'Não-Nulos': df.count().values,
                    '% Nulos': (df.isnull().sum() / len(df) * 100).values
                })
                st.dataframe(tipo_info, use_container_width=True)
            
            with tab2:
                st.markdown("#### Estatísticas Descritivas")
                st.dataframe(df.describe(), use_container_width=True)
                
                # Distribuição de valores nulos
                st.markdown("#### Top 10 Colunas com Mais Valores Nulos")
                nulos = df.isnull().sum().sort_values(ascending=False).head(10)
                if nulos.sum() > 0:
                    fig = px.bar(
                        x=nulos.index,
                        y=nulos.values,
                        labels={'x': 'Coluna', 'y': 'Quantidade de Nulos'},
                        title="Valores Nulos por Coluna"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("✅ Nenhum valor nulo encontrado!")
            
            with tab3:
                st.markdown("#### Gráficos Interativos")
                
                # Selecionar colunas numéricas
                colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(colunas_numericas) > 0:
                    col_selecionada = st.selectbox(
                        "Selecione uma coluna para visualizar",
                        colunas_numericas
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Histograma
                        fig_hist = px.histogram(
                            df,
                            x=col_selecionada,
                            nbins=50,
                            title=f"Distribuição: {col_selecionada}"
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with col2:
                        # Box plot
                        fig_box = px.box(
                            df,
                            y=col_selecionada,
                            title=f"Box Plot: {col_selecionada}"
                        )
                        st.plotly_chart(fig_box, use_container_width=True)
                    
                    # Série temporal (se houver coluna de data)
                    if 'datetime' in df.columns or 'Data' in df.columns:
                        fig_tempo = criar_grafico_temporal(df, col_selecionada, f"Evolução Temporal: {col_selecionada}")
                        st.plotly_chart(fig_tempo, use_container_width=True)
                
                else:
                    st.warning("⚠️ Nenhuma coluna numérica encontrada")
            
            with tab4:
                st.markdown("#### Matriz de Correlação")
                
                if len(colunas_numericas) > 1:
                    with st.spinner("Calculando correlações..."):
                        fig_corr = criar_grafico_correlacao(df)
                        st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Top correlações
                    st.markdown("#### Top 10 Correlações Mais Fortes")
                    df_numeric = df.select_dtypes(include=[np.number])
                    corr_matrix = df_numeric.corr()
                    
                    # Obter correlações únicas
                    correlacoes = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            correlacoes.append({
                                'Variável 1': corr_matrix.columns[i],
                                'Variável 2': corr_matrix.columns[j],
                                'Correlação': corr_matrix.iloc[i, j]
                            })
                    
                    df_corr = pd.DataFrame(correlacoes)
                    df_corr = df_corr.sort_values('Correlação', key=abs, ascending=False).head(10)
                    st.dataframe(df_corr, use_container_width=True)
                else:
                    st.warning("⚠️ Necessário pelo menos 2 colunas numéricas")
            
            # Botão para salvar dados processados
            st.markdown("---")
            st.markdown("### 💾 Salvar Dados")
            
            if st.button("💾 Salvar Dados Processados", key="salvar_dados"):
                try:
                    output_path = project_root / 'data' / 'dados_uploaded.csv'
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(output_path, index=False)
                    st.success(f"✅ Dados salvos em: {output_path}")
                except Exception as e:
                    st.error(f"❌ Erro ao salvar: {e}")

# ============================================================================
# PÁGINA: TREINAR MODELO
# ============================================================================
elif pagina == "🤖 Treinar Modelo":
    st.title("🤖 Treinamento de Modelos")
    
    st.markdown("""
    Configure e treine modelos de Machine Learning para previsão meteorológica.
    """)
    
    # Verificar se há dados
    data_dir = project_root / 'src' / 'data'
    dados_processados = data_dir / 'dados_processados_ml.csv'
    
    if not dados_processados.exists():
        st.warning("⚠️ Dados pré-processados não encontrados!")
        st.info("""
        **Como obter dados processados:**
        1. Faça upload de dados na aba "Upload & EDA"
        2. Ou execute o pré-processamento: `python -m notebooks.exemplo_preprocess`
        """)
    else:
        # Carregar dados
        df = pd.read_csv(dados_processados)
        
        st.success(f"✅ Dados carregados: {len(df)} registros")
        
        # Configurações de treinamento
        st.markdown("### ⚙️ Configurações de Treinamento")
        
        col1, col2 = st.columns(2)
        
        with col1:
            treinar_classificacao = st.checkbox("Treinar Classificação (Chuva Sim/Não)", value=True)
            target_class = st.text_input("Coluna Target (Classificação)", value="Chuva")
        
        with col2:
            treinar_regressao = st.checkbox("Treinar Regressão (Quantidade)", value=True)
            target_reg = st.text_input("Coluna Target (Regressão)", value="PRECIPITAÇÃO TOTAL, HORÁRIO (mm)")
        
        # Parâmetros avançados
        with st.expander("🔧 Parâmetros Avançados"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_estimators = st.slider("N° de Árvores", 50, 200, 100, 10)
            with col2:
                max_depth = st.slider("Profundidade Máxima", 5, 20, 10, 1)
            with col3:
                test_size = st.slider("% Teste", 10, 30, 20, 5)
        
        st.markdown("---")
        
        # Botão de treinamento
        if st.button("🚀 TREINAR MODELOS", key="treinar", type="primary"):
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                if not MODULOS_DISPONIVEIS:
                    st.error("❌ Módulos de treinamento não disponíveis")
                else:
                    # Inicializar treinador
                    status_text.text("Inicializando treinador...")
                    progress_bar.progress(10)
                    
                    trainer = WeatherModelTrainer(verbose=False)
                    
                    # Preparar dados
                    status_text.text("Preparando dados...")
                    progress_bar.progress(20)
                    
                    X, y_class, y_reg = trainer.preparar_dados(
                        df,
                        target_class=target_class,
                        target_reg=target_reg if treinar_regressao else None
                    )
                    
                    resultados = {}
                    
                    # Treinar classificação
                    if treinar_classificacao:
                        status_text.text("Treinando modelo de classificação...")
                        progress_bar.progress(40)
                        
                        model_class = trainer.treinar_classificacao(X, y_class)
                        
                        # Salvar
                        caminho_class = trainer.salvar_modelo(model_class, 'modelo_classificacao')
                        resultados['classificacao'] = trainer.metrics['classificacao']
                        
                        progress_bar.progress(60)
                    
                    # Treinar regressão
                    if treinar_regressao and y_reg is not None:
                        status_text.text("Treinando modelo de regressão...")
                        progress_bar.progress(70)
                        
                        model_reg = trainer.treinar_regressao(X, y_reg)
                        
                        # Salvar
                        caminho_reg = trainer.salvar_modelo(model_reg, 'modelo_regressao')
                        resultados['regressao'] = trainer.metrics['regressao']
                        
                        progress_bar.progress(90)
                    
                    # Finalizar
                    status_text.text("Treinamento concluído!")
                    progress_bar.progress(100)
                    time.sleep(0.5)
                    status_text.empty()
                    progress_bar.empty()
                    
                    # Exibir resultados
                    st.success("🎉 Treinamento Concluído com Sucesso!")
                    
                    st.markdown("### 📊 Resultados do Treinamento")
                    
                    if 'classificacao' in resultados:
                        st.markdown("#### 🎯 Classificação")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Accuracy", f"{resultados['classificacao']['accuracy']:.3f}")
                        with col2:
                            st.metric("Precision", f"{resultados['classificacao']['precision']:.3f}")
                        with col3:
                            st.metric("Recall", f"{resultados['classificacao']['recall']:.3f}")
                        with col4:
                            st.metric("F1-Score", f"{resultados['classificacao']['f1_score']:.3f}")
                    
                    if 'regressao' in resultados:
                        st.markdown("#### 📈 Regressão")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("R² Score", f"{resultados['regressao']['r2_score']:.3f}")
                        with col2:
                            st.metric("RMSE", f"{resultados['regressao']['rmse']:.3f}")
                        with col3:
                            st.metric("MAE", f"{resultados['regressao']['mae']:.3f}")
                    
                    st.balloons()
                    
            except Exception as e:
                st.error(f"❌ Erro durante o treinamento: {e}")
                st.exception(e)

# ============================================================================
# PÁGINA: FAZER PREVISÃO
# ============================================================================
elif pagina == "🔮 Fazer Previsão":
    st.title("🔮 Fazer Previsão")
    
    st.markdown("Faça previsões meteorológicas usando os modelos treinados.")
    
    # Verificar modelos
    data_dir = project_root / 'src' / 'data'
    modelo_class_path = data_dir / 'models' / 'modelo_classificacao.joblib'
    modelo_reg_path = data_dir / 'models' / 'modelo_regressao.joblib'
    
    modelos_disponiveis = {
        'classificacao': modelo_class_path.exists(),
        'regressao': modelo_reg_path.exists()
    }
    
    if not any(modelos_disponiveis.values()):
        st.warning("⚠️ Nenhum modelo treinado encontrado!")
        st.info("Vá para 'Treinar Modelo' primeiro")
    else:
        # Escolher modo de previsão
        modo = st.radio(
            "Modo de Previsão",
            ["📝 Inserir Dados Manualmente", "📁 Upload de Arquivo CSV"]
        )
        
        if modo == "📝 Inserir Dados Manualmente":
            st.markdown("### 📝 Inserir Dados Meteorológicos")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                temperatura = st.number_input("🌡️ Temperatura (°C)", 10.0, 45.0, 25.0, 0.5)
                umidade = st.number_input("💧 Umidade (%)", 0.0, 100.0, 70.0, 1.0)
                pressao = st.number_input("🔽 Pressão (mB)", 900.0, 1100.0, 1013.0, 1.0)
            
            with col2:
                hora = st.slider("🕐 Hora do Dia", 0, 23, 12)
                mes = st.slider("📅 Mês", 1, 12, 6)
                dia_semana = st.slider("📆 Dia da Semana", 0, 6, 3, help="0=Segunda, 6=Domingo")
            
            with col3:
                radiacao = st.number_input("☀️ Radiação (KJ/m²)", 0.0, 5000.0, 1000.0, 100.0)
                vento = st.number_input("💨 Velocidade Vento (m/s)", 0.0, 30.0, 3.0, 0.5)

            # Slider de limiar (threshold) — exibido antes de processar a previsão
            limiar_default = st.session_state.get('previsao_threshold', 0.5)
            limiar = st.slider(
                "Limiar para considerar 'vai chover'",
                min_value=0.0,
                max_value=1.0,
                value=float(limiar_default),
                step=0.01
            )
            # Persistir escolha na sessão
            st.session_state['previsao_threshold'] = float(limiar)

            if st.button("🔮 FAZER PREVISÃO", key="prever_manual", type="primary"):
                try:
                    # Usar FeatureCalculator para criar entrada completa
                    calculator = FeatureCalculator(verbose=False)
                    
                    df_input = calculator.criar_entrada_completa(
                        temperatura=temperatura,
                        umidade=umidade,
                        pressao=pressao,
                        radiacao=radiacao,
                        hora=hora,
                        mes=mes,
                        dia_semana=dia_semana,
                        vento=vento
                    )
                    
                    # Carregar modelos e fazer previsão
                    if modelos_disponiveis['classificacao']:
                        predictor = WeatherPredictor(
                            caminho_modelo_class=str(modelo_class_path),
                            verbose=False
                        )
                        
                        resultado = predictor.prever_classificacao(df_input)
                        
                        st.markdown("---")
                        st.markdown("### 🎯 Resultado da Previsão")
                        
                        # Probabilidade retornada pelo modelo
                        prob_chuva = float(resultado['prob_com_chuva'][0])

                        # Decisão final usando o limiar já selecionado acima
                        vai_chover = prob_chuva >= limiar

                        col1, col2 = st.columns(2)

                        with col1:
                            if vai_chover:
                                st.error("### 🌧️ VAI CHOVER")
                            else:
                                st.success("### ☀️ NÃO VAI CHOVER")

                        with col2:
                            # Mostrar probabilidade e delta relativo ao limiar atual
                            delta_pct = (prob_chuva - limiar) * 100
                            st.metric(
                                "Probabilidade de Chuva",
                                f"{prob_chuva*100:.1f}%",
                                delta=f"{delta_pct:.1f}% vs limiar ({limiar*100:.0f}%)"
                            )
                        
                        # Gauge chart
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = prob_chuva * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Probabilidade de Chuva (%)"},
                            delta = {'reference': 50},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps' : [
                                    {'range': [0, 30], 'color': "lightgreen"},
                                    {'range': [30, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "lightcoral"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Mostrar features calculadas (debug)
                        with st.expander("🔍 Debug: Features Calculadas"):
                            st.write("**Entrada criada com as seguintes features:**")
                            st.dataframe(df_input.T, use_container_width=True)

                            try:
                                # Obter vetor codificado/alinhado que será passado ao modelo
                                X_encoded = predictor.preparar_features(df_input.copy())

                                st.markdown("**Vetor codificado / alinhado (usado pelo modelo):**")
                                st.dataframe(X_encoded.T, use_container_width=True)

                                # Tentar carregar dados de treino para comparar percentis
                                treino_path = project_root / 'src' / 'data' / 'dados_processados_ml.csv'
                                if treino_path.exists():
                                    df_treino = pd.read_csv(treino_path)

                                    percent_rows = []
                                    for col in X_encoded.columns:
                                        val = float(X_encoded.iloc[0][col])
                                        pct = None
                                        mean = None
                                        std = None
                                        if col in df_treino.columns:
                                            try:
                                                series = pd.to_numeric(df_treino[col], errors='coerce').dropna()
                                                if len(series) > 0:
                                                    pct = (series <= val).mean() * 100
                                                    mean = series.mean()
                                                    std = series.std()
                                            except Exception:
                                                pct = None

                                        percent_rows.append({
                                            'feature': col,
                                            'value': val,
                                            'train_mean': mean,
                                            'train_std': std,
                                            'percentil_vs_treino_%': round(pct, 2) if pct is not None else None
                                        })

                                    df_percent = pd.DataFrame(percent_rows)
                                    st.markdown("**Comparação com distribuição de treino:**")
                                    st.dataframe(df_percent.set_index('feature'), use_container_width=True)
                                else:
                                    st.info("⚠️ Arquivo de treino não encontrado para comparação de percentis")

                                # Mostrar metadados do modelo
                                st.markdown("**Metadados do modelo:**")
                                meta_info = {}
                                meta_info['features_esperadas'] = predictor.features_esperadas if hasattr(predictor, 'features_esperadas') else None
                                try:
                                    if predictor.modelo_classificacao is not None and hasattr(predictor.modelo_classificacao, 'classes_'):
                                        meta_info['classes'] = list(predictor.modelo_classificacao.classes_)
                                except Exception:
                                    pass

                                # Importância das features (se disponível)
                                try:
                                    if predictor.modelo_classificacao is not None and hasattr(predictor.modelo_classificacao, 'feature_importances_'):
                                        importances = predictor.modelo_classificacao.feature_importances_
                                        feat_names = predictor.features_esperadas or X_encoded.columns.tolist()
                                        df_imp = pd.DataFrame({'feature': feat_names, 'importance': importances})
                                        df_imp = df_imp.sort_values('importance', ascending=False).head(20).set_index('feature')
                                        st.markdown("**Top feature importances (classificador):**")
                                        st.dataframe(df_imp, use_container_width=True)
                                        meta_info['has_feature_importances'] = True
                                except Exception:
                                    meta_info['has_feature_importances'] = False

                                st.json(meta_info)

                            except Exception as e:
                                st.warning(f"⚠️ Não foi possível gerar debug extra: {e}")
                    
                except Exception as e:
                    st.error(f"❌ Erro ao fazer previsão: {e}")
                    st.exception(e)
        
        else:  # Upload CSV
            st.markdown("### 📁 Upload de Arquivo para Previsão em Lote")
            
            uploaded_file = st.file_uploader(
                "Escolha um arquivo CSV",
                type=['csv'],
                key="upload_predict"
            )
            
            if uploaded_file is not None:
                df_input, erro = carregar_dados(uploaded_file)
                
                if erro:
                    st.error(f"❌ Erro: {erro}")
                else:
                    st.success(f"✅ Arquivo carregado: {len(df_input)} registros")
                    
                    st.dataframe(df_input.head(), use_container_width=True)
                    
                    if st.button("🔮 FAZER PREVISÕES EM LOTE", key="prever_lote"):
                        try:
                            with st.spinner("Fazendo previsões..."):
                                predictor = WeatherPredictor(
                                    caminho_modelo_class=str(modelo_class_path) if modelos_disponiveis['classificacao'] else None,
                                    caminho_modelo_reg=str(modelo_reg_path) if modelos_disponiveis['regressao'] else None,
                                    verbose=False
                                )
                                
                                df_resultado = predictor.prever_completo(df_input)
                            
                            st.success("✅ Previsões concluídas!")
                            
                            st.markdown("### 📊 Resultados")
                            st.dataframe(df_resultado, use_container_width=True)
                            
                            # Estatísticas
                            if 'previsao_chuva' in df_resultado.columns:
                                n_chuva = df_resultado['previsao_chuva'].sum()
                                pct_chuva = (n_chuva / len(df_resultado)) * 100
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total de Previsões", len(df_resultado))
                                with col2:
                                    st.metric("Previsão: Vai Chover", n_chuva)
                                with col3:
                                    st.metric("% Chuva", f"{pct_chuva:.1f}%")
                                
                                # Gráfico de pizza
                                fig = px.pie(
                                    values=[n_chuva, len(df_resultado)-n_chuva],
                                    names=['Vai Chover', 'Não Vai Chover'],
                                    title='Distribuição de Previsões',
                                    color_discrete_sequence=['#1f77b4', '#ff7f0e']
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Download dos resultados
                            csv = df_resultado.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Resultados (CSV)",
                                data=csv,
                                file_name=f"previsoes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Erro ao fazer previsões: {e}")
                            st.exception(e)

# ============================================================================
# PÁGINA: ANÁLISE DE RESULTADOS
# ============================================================================
elif pagina == "📈 Análise de Resultados":
    st.title("📈 Análise Detalhada de Resultados")
    
    st.markdown("Visualize métricas e análises detalhadas dos modelos treinados.")
    
    # Verificar se há modelos salvos
    data_dir = project_root / 'src' / 'data'
    modelo_class_path = data_dir / 'models' / 'modelo_classificacao.joblib'
    modelo_reg_path = data_dir / 'models' / 'modelo_regressao.joblib'
    
    col1, col2 = st.columns(2)
    
    with col1:
        if modelo_class_path.exists():
            st.success("✅ Modelo de Classificação Disponível")
            analisar_class = st.checkbox("Analisar Classificação", value=True)
        else:
            st.warning("⚠️ Modelo de Classificação não encontrado")
            analisar_class = False
    
    with col2:
        if modelo_reg_path.exists():
            st.success("✅ Modelo de Regressão Disponível")
            analisar_reg = st.checkbox("Analisar Regressão", value=True)
        else:
            st.warning("⚠️ Modelo de Regressão não encontrado")
            analisar_reg = False
    
    if not (analisar_class or analisar_reg):
        st.info("📌 Treine os modelos primeiro na aba 'Treinar Modelo'")
    else:
        # Carregar dados de teste
        dados_processados = data_dir / 'dados_processados_ml.csv'
        
        if dados_processados.exists():
            df = pd.read_csv(dados_processados)
            
            if analisar_class:
                st.markdown("## 🎯 Análise: Modelo de Classificação")
                
                try:
                    # Carregar modelo
                    modelo_class, _ = carregar_modelo(modelo_class_path)
                    
                    if modelo_class is not None:
                        # Preparar dados
                        if MODULOS_DISPONIVEIS:
                            trainer = WeatherModelTrainer(verbose=False)
                            
                            # Determinar target
                            target_class = 'Chuva' if 'Chuva' in df.columns else df.columns[-1]
                            
                            X, y_class, _ = trainer.preparar_dados(df, target_class=target_class)
                            
                            # Split
                            from sklearn.model_selection import train_test_split
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y_class, test_size=0.2, random_state=42
                            )
                            
                            # Fazer previsões
                            # Remover coluna target se existir
                            if 'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)' in X_test.columns:
                                X_test = X_test.drop(columns=['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'])

                            # Fazer previsão
                            y_pred = modelo_class.predict(X_test)
                            y_proba = modelo_class.predict_proba(X_test)[:, 1]
                            
                            # Métricas
                            from sklearn.metrics import (
                                accuracy_score, precision_score, recall_score, 
                                f1_score, confusion_matrix, roc_curve, auc,
                                classification_report
                            )
                            
                            acc = accuracy_score(y_test, y_pred)
                            prec = precision_score(y_test, y_pred)
                            rec = recall_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred)
                            
                            # Exibir métricas
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("🎯 Accuracy", f"{acc:.3f}")
                            with col2:
                                st.metric("🎯 Precision", f"{prec:.3f}")
                            with col3:
                                st.metric("🎯 Recall", f"{rec:.3f}")
                            with col4:
                                st.metric("🎯 F1-Score", f"{f1:.3f}")
                            
                            # Matriz de Confusão
                            st.markdown("### 📊 Matriz de Confusão")
                            
                            cm = confusion_matrix(y_test, y_pred)
                            
                            fig = go.Figure(data=go.Heatmap(
                                z=cm,
                                x=['Não Chove', 'Chove'],
                                y=['Não Chove', 'Chove'],
                                text=cm,
                                texttemplate="%{text}",
                                textfont={"size": 20},
                                colorscale='Blues'
                            ))
                            
                            fig.update_layout(
                                title="Matriz de Confusão",
                                xaxis_title="Previsto",
                                yaxis_title="Real",
                                width=600,
                                height=500
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Curva ROC
                            st.markdown("### 📈 Curva ROC")
                            
                            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                            roc_auc = auc(fpr, tpr)
                            
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=fpr, y=tpr,
                                mode='lines',
                                name=f'ROC (AUC = {roc_auc:.3f})',
                                line=dict(color='blue', width=2)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=[0, 1], y=[0, 1],
                                mode='lines',
                                name='Baseline',
                                line=dict(color='red', width=2, dash='dash')
                            ))
                            
                            fig.update_layout(
                                title=f'Curva ROC (AUC = {roc_auc:.3f})',
                                xaxis_title='Taxa de Falsos Positivos',
                                yaxis_title='Taxa de Verdadeiros Positivos',
                                width=700,
                                height=500
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Relatório de Classificação
                            st.markdown("### 📋 Relatório Detalhado")
                            
                            report = classification_report(y_test, y_pred, output_dict=True)
                            df_report = pd.DataFrame(report).transpose()
                            st.dataframe(df_report.style.highlight_max(axis=0), use_container_width=True)
                        
                        else:
                            st.warning("⚠️ Módulos de análise não disponíveis")
                
                except Exception as e:
                    st.error(f"❌ Erro ao analisar modelo de classificação: {e}")
                    st.exception(e)
            
            if analisar_reg:
                st.markdown("---")
                st.markdown("## 📈 Análise: Modelo de Regressão")
                
                try:
                    # Carregar modelo
                    modelo_reg, _ = carregar_modelo(modelo_reg_path)
                    
                    if modelo_reg is not None and MODULOS_DISPONIVEIS:
                        # Preparar dados
                        trainer = WeatherModelTrainer(verbose=False)
                        
                        target_reg = 'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'
                        if target_reg not in df.columns:
                            target_reg = df.columns[-1]
                        
                        X, _, y_reg = trainer.preparar_dados(df, target_reg=target_reg)
                        
                        if y_reg is not None:
                            # Split
                            from sklearn.model_selection import train_test_split
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y_reg, test_size=0.2, random_state=42
                            )
                            
                            # Fazer previsões
                            y_pred = modelo_reg.predict(X_test)
                            
                            # Métricas
                            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                            
                            r2 = r2_score(y_test, y_pred)
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            mae = mean_absolute_error(y_test, y_pred)
                            
                            # Exibir métricas
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("📊 R² Score", f"{r2:.3f}")
                            with col2:
                                st.metric("📊 RMSE", f"{rmse:.3f}")
                            with col3:
                                st.metric("📊 MAE", f"{mae:.3f}")
                            
                            # Gráfico: Real vs Previsto
                            st.markdown("### 📊 Valores Reais vs Previstos")
                            
                            # Limitar a 1000 pontos para performance
                            n_samples = min(1000, len(y_test))
                            indices = np.random.choice(len(y_test), n_samples, replace=False)
                            
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=y_test.iloc[indices],
                                y=y_pred[indices],
                                mode='markers',
                                name='Previsões',
                                marker=dict(size=5, opacity=0.6)
                            ))
                            
                            # Linha diagonal (perfeita)
                            min_val = min(y_test.min(), y_pred.min())
                            max_val = max(y_test.max(), y_pred.max())
                            
                            fig.add_trace(go.Scatter(
                                x=[min_val, max_val],
                                y=[min_val, max_val],
                                mode='lines',
                                name='Predição Perfeita',
                                line=dict(color='red', width=2, dash='dash')
                            ))
                            
                            fig.update_layout(
                                title=f'Real vs Previsto (R² = {r2:.3f})',
                                xaxis_title='Valor Real',
                                yaxis_title='Valor Previsto',
                                width=700,
                                height=500
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Distribuição de Resíduos
                            st.markdown("### 📉 Distribuição de Resíduos")
                            
                            residuos = y_test.values - y_pred
                            
                            fig = px.histogram(
                                x=residuos,
                                nbins=50,
                                title='Distribuição de Resíduos',
                                labels={'x': 'Resíduo', 'y': 'Frequência'}
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Feature Importance
                            if hasattr(modelo_reg, 'feature_importances_'):
                                st.markdown("### 🔍 Importância das Features")
                                
                                importances = modelo_reg.feature_importances_
                                feature_names = X.columns
                                
                                df_importance = pd.DataFrame({
                                    'Feature': feature_names,
                                    'Importance': importances
                                }).sort_values('Importance', ascending=False).head(15)
                                
                                fig = px.bar(
                                    df_importance,
                                    x='Importance',
                                    y='Feature',
                                    orientation='h',
                                    title='Top 15 Features Mais Importantes'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Erro ao analisar modelo de regressão: {e}")
                    st.exception(e)
        
        else:
            st.warning("⚠️ Dados processados não encontrados")
            st.info("Execute o pré-processamento primeiro")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem;'>
    <p><strong>Sistema de Previsão Meteorológica (ClimaPrev)</strong></p>
    <p>Desenvolvido com ❤️ usando Streamlit | PI4 Machine Learning 2025</p>
            <p>🔗 GitHub: https://github.com/Gavi189/ProjetoPI4ML2025</p>
    <p>📧 Contato:<br><a href="mailto:2211273@aluno.univesp.br">Gabriel Kaique de Areal Rodrigues</a><br>
            <a href="mailto:2215969@aluno.univesp.br">Gabriel Valério Andrade Rodrigues</a><br>
            <a href="mailto:2215890@aluno.univesp.br">Grace Kelly Coracin</a><br>
            <a href="mailto:2229846@aluno.univesp.br">Leandro Junior Gaspar de Oliveira</a><br>
            <a href="mailto:2219233@aluno.univesp.br">Simone Pereira do Nascimento</a><br>
    </p>
</div>
""", unsafe_allow_html=True)

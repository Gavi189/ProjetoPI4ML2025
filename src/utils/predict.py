"""
Módulo de Previsão e Teste para Modelos Meteorológicos
Autor: PI4-MachineLearning-2025
Descrição: Carrega modelos treinados e faz previsões em dados novos
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Union
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')


class WeatherPredictor:
    """Classe para fazer previsões com modelos treinados"""
    
    def __init__(self, caminho_modelo_class: Optional[str] = None,
                 caminho_modelo_reg: Optional[str] = None,
                 verbose: bool = True):
        """
        Inicializa o preditor
        
        Args:
            caminho_modelo_class: Caminho para modelo de classificação
            caminho_modelo_reg: Caminho para modelo de regressão
            verbose: Se True, imprime informações
        """
        self.verbose = verbose
        self.modelo_classificacao = None
        self.modelo_regressao = None
        self.features_esperadas = None
        
        # Carregar modelos
        if caminho_modelo_class:
            self.carregar_modelo_classificacao(caminho_modelo_class)
        
        if caminho_modelo_reg:
            self.carregar_modelo_regressao(caminho_modelo_reg)
    
    def _log(self, message: str):
        """Imprime mensagem se verbose=True"""
        if self.verbose:
            print(message)
    
    def carregar_modelo_classificacao(self, caminho: str):
        """
        Carrega modelo de classificação do disco
        
        Args:
            caminho: Caminho para o arquivo .joblib
        """
        self._log(f"📂 Carregando modelo de classificação: {caminho}")
        
        try:
            self.modelo_classificacao = joblib.load(caminho)
            
            # Extrair features esperadas
            if hasattr(self.modelo_classificacao, 'feature_names_in_'):
                self.features_esperadas = list(self.modelo_classificacao.feature_names_in_)
                self._log(f"   ✅ Modelo carregado com {len(self.features_esperadas)} features")
            else:
                self._log(f"   ✅ Modelo carregado (features não disponíveis)")
                
        except Exception as e:
            self._log(f"   ❌ Erro ao carregar modelo: {e}")
            raise
    
    def carregar_modelo_regressao(self, caminho: str):
        """
        Carrega modelo de regressão do disco
        
        Args:
            caminho: Caminho para o arquivo .joblib
        """
        self._log(f"📂 Carregando modelo de regressão: {caminho}")
        
        try:
            self.modelo_regressao = joblib.load(caminho)
            self._log(f"   ✅ Modelo de regressão carregado")
        except Exception as e:
            self._log(f"   ❌ Erro ao carregar modelo: {e}")
            raise
    
    def preparar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara features para previsão (alinha com features de treino)
        
        Args:
            df: DataFrame com dados novos
            
        Returns:
            DataFrame com features alinhadas
        """
        self._log("🔧 Preparando features para previsão...")
        
        if self.features_esperadas is None:
            self._log("   ⚠️ Features esperadas não definidas. Usando todas as colunas numéricas.")
            return df.select_dtypes(include=[np.number])
        
        # Selecionar apenas features que o modelo espera
        features_faltando = [f for f in self.features_esperadas if f not in df.columns]
        features_extras = [f for f in df.columns if f not in self.features_esperadas and f in df.select_dtypes(include=[np.number]).columns]
        
        if features_faltando:
            self._log(f"   ⚠️ {len(features_faltando)} features faltando: {features_faltando[:5]}...")
            # Adicionar features faltando com valor 0
            for feat in features_faltando:
                df[feat] = 0
        
        if features_extras:
            self._log(f"   ℹ️ {len(features_extras)} features extras serão ignoradas")
        
        # Retornar apenas as features na ordem correta
        df_aligned = df[self.features_esperadas].copy()
        
        # Preencher NaN com 0
        if df_aligned.isna().any().any():
            self._log("   💉 Preenchendo valores NaN com 0...")
            df_aligned = df_aligned.fillna(0)
        
        self._log(f"   ✅ Features preparadas: {len(df_aligned.columns)} colunas")
        return df_aligned
    
    def prever_classificacao(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Faz previsão de classificação (vai chover ou não?)
        
        Args:
            df: DataFrame com features
            
        Returns:
            Dicionário com previsões e probabilidades
        """
        if self.modelo_classificacao is None:
            raise ValueError("Modelo de classificação não carregado!")
        
        self._log("🔮 Fazendo previsão de classificação...")
        
        # Preparar features
        X = self.preparar_features(df)
        
        # Fazer previsão
        predicoes = self.modelo_classificacao.predict(X)
        probabilidades = self.modelo_classificacao.predict_proba(X)
        
        self._log(f"   ✅ {len(predicoes)} previsões realizadas")
        self._log(f"   📊 Resumo: {predicoes.sum()} eventos de chuva previstos ({predicoes.sum()/len(predicoes)*100:.1f}%)")
        
        return {
            'predicoes': predicoes,
            'probabilidades': probabilidades,
            'prob_sem_chuva': probabilidades[:, 0],
            'prob_com_chuva': probabilidades[:, 1]
        }
    
    def prever_regressao(self, df: pd.DataFrame) -> np.ndarray:
        """
        Faz previsão de regressão (quantidade de chuva)
        
        Args:
            df: DataFrame com features
            
        Returns:
            Array com valores previstos
        """
        if self.modelo_regressao is None:
            raise ValueError("Modelo de regressão não carregado!")
        
        self._log("🔮 Fazendo previsão de regressão...")
        
        # Preparar features
        X = self.preparar_features(df)
        
        # Fazer previsão
        predicoes = self.modelo_regressao.predict(X)
        
        # Garantir valores não negativos
        predicoes = np.maximum(predicoes, 0)
        
        self._log(f"   ✅ {len(predicoes)} previsões realizadas")
        self._log(f"   📊 Média prevista: {predicoes.mean():.2f} mm")
        self._log(f"   📊 Máximo previsto: {predicoes.max():.2f} mm")
        
        return predicoes
    
    def prever_completo(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Faz previsão completa (classificação + regressão)
        
        Args:
            df: DataFrame com features
            
        Returns:
            DataFrame com previsões adicionadas
        """
        self._log("=" * 80)
        self._log("🚀 INICIANDO PREVISÃO COMPLETA")
        self._log("=" * 80)
        
        df_result = df.copy()
        
        # Classificação
        if self.modelo_classificacao:
            resultado_class = self.prever_classificacao(df)
            df_result['previsao_chuva'] = resultado_class['predicoes']
            df_result['prob_chuva'] = resultado_class['prob_com_chuva']
        
        # Regressão
        if self.modelo_regressao:
            df_result['quantidade_chuva_prevista'] = self.prever_regressao(df)
        
        self._log("=" * 80)
        self._log("✅ PREVISÃO COMPLETA CONCLUÍDA")
        self._log("=" * 80)
        
        return df_result


class ModelEvaluator:
    """Classe para avaliar performance dos modelos em dados de teste"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.metricas = {}
    
    def _log(self, message: str):
        if self.verbose:
            print(message)
    
    def avaliar_classificacao(self, y_true: pd.Series, y_pred: np.ndarray,
                              y_proba: Optional[np.ndarray] = None) -> Dict:
        """
        Avalia modelo de classificação
        
        Args:
            y_true: Valores reais
            y_pred: Previsões
            y_proba: Probabilidades (opcional)
            
        Returns:
            Dicionário com métricas
        """
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                     f1_score, confusion_matrix, classification_report,
                                     roc_auc_score)
        
        self._log("📊 Avaliando modelo de classificação...")
        
        metricas = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        # ROC AUC se probabilidades disponíveis
        if y_proba is not None:
            try:
                metricas['roc_auc'] = roc_auc_score(y_true, y_proba)
            except:
                pass
        
        # Imprimir relatório
        self._log("\n" + "=" * 80)
        self._log("📈 MÉTRICAS DE CLASSIFICAÇÃO")
        self._log("=" * 80)
        self._log(f"Accuracy:  {metricas['accuracy']:.4f}")
        self._log(f"Precision: {metricas['precision']:.4f}")
        self._log(f"Recall:    {metricas['recall']:.4f}")
        self._log(f"F1-Score:  {metricas['f1_score']:.4f}")
        
        if 'roc_auc' in metricas:
            self._log(f"ROC AUC:   {metricas['roc_auc']:.4f}")
        
        self._log("\n📊 Matriz de Confusão:")
        self._log(f"{metricas['confusion_matrix']}")
        
        self._log("\n📋 Relatório Detalhado:")
        self._log(classification_report(y_true, y_pred, zero_division=0))
        self._log("=" * 80)
        
        self.metricas['classificacao'] = metricas
        return metricas
    
    def avaliar_regressao(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict:
        """
        Avalia modelo de regressão
        
        Args:
            y_true: Valores reais
            y_pred: Previsões
            
        Returns:
            Dicionário com métricas
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        self._log("📊 Avaliando modelo de regressão...")
        
        metricas = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        # Imprimir relatório
        self._log("\n" + "=" * 80)
        self._log("📈 MÉTRICAS DE REGRESSÃO")
        self._log("=" * 80)
        self._log(f"RMSE (Root Mean Squared Error): {metricas['rmse']:.4f}")
        self._log(f"MAE (Mean Absolute Error):      {metricas['mae']:.4f}")
        self._log(f"R² Score:                       {metricas['r2']:.4f}")
        self._log("=" * 80)
        
        self.metricas['regressao'] = metricas
        return metricas
    
    def plotar_resultados_classificacao(self, y_true: pd.Series, y_pred: np.ndarray,
                                       y_proba: Optional[np.ndarray] = None,
                                       salvar: bool = True):
        """Plota gráficos de avaliação de classificação"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix, roc_curve, auc
        
        self._log("📊 Gerando gráficos de avaliação...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Matriz de confusão
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Matriz de Confusão', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Previsão')
        axes[0].set_ylabel('Real')
        
        # Curva ROC
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            
            axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                        label=f'ROC curve (AUC = {roc_auc:.2f})')
            axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[1].set_xlim([0.0, 1.0])
            axes[1].set_ylim([0.0, 1.05])
            axes[1].set_xlabel('Taxa de Falsos Positivos')
            axes[1].set_ylabel('Taxa de Verdadeiros Positivos')
            axes[1].set_title('Curva ROC', fontsize=14, fontweight='bold')
            axes[1].legend(loc="lower right")
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if salvar:
            caminho = Path('../data/plots/avaliacao_classificacao.png')
            caminho.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(caminho, dpi=300, bbox_inches='tight')
            self._log(f"   ✅ Gráfico salvo em: {caminho}")
        
        plt.show()
        plt.close()


def testar_modelo_em_dados_novos(caminho_dados: str,
                                 caminho_modelo_class: str,
                                 caminho_modelo_reg: Optional[str] = None,
                                 col_target_class: str = 'Chuva',
                                 col_target_reg: Optional[str] = None,
                                 avaliar: bool = True,
                                 salvar_predicoes: bool = True):
    """
    Função principal para testar modelos em dados novos
    
    Args:
        caminho_dados: Caminho para dados de teste (pré-processados)
        caminho_modelo_class: Caminho para modelo de classificação
        caminho_modelo_reg: Caminho para modelo de regressão
        col_target_class: Nome da coluna target de classificação
        col_target_reg: Nome da coluna target de regressão
        avaliar: Se True, calcula métricas de avaliação
        salvar_predicoes: Se True, salva previsões em CSV
    
    Returns:
        DataFrame com previsões
    """
    print("=" * 80)
    print("🧪 TESTANDO MODELOS EM DADOS NOVOS")
    print("=" * 80)
    
    # Carregar dados
    print(f"📂 Carregando dados de teste: {caminho_dados}")
    df = pd.read_csv(caminho_dados)
    print(f"✅ Dados carregados: {len(df)} registros")
    
    # Inicializar preditor
    predictor = WeatherPredictor(
        caminho_modelo_class=caminho_modelo_class,
        caminho_modelo_reg=caminho_modelo_reg,
        verbose=True
    )
    
    # Fazer previsões
    df_resultado = predictor.prever_completo(df)
    
    # Avaliar (se targets disponíveis)
    if avaliar and col_target_class in df.columns:
        evaluator = ModelEvaluator(verbose=True)
        
        # Avaliar classificação
        resultado_class = predictor.prever_classificacao(df)
        evaluator.avaliar_classificacao(
            y_true=df[col_target_class],
            y_pred=resultado_class['predicoes'],
            y_proba=resultado_class['prob_com_chuva']
        )
        evaluator.plotar_resultados_classificacao(
            y_true=df[col_target_class],
            y_pred=resultado_class['predicoes'],
            y_proba=resultado_class['prob_com_chuva']
        )
        
        # Avaliar regressão (se disponível)
        if col_target_reg and col_target_reg in df.columns and predictor.modelo_regressao:
            predicoes_reg = predictor.prever_regressao(df)
            evaluator.avaliar_regressao(
                y_true=df[col_target_reg],
                y_pred=predicoes_reg
            )
    
    # Salvar previsões
    if salvar_predicoes:
        caminho_saida = Path('../data/predicoes.csv')
        df_resultado.to_csv(caminho_saida, index=False)
        print(f"\n💾 Previsões salvas em: {caminho_saida}")
    
    print("\n" + "=" * 80)
    print("✅ TESTE CONCLUÍDO COM SUCESSO")
    print("=" * 80)
    
    return df_resultado


def prever_com_dados_manuais(dados_manuais: Dict[str, float],
                             caminho_modelo_class: str,
                             caminho_modelo_reg: Optional[str] = None) -> Dict:
    """
    Faz previsão com dados inseridos manualmente
    
    Args:
        dados_manuais: Dicionário com features {nome_feature: valor}
        caminho_modelo_class: Caminho para modelo de classificação
        caminho_modelo_reg: Caminho para modelo de regressão
        
    Returns:
        Dicionário com previsões
    """
    print("=" * 80)
    print("🎯 FAZENDO PREVISÃO COM DADOS MANUAIS")
    print("=" * 80)
    
    # Criar DataFrame
    df = pd.DataFrame([dados_manuais])
    
    # Inicializar preditor
    predictor = WeatherPredictor(
        caminho_modelo_class=caminho_modelo_class,
        caminho_modelo_reg=caminho_modelo_reg,
        verbose=True
    )
    
    # Fazer previsões
    resultado = {}
    
    if predictor.modelo_classificacao:
        pred_class = predictor.prever_classificacao(df)
        resultado['vai_chover'] = bool(pred_class['predicoes'][0])
        resultado['probabilidade_chuva'] = float(pred_class['prob_com_chuva'][0])
    
    if predictor.modelo_regressao:
        pred_reg = predictor.prever_regressao(df)
        resultado['quantidade_prevista_mm'] = float(pred_reg[0])
    
    # Exibir resultado
    print("\n" + "=" * 80)
    print("🎯 RESULTADO DA PREVISÃO")
    print("=" * 80)
    
    if 'vai_chover' in resultado:
        print(f"🌧️  Vai chover? {'SIM' if resultado['vai_chover'] else 'NÃO'}")
        print(f"📊 Probabilidade: {resultado['probabilidade_chuva']*100:.1f}%")
    
    if 'quantidade_prevista_mm' in resultado:
        print(f"💧 Quantidade prevista: {resultado['quantidade_prevista_mm']:.2f} mm")
    
    print("=" * 80)
    
    return resultado


if __name__ == '__main__':
    """
    Exemplo de uso do módulo
    """
    print("Módulo de previsão carregado com sucesso!")
    print("\n📚 Exemplos de uso:")
    print("""
# 1. Testar modelo em dados novos
from src.predict import testar_modelo_em_dados_novos

testar_modelo_em_dados_novos(
    caminho_dados='data/dados_processados_ml.csv',
    caminho_modelo_class='data/models/modelo_classificacao.joblib',
    caminho_modelo_reg='data/models/modelo_regressao.joblib',
    col_target_class='Chuva',
    col_target_reg='PRECIPITAÇÃO TOTAL, HORÁRIO (mm)',
    avaliar=True,
    salvar_predicoes=True
)

# 2. Fazer previsão com dados manuais
from src.predict import prever_com_dados_manuais

dados = {
    'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)': 25.5,
    'UMIDADE RELATIVA DO AR, HORARIA (%)': 80,
    'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)': 1013.2,
    'VENTO, VELOCIDADE HORARIA (m/s)': 3.5,
    'hora': 14,
    'mes': 10,
    'dia_semana': 3
}

resultado = prever_com_dados_manuais(
    dados_manuais=dados,
    caminho_modelo_class='data/models/modelo_classificacao.joblib',
    caminho_modelo_reg='data/models/modelo_regressao.joblib'
)
    """)
    
    # Teste rápido (se modelos existirem)
    from pathlib import Path
    
    caminho_modelo = Path(__file__).parent.parent / 'data' / 'models' / 'modelo_classificacao.joblib'
    caminho_dados = Path(__file__).parent.parent / 'data' / 'dados_processados_ml.csv'
    
    if caminho_modelo.exists() and caminho_dados.exists():
        print("\n🧪 Executando teste rápido...")
        testar_modelo_em_dados_novos(
            caminho_dados=str(caminho_dados),
            caminho_modelo_class=str(caminho_modelo),
            col_target_class='Chuva',
            avaliar=True
        )
    else:
        print("\n⚠️ Modelos ou dados não encontrados. Execute primeiro o treinamento!")
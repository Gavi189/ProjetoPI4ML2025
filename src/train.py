"""
Módulo de Treinamento de Modelos para Previsão Meteorológica
Autor: PI4-MachineLearning-2025
Descrição: Treinamento de modelos de classificação (chuva sim/não) e regressão (quantidade de chuva)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import warnings
from typing import Optional, Tuple

warnings.filterwarnings('ignore')

class WeatherModelTrainer:
    """Classe para treinamento e avaliação de modelos meteorológicos"""
    
    def __init__(self, verbose: bool = True):
        """
        Inicializa o treinador de modelos
        
        Args:
            verbose: Se True, imprime informações durante o treinamento
        """
        self.verbose = verbose
        self.models = {}
        self.metrics = {}
        self.label_encoders = {}
    
    def _log(self, message: str):
        """Imprime mensagem se verbose=True"""
        if self.verbose:
            print(message)
    
    def preparar_dados(self, df: pd.DataFrame, 
                      target_class: str = 'Chuva',
                      target_reg: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepara os dados para treinamento
        
        Args:
            df: DataFrame pré-processado
            target_class: Nome da coluna alvo para classificação
            target_reg: Nome da coluna alvo para regressão (se None, não treina regressão)
            
        Returns:
            X: Features
            y_class: Target para classificação
            y_reg: Target para regressão (ou None)
        """
        self._log("📊 Preparando dados para treinamento...")
        self._log(f"   📈 Dados iniciais: {len(df)} registros, {len(df.columns)} colunas")
        
        # Criar cópia para não modificar o original
        df_work = df.copy()
        
        # Identificar colunas a excluir (não são features)
        colunas_excluir = []
        
        # Excluir colunas temporais explícitas
        colunas_temporais = ['datetime', 'data_temp', 'hora_temp', 'Data', 'Hora', 
                            'Data Medicao', 'Hora Medicao', 'DATA (YYYY-MM-DD)', 'HORA (UTC)']
        colunas_excluir.extend([col for col in colunas_temporais if col in df_work.columns])
        
        # Excluir targets
        if target_class in df_work.columns:
            colunas_excluir.append(target_class)
        if target_reg and target_reg in df_work.columns:
            colunas_excluir.append(target_reg)
        
        # Remover duplicatas da lista de exclusão
        colunas_excluir = list(set(colunas_excluir))
        
        self._log(f"   🗑️ Excluindo {len(colunas_excluir)} colunas: {colunas_excluir[:5]}...")
        
        # Selecionar features
        colunas_features = [col for col in df_work.columns if col not in colunas_excluir]
        
        self._log(f"   🎯 Features candidatas: {len(colunas_features)}")
        
        # Processar features categóricas
        for col in colunas_features.copy():
            if df_work[col].dtype == 'object':
                self._log(f"   🔤 Codificando coluna categórica: {col}")
                try:
                    le = LabelEncoder()
                    df_work[col] = le.fit_transform(df_work[col].astype(str))
                    self.label_encoders[col] = le
                except Exception as e:
                    self._log(f"   ⚠️ Erro ao codificar {col}: {e}. Removendo coluna.")
                    colunas_features.remove(col)
        
        # Criar DataFrame de features
        X = df_work[colunas_features].copy()
        
        # Verificar e preparar targets
        if target_class not in df_work.columns:
            raise ValueError(f"Coluna target '{target_class}' não encontrada no DataFrame!")
        
        y_class = df_work[target_class].copy()
        y_reg = df_work[target_reg].copy() if target_reg and target_reg in df_work.columns else None
        
        self._log(f"   📊 Features: {len(X.columns)}, Target class: {target_class}, Target reg: {target_reg}")
        
        # Remover linhas com NaN APENAS se houver poucos
        # Estatísticas de NaN antes
        nan_por_linha = X.isna().sum(axis=1)
        linhas_com_nan = (nan_por_linha > 0).sum()
        
        self._log(f"   🔍 Análise de NaN:")
        self._log(f"      - Linhas com pelo menos 1 NaN: {linhas_com_nan}")
        self._log(f"      - Total de valores NaN: {X.isna().sum().sum()}")
        
        # Estratégia: remover linhas apenas se tiverem muitos NaNs (>50% das features)
        limite_nan = len(X.columns) * 0.5
        mask_linhas_validas = nan_por_linha <= limite_nan
        
        # Também garantir que o target não é NaN
        mask_target_valido = ~y_class.isna()
        if y_reg is not None:
            mask_target_valido = mask_target_valido & ~y_reg.isna()
        
        # Combinar máscaras
        mask_final = mask_linhas_validas & mask_target_valido
        
        linhas_removidas = (~mask_final).sum()
        
        if linhas_removidas > 0:
            self._log(f"   🧹 Removendo {linhas_removidas} linhas com muitos NaN ou target inválido")
            X = X[mask_final]
            y_class = y_class[mask_final]
            if y_reg is not None:
                y_reg = y_reg[mask_final]
        
        # Preencher NaN restantes com a mediana de cada coluna
        nan_restantes = X.isna().sum().sum()
        if nan_restantes > 0:
            self._log(f"   💉 Preenchendo {nan_restantes} NaN restantes com a mediana...")
            for col in X.columns:
                if X[col].isna().any():
                    mediana = X[col].median()
                    if pd.notna(mediana):
                        X[col] = X[col].fillna(mediana)
                    else:
                        # Se a mediana for NaN (coluna toda NaN), usar 0
                        X[col] = X[col].fillna(0)
        
        # Reset dos índices
        X = X.reset_index(drop=True)
        y_class = y_class.reset_index(drop=True)
        if y_reg is not None:
            y_reg = y_reg.reset_index(drop=True)
        
        # Verificação final
        if len(X) == 0:
            raise ValueError("❌ ERRO: Todos os registros foram removidos! Verifique os dados de entrada.")
        
        # Estatísticas finais
        self._log(f"   ✅ Dados preparados:")
        self._log(f"      - Registros finais: {len(X)}")
        self._log(f"      - Features: {len(X.columns)}")
        self._log(f"      - Distribuição do target '{target_class}':")
        distribuicao = y_class.value_counts()
        for valor, count in distribuicao.items():
            self._log(f"        • {valor}: {count} ({count/len(y_class)*100:.2f}%)")
        
        return X, y_class, y_reg
    
    def treinar_classificacao(self, X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
        """
        Treina modelo de classificação
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Modelo treinado
        """
        self._log("🚀 Treinando modelo de classificação (Random Forest)...")
        
        # Verificar se há dados suficientes
        if len(X) < 10:
            raise ValueError(f"Dados insuficientes para treinamento: apenas {len(X)} registros")
        
        # Dividir dados
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError as e:
            # Se stratify falhar (classes muito desbalanceadas), treinar sem stratify
            self._log("   ⚠️ Não foi possível usar stratify (classes muito desbalanceadas)")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        self._log(f"   📊 Conjunto de treino: {len(X_train)} registros")
        self._log(f"   📊 Conjunto de teste: {len(X_test)} registros")
        
        # Treinar modelo
        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1,
            max_depth=10,  # Limitar profundidade para evitar overfitting
            min_samples_split=5
        )
        model.fit(X_train, y_train)
        
        # Avaliar
        y_pred = model.predict(X_test)
        
        # Calcular métricas (com zero_division para evitar warnings)
        self.metrics['classificacao'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0)
        }
        
        self._log(f"   ✅ Métricas de Classificação:")
        for metric, value in self.metrics['classificacao'].items():
            self._log(f"      - {metric.capitalize()}: {value:.4f}")
        
        self.models['classificacao'] = model
        return model
    
    def treinar_regressao(self, X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
        """
        Treina modelo de regressão
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Modelo treinado
        """
        self._log("🚀 Treinando modelo de regressão (Random Forest)...")
        
        # Verificar se há dados suficientes
        if len(X) < 10:
            raise ValueError(f"Dados insuficientes para treinamento: apenas {len(X)} registros")
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self._log(f"   📊 Conjunto de treino: {len(X_train)} registros")
        self._log(f"   📊 Conjunto de teste: {len(X_test)} registros")
        
        # Treinar modelo
        model = RandomForestRegressor(
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1,
            max_depth=10,
            min_samples_split=5
        )
        model.fit(X_train, y_train)
        
        # Avaliar (usando R² como métrica principal)
        r2 = model.score(X_test, y_test)
        
        # Calcular RMSE
        from sklearn.metrics import mean_squared_error
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        self.metrics['regressao'] = {
            'r2_score': r2,
            'rmse': rmse
        }
        
        self._log(f"   ✅ Métricas de Regressão:")
        self._log(f"      - R² Score: {r2:.4f}")
        self._log(f"      - RMSE: {rmse:.4f}")
        
        self.models['regressao'] = model
        return model
    
    def plotar_importancia_features(self, model: RandomForestClassifier, 
                                   X: pd.DataFrame, 
                                   top_n: int = 15, 
                                   salvar: bool = True):
        """
        Plota gráfico de importância das features
        
        Args:
            model: Modelo treinado
            X: DataFrame de features
            top_n: Número de features a exibir
            salvar: Se True, salva o gráfico
        """
        self._log("📈 Gerando gráfico de importância das features...")
        
        importances = model.feature_importances_
        feature_names = X.columns
        feature_importance = pd.DataFrame({
            'Feature': feature_names, 
            'Importance': importances
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
        plt.title('Top Features Mais Importantes - Modelo de Classificação', fontsize=16, fontweight='bold')
        plt.xlabel('Importância', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        if salvar:
            caminho = Path('../data/plots/feature_importance.png')
            caminho.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(caminho, dpi=300, bbox_inches='tight')
            self._log(f"   ✅ Gráfico salvo em: {caminho}")
        
        plt.show()
        plt.close()
    
#CORRIGIR salvar_modelo

    # def salvar_modelo(self, model, nome: str, caminho: str = '../data/models/'):
    #     """
    #     Salva o modelo treinado
        
    #     Args:
    #         model: Modelo treinado
    #         nome: Nome do arquivo
    #         caminho: Diretório de destino
    #     """
    #     caminho_completo = Path(caminho) / f"{nome}.joblib"
    #     caminho_completo.parent.mkdir(parents=True, exist_ok=True)
    #     joblib.dump(model, caminho_completo)
    #     self._log(f"💾 Modelo salvo em: {caminho_completo}")

def treinar_modelos(caminho_dados: str, 
                   target_class: str = 'Chuva',
                   target_reg: Optional[str] = None,
                   plotar: bool = True):
    """
    Função principal para treinar modelos
    
    Args:
        caminho_dados: Caminho para dados pré-processados
        target_class: Coluna alvo para classificação
        target_reg: Coluna alvo para regressão
        plotar: Se True, plota importância das features
    """
    print("=" * 80)
    print("🚀 INICIANDO TREINAMENTO DE MODELOS")
    print("=" * 80)
    
    # Carregar dados
    print(f"📂 Carregando dados de: {caminho_dados}")
    df = pd.read_csv(caminho_dados)
    print(f"✅ Dados carregados: {len(df)} registros, {len(df.columns)} colunas")
    
    # Inicializar treinador
    trainer = WeatherModelTrainer(verbose=True)
    
    try:
        # Preparar dados
        X, y_class, y_reg = trainer.preparar_dados(df, target_class, target_reg)
        
        # Treinar modelo de classificação
        model_class = trainer.treinar_classificacao(X, y_class)
        trainer.salvar_modelo(model_class, 'modelo_classificacao')
        
        # Plotar importância das features
        if plotar:
            trainer.plotar_importancia_features(model_class, X)
        
        # Treinar modelo de regressão (se especificado)
        if target_reg and y_reg is not None:
            model_reg = trainer.treinar_regressao(X, y_reg)
            trainer.salvar_modelo(model_reg, 'modelo_regressao')
        
        print("=" * 80)
        print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO")
        print("=" * 80)
        
    except Exception as e:
        print("=" * 80)
        print(f"❌ ERRO DURANTE O TREINAMENTO: {e}")
        print("=" * 80)
        raise

if __name__ == '__main__':
    """
    Exemplo de uso direto do módulo
    """
    print("Módulo de treinamento carregado com sucesso!")
    print("\nExemplo de uso:")
    print("""
    from src.train import treinar_modelos
    
    # Treinar modelos
    treinar_modelos(
        caminho_dados='data/dados_processados_ml.csv',
        target_class='Chuva',
        target_reg='PRECIPITAÇÃO TOTAL, HORÁRIO (mm)',
        plotar=True
    )
    """)
    
    # Executar treinamento (se o arquivo existir)
    from pathlib import Path
    caminho_dados = str(Path(__file__).parent.parent / 'data' / 'dados_processados_ml.csv')
    
    if Path(caminho_dados).exists():
        treinar_modelos(
            caminho_dados=caminho_dados,
            target_class='Chuva',
            target_reg='PRECIPITAÇÃO TOTAL, HORÁRIO (mm)',
            plotar=True
        )
    else:
        print(f"\n⚠️ Arquivo de dados não encontrado: {caminho_dados}")
        print("Execute primeiro o pré-processamento dos dados!")
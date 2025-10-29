"""
Módulo de Treinamento de Modelos para Previsão Meteorológica
Autor: PI4-MachineLearning-2025
Descrição: Treinamento de modelos de classificação (chuva sim/não) e regressão (quantidade de chuva)
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from typing import Optional, Tuple, Dict
from datetime import datetime

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
        self.feature_names = []
        
        # Obter diretórios do projeto
        self.project_root = Path(__file__).resolve().parent.parent
        self.data_dir = self.project_root / 'data'
        self.models_dir = self.data_dir / 'models'
        self.plots_dir = self.data_dir / 'plots'
        
        # Criar diretórios se não existirem
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        self._log(f"📁 Diretório do projeto: {self.project_root}")
        self._log(f"📁 Diretório de modelos: {self.models_dir}")
        self._log(f"📁 Diretório de gráficos: {self.plots_dir}")
    
    def _log(self, message: str):
        """Imprime mensagem se verbose=True"""
        if self.verbose:
            print(message)
    
    def preparar_dados(self, df: pd.DataFrame, 
                      target_class: str = 'Chuva',
                      target_reg: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
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
        self._log("\n" + "="*80)
        self._log("📊 PREPARANDO DADOS PARA TREINAMENTO")
        self._log("="*80)
        self._log(f"📈 Dados iniciais: {len(df)} registros, {len(df.columns)} colunas")
        
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
        
        self._log(f"🗑️  Colunas excluídas ({len(colunas_excluir)}): {colunas_excluir[:5]}{'...' if len(colunas_excluir) > 5 else ''}")
        
        # Selecionar features
        colunas_features = [col for col in df_work.columns if col not in colunas_excluir]
        
        self._log(f"🎯 Features candidatas: {len(colunas_features)}")
        
        # Processar features categóricas
        colunas_codificadas = []
        for col in colunas_features.copy():
            if df_work[col].dtype == 'object':
                try:
                    le = LabelEncoder()
                    df_work[col] = le.fit_transform(df_work[col].astype(str))
                    self.label_encoders[col] = le
                    colunas_codificadas.append(col)
                except Exception as e:
                    self._log(f"⚠️  Erro ao codificar {col}: {e}. Removendo coluna.")
                    colunas_features.remove(col)
        
        if colunas_codificadas:
            self._log(f"🔤 Colunas categóricas codificadas ({len(colunas_codificadas)}): {colunas_codificadas}")
        
        # Criar DataFrame de features
        X = df_work[colunas_features].copy()
        
        # Verificar e preparar targets
        if target_class not in df_work.columns:
            raise ValueError(f"Coluna target '{target_class}' não encontrada no DataFrame!")
        
        y_class = df_work[target_class].copy()
        y_reg = df_work[target_reg].copy() if target_reg and target_reg in df_work.columns else None
        
        # Análise de NaN
        self._log(f"\n🔍 ANÁLISE DE VALORES FALTANTES:")
        nan_por_coluna = X.isna().sum()
        colunas_com_nan = nan_por_coluna[nan_por_coluna > 0]
        
        if len(colunas_com_nan) > 0:
            self._log(f"   Colunas com NaN: {len(colunas_com_nan)}/{len(X.columns)}")
            self._log(f"   Top 5 colunas com mais NaN:")
            for col, count in colunas_com_nan.sort_values(ascending=False).head(5).items():
                pct = (count / len(X)) * 100
                self._log(f"      • {col[:50]}: {count} ({pct:.1f}%)")
        else:
            self._log(f"   ✅ Nenhuma coluna com NaN")
        
        # Estratégia de limpeza
        nan_por_linha = X.isna().sum(axis=1)
        linhas_com_nan = (nan_por_linha > 0).sum()
        self._log(f"\n   Linhas com pelo menos 1 NaN: {linhas_com_nan} ({linhas_com_nan/len(X)*100:.1f}%)")
        self._log(f"   Total de valores NaN: {X.isna().sum().sum()}")
        
        # Remover linhas com muitos NaNs (>50% das features)
        limite_nan = len(X.columns) * 0.5
        mask_linhas_validas = nan_por_linha <= limite_nan
        
        # Garantir que o target não é NaN
        mask_target_valido = ~y_class.isna()
        if y_reg is not None:
            mask_target_valido = mask_target_valido & ~y_reg.isna()
        
        # Combinar máscaras
        mask_final = mask_linhas_validas & mask_target_valido
        linhas_removidas = (~mask_final).sum()
        
        if linhas_removidas > 0:
            self._log(f"\n🧹 Removendo {linhas_removidas} linhas ({linhas_removidas/len(X)*100:.1f}%)")
            X = X[mask_final]
            y_class = y_class[mask_final]
            if y_reg is not None:
                y_reg = y_reg[mask_final]
        
        # Preencher NaN restantes com a mediana
        nan_restantes_antes = X.isna().sum().sum()
        if nan_restantes_antes > 0:
            self._log(f"\n💉 Preenchendo {nan_restantes_antes} NaN restantes com mediana...")
            for col in X.columns:
                if X[col].isna().any():
                    mediana = X[col].median()
                    if pd.notna(mediana):
                        X[col] = X[col].fillna(mediana)
                    else:
                        X[col] = X[col].fillna(0)
            
            nan_restantes_depois = X.isna().sum().sum()
            self._log(f"   ✅ NaN restantes após preenchimento: {nan_restantes_depois}")
        
        # Reset dos índices
        X = X.reset_index(drop=True)
        y_class = y_class.reset_index(drop=True)
        if y_reg is not None:
            y_reg = y_reg.reset_index(drop=True)
        
        # Armazenar nomes das features
        self.feature_names = X.columns.tolist()
        
        # Verificação final
        if len(X) == 0:
            raise ValueError("❌ ERRO: Todos os registros foram removidos! Verifique os dados de entrada.")
        
        # Estatísticas finais
        self._log(f"\n✅ DADOS PREPARADOS COM SUCESSO:")
        self._log(f"   • Registros finais: {len(X)}")
        self._log(f"   • Features: {len(X.columns)}")
        self._log(f"   • Valores faltantes: {X.isna().sum().sum()}")
        
        self._log(f"\n📊 DISTRIBUIÇÃO DO TARGET '{target_class}':")
        distribuicao = y_class.value_counts().sort_index()
        for valor, count in distribuicao.items():
            pct = (count / len(y_class)) * 100
            barra = "█" * int(pct / 2)
            self._log(f"   • Classe {valor}: {count:6d} ({pct:5.2f}%) {barra}")
        
        # Calcular balanceamento
        classes = distribuicao.values
        if len(classes) == 2:
            ratio = min(classes) / max(classes)
            self._log(f"   • Razão de balanceamento: {ratio:.3f} (1.0 = perfeito)")
            if ratio < 0.3:
                self._log(f"   ⚠️  Classes muito desbalanceadas! Considere usar SMOTE ou class_weight")
        
        self._log("="*80 + "\n")
        
        return X, y_class, y_reg
    
    def treinar_classificacao(self, X: pd.DataFrame, y: pd.Series, 
                             cv_folds: int = 5) -> RandomForestClassifier:
        """
        Treina modelo de classificação com validação cruzada
        
        Args:
            X: Features
            y: Target
            cv_folds: Número de folds para validação cruzada
            
        Returns:
            Modelo treinado
        """
        self._log("="*80)
        self._log("🚀 TREINANDO MODELO DE CLASSIFICAÇÃO")
        self._log("="*80)
        
        # Verificar se há dados suficientes
        if len(X) < 10:
            raise ValueError(f"Dados insuficientes para treinamento: apenas {len(X)} registros")
        
        # Dividir dados
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            self._log("✅ Split estratificado aplicado com sucesso")
        except ValueError as e:
            self._log("⚠️  Stratify falhou (classes desbalanceadas), usando split aleatório")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        self._log(f"\n📊 DIVISÃO DOS DADOS:")
        self._log(f"   • Treino: {len(X_train)} registros ({len(X_train)/len(X)*100:.1f}%)")
        self._log(f"   • Teste:  {len(X_test)} registros ({len(X_test)/len(X)*100:.1f}%)")
        
        # Configurar modelo
        self._log(f"\n🔧 CONFIGURANDO MODELO:")
        self._log(f"   • Algoritmo: Random Forest Classifier")
        self._log(f"   • N° de árvores: 100")
        self._log(f"   • Profundidade máxima: 10")
        self._log(f"   • Min samples split: 5")
        
        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced'  # Ajustar para classes desbalanceadas
        )
        
        # Validação cruzada
        self._log(f"\n🔄 VALIDAÇÃO CRUZADA ({cv_folds} folds)...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='f1')
        self._log(f"   • F1-Score médio: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # Treinar modelo
        self._log(f"\n⏳ Treinando modelo final...")
        model.fit(X_train, y_train)
        self._log(f"✅ Modelo treinado com sucesso!")
        
        # Fazer predições
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calcular métricas
        self._log(f"\n📈 MÉTRICAS DE DESEMPENHO:")
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        self._log(f"   • Acurácia:  {accuracy:.4f}")
        self._log(f"   • Precisão:  {precision:.4f}")
        self._log(f"   • Recall:    {recall:.4f}")
        self._log(f"   • F1-Score:  {f1:.4f}")
        
        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        self._log(f"\n🎯 MATRIZ DE CONFUSÃO:")
        self._log(f"   {cm}")
        
        # Interpretação
        if len(cm) == 2:
            tn, fp, fn, tp = cm.ravel()
            self._log(f"\n   Verdadeiros Negativos (TN): {tn}")
            self._log(f"   Falsos Positivos (FP):      {fp}")
            self._log(f"   Falsos Negativos (FN):      {fn}")
            self._log(f"   Verdadeiros Positivos (TP): {tp}")
        
        # Armazenar métricas
        self.metrics['classificacao'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_scores': cv_scores,
            'confusion_matrix': cm,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        self.models['classificacao'] = model
        self._log("="*80 + "\n")
        
        return model
    
    def treinar_regressao(self, X: pd.DataFrame, y: pd.Series,
                         cv_folds: int = 5) -> RandomForestRegressor:
        """
        Treina modelo de regressão com validação cruzada
        
        Args:
            X: Features
            y: Target
            cv_folds: Número de folds para validação cruzada
            
        Returns:
            Modelo treinado
        """
        self._log("="*80)
        self._log("🚀 TREINANDO MODELO DE REGRESSÃO")
        self._log("="*80)
        
        # Verificar se há dados suficientes
        if len(X) < 10:
            raise ValueError(f"Dados insuficientes para treinamento: apenas {len(X)} registros")
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self._log(f"\n📊 DIVISÃO DOS DADOS:")
        self._log(f"   • Treino: {len(X_train)} registros ({len(X_train)/len(X)*100:.1f}%)")
        self._log(f"   • Teste:  {len(X_test)} registros ({len(X_test)/len(X)*100:.1f}%)")
        
        # Configurar modelo
        self._log(f"\n🔧 CONFIGURANDO MODELO:")
        self._log(f"   • Algoritmo: Random Forest Regressor")
        self._log(f"   • N° de árvores: 100")
        self._log(f"   • Profundidade máxima: 10")
        
        model = RandomForestRegressor(
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1,
            max_depth=10,
            min_samples_split=5
        )
        
        # Validação cruzada
        self._log(f"\n🔄 VALIDAÇÃO CRUZADA ({cv_folds} folds)...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
        self._log(f"   • R² médio: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # Treinar modelo
        self._log(f"\n⏳ Treinando modelo final...")
        model.fit(X_train, y_train)
        self._log(f"✅ Modelo treinado com sucesso!")
        
        # Fazer predições
        y_pred = model.predict(X_test)
        
        # Calcular métricas
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = np.mean(np.abs(y_test - y_pred))
        
        self._log(f"\n📈 MÉTRICAS DE DESEMPENHO:")
        self._log(f"   • R² Score: {r2:.4f}")
        self._log(f"   • RMSE:     {rmse:.4f}")
        self._log(f"   • MAE:      {mae:.4f}")
        
        # Armazenar métricas
        self.metrics['regressao'] = {
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae,
            'cv_scores': cv_scores,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        self.models['regressao'] = model
        self._log("="*80 + "\n")
        
        return model
    
    def plotar_importancia_features(self, tipo: str = 'classificacao',
                                   top_n: int = 15, 
                                   salvar: bool = True) -> Path:
        """
        Plota gráfico de importância das features
        
        Args:
            tipo: 'classificacao' ou 'regressao'
            top_n: Número de features a exibir
            salvar: Se True, salva o gráfico
            
        Returns:
            Caminho do arquivo salvo
        """
        self._log(f"📈 Gerando gráfico de importância das features ({tipo})...")
        
        model = self.models.get(tipo)
        if model is None:
            self._log(f"⚠️  Modelo de {tipo} não encontrado!")
            return None
        
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names, 
            'Importance': importances
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
        plt.title(f'Top {top_n} Features Mais Importantes - {tipo.capitalize()}', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Importância', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        if salvar:
            caminho = self.plots_dir / f'feature_importance_{tipo}.png'
            plt.savefig(caminho, dpi=300, bbox_inches='tight')
            self._log(f"✅ Gráfico salvo em: {caminho}")
        
        plt.show()
        plt.close()
        
        return caminho if salvar else None
    
    def plotar_matriz_confusao(self, salvar: bool = True) -> Path:
        """Plota matriz de confusão do modelo de classificação"""
        self._log("📊 Gerando matriz de confusão...")
        
        metrics = self.metrics.get('classificacao')
        if metrics is None:
            self._log("⚠️  Métricas de classificação não encontradas!")
            return None
        
        cm = metrics['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Sem Chuva', 'Com Chuva'],
                   yticklabels=['Sem Chuva', 'Com Chuva'])
        plt.title('Matriz de Confusão - Classificação', fontsize=14, fontweight='bold')
        plt.ylabel('Valor Real', fontsize=12)
        plt.xlabel('Valor Predito', fontsize=12)
        plt.tight_layout()
        
        if salvar:
            caminho = self.plots_dir / 'confusion_matrix.png'
            plt.savefig(caminho, dpi=300, bbox_inches='tight')
            self._log(f"✅ Gráfico salvo em: {caminho}")
        
        plt.show()
        plt.close()
        
        return caminho if salvar else None
    
    def salvar_modelo(self, model, nome: str) -> Optional[Path]:
        """
        Salva o modelo treinado usando caminho absoluto
        
        Args:
            model: Modelo treinado
            nome: Nome do arquivo (sem extensão)
            
        Returns:
            Caminho absoluto do arquivo salvo ou None se falhar
        """
        caminho_completo = self.models_dir / f"{nome}.joblib"
        
        self._log(f"\n💾 SALVANDO MODELO:")
        self._log(f"   • Nome: {nome}")
        self._log(f"   • Caminho: {caminho_completo}")
        self._log(f"   • Diretório existe: {self.models_dir.exists()}")
        self._log(f"   • Permissão de escrita: {os.access(self.models_dir, os.W_OK)}")
        
        try:
            # Salvar modelo
            joblib.dump(model, caminho_completo)
            
            # Verificar se salvou
            if caminho_completo.exists():
                tamanho_bytes = caminho_completo.stat().st_size
                tamanho_mb = tamanho_bytes / (1024 * 1024)
                
                if tamanho_bytes > 0:
                    self._log(f"✅ Modelo salvo com sucesso!")
                    self._log(f"   • Tamanho: {tamanho_mb:.2f} MB")
                    self._log(f"   • Caminho absoluto: {caminho_completo.absolute()}")
                    return caminho_completo
                else:
                    self._log(f"❌ Arquivo criado mas está vazio!")
                    return None
            else:
                self._log(f"❌ Arquivo não foi criado!")
                return None
                
        except PermissionError:
            self._log(f"❌ Erro de permissão! Execute como administrador ou verifique permissões.")
            return None
        except Exception as e:
            self._log(f"❌ Erro inesperado ao salvar: {type(e).__name__}: {e}")
            return None
    
    def salvar_relatorio(self, nome_arquivo: str = 'relatorio_treinamento.txt'):
        """Salva relatório detalhado do treinamento"""
        caminho = self.data_dir / nome_arquivo
        
        self._log(f"\n📝 Gerando relatório de treinamento...")
        
        with open(caminho, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RELATÓRIO DE TREINAMENTO DE MODELOS\n")
            f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # Métricas de classificação
            if 'classificacao' in self.metrics:
                f.write("MODELO DE CLASSIFICAÇÃO\n")
                f.write("-"*40 + "\n")
                metrics = self.metrics['classificacao']
                f.write(f"Acurácia:  {metrics['accuracy']:.4f}\n")
                f.write(f"Precisão:  {metrics['precision']:.4f}\n")
                f.write(f"Recall:    {metrics['recall']:.4f}\n")
                f.write(f"F1-Score:  {metrics['f1_score']:.4f}\n\n")
                
                f.write("Validação Cruzada:\n")
                cv = metrics['cv_scores']
                f.write(f"  Média: {cv.mean():.4f}\n")
                f.write(f"  Desvio Padrão: {cv.std():.4f}\n")
                f.write(f"  Scores: {cv}\n\n")
            
            # Métricas de regressão
            if 'regressao' in self.metrics:
                f.write("MODELO DE REGRESSÃO\n")
                f.write("-"*40 + "\n")
                metrics = self.metrics['regressao']
                f.write(f"R² Score: {metrics['r2_score']:.4f}\n")
                f.write(f"RMSE:     {metrics['rmse']:.4f}\n")
                f.write(f"MAE:      {metrics['mae']:.4f}\n\n")
            
            # Features
            f.write("FEATURES UTILIZADAS\n")
            f.write("-"*40 + "\n")
            f.write(f"Total: {len(self.feature_names)}\n\n")
            for i, feat in enumerate(self.feature_names, 1):
                f.write(f"{i:3d}. {feat}\n")
        
        self._log(f"✅ Relatório salvo em: {caminho}")
        return caminho


def treinar_modelos(caminho_dados: str, 
                   target_class: str = 'Chuva',
                   target_reg: Optional[str] = None,
                   plotar: bool = True,
                   salvar_relatorio: bool = True):
    """
    Função principal para treinar modelos
    
    Args:
        caminho_dados: Caminho para dados pré-processados
        target_class: Coluna alvo para classificação
        target_reg: Coluna alvo para regressão
        plotar: Se True, plota gráficos
        salvar_relatorio: Se True, salva relatório
    """
    print("\n" + "="*80)
    print("🚀 SISTEMA DE TREINAMENTO DE MODELOS METEOROLÓGICOS")
    print("="*80)
    print(f"⏰ Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Carregar dados
    print(f"📂 Carregando dados de: {caminho_dados}")
    
    if not Path(caminho_dados).exists():
        print(f"❌ ERRO: Arquivo não encontrado: {caminho_dados}")
        return None
    
    df = pd.read_csv(caminho_dados)
    print(f"✅ Dados carregados: {len(df)} registros, {len(df.columns)} colunas\n")
    
    # Inicializar treinador
    trainer = WeatherModelTrainer(verbose=True)
    
    try:
        # Preparar dados
        X, y_class, y_reg = trainer.preparar_dados(df, target_class, target_reg)
        
        # Treinar modelo de classificação
        print("🎯 Iniciando treinamento de classificação...\n")
        model_class = trainer.treinar_classificacao(X, y_class)
        
        # Salvar modelo
        caminho_modelo = trainer.salvar_modelo(model_class, 'modelo_classificacao')
        
        # Plotar gráficos
        if plotar:
            trainer.plotar_importancia_features('classificacao')
            trainer.plotar_matriz_confusao()
        
        # Treinar modelo de regressão (se especificado)
        if target_reg and y_reg is not None:
            print("🎯 Iniciando treinamento de regressão...\n")
            model_reg = trainer.treinar_regressao(X, y_reg)
            trainer.salvar_modelo(model_reg, 'modelo_regressao')
            
            if plotar:
                trainer.plotar_importancia_features('regressao')
        
        # Salvar relatório
        if salvar_relatorio:
            trainer.salvar_relatorio()
        
        print("\n" + "="*80)
        print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
        print("="*80)
        print(f"⏰ Término: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n📊 RESUMO:")
        print(f"   • Modelo de classificação: {'✅ Treinado' if 'classificacao' in trainer.models else '❌ Falhou'}")
        print(f"   • Modelo de regressão: {'✅ Treinado' if 'regressao' in trainer.models else '⏭️  Não solicitado'}")
        print(f"   • Modelos salvos em: {trainer.models_dir}")
        print(f"   • Gráficos salvos em: {trainer.plots_dir}")
        print("="*80 + "\n")
        
        return trainer
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ ERRO DURANTE O TREINAMENTO")
        print("="*80)
        print(f"Tipo: {type(e).__name__}")
        print(f"Mensagem: {e}")
        print("="*80 + "\n")
        
        import traceback
        traceback.print_exc()
        
        return None


def carregar_modelo(nome: str, diretorio: Optional[str] = None) -> object:
    """
    Carrega um modelo salvo
    
    Args:
        nome: Nome do modelo (com ou sem extensão .joblib)
        diretorio: Diretório dos modelos (usa padrão se None)
        
    Returns:
        Modelo carregado
    """
    if diretorio is None:
        project_root = Path(__file__).resolve().parent.parent
        diretorio = project_root / 'data' / 'models'
    else:
        diretorio = Path(diretorio)
    
    # Adicionar extensão se necessário
    if not nome.endswith('.joblib'):
        nome = f"{nome}.joblib"
    
    caminho = diretorio / nome
    
    print(f"📂 Carregando modelo de: {caminho}")
    
    if not caminho.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {caminho}")
    
    modelo = joblib.load(caminho)
    print(f"✅ Modelo carregado com sucesso!")
    
    return modelo


def fazer_previsao(modelo, dados: pd.DataFrame, 
                  label_encoders: Optional[Dict] = None) -> np.ndarray:
    """
    Faz previsão usando modelo treinado
    
    Args:
        modelo: Modelo treinado
        dados: DataFrame com features (já pré-processadas)
        label_encoders: Dicionário de label encoders (se houver features categóricas)
        
    Returns:
        Array com previsões
    """
    print(f"🔮 Fazendo previsões...")
    print(f"   • Registros: {len(dados)}")
    print(f"   • Features: {len(dados.columns)}")
    
    # Aplicar label encoders se necessário
    if label_encoders:
        dados_encoded = dados.copy()
        for col, encoder in label_encoders.items():
            if col in dados_encoded.columns:
                dados_encoded[col] = encoder.transform(dados_encoded[col].astype(str))
        dados = dados_encoded
    
    # Fazer previsão
    previsoes = modelo.predict(dados)
    
    print(f"✅ Previsões concluídas!")
    
    return previsoes


if __name__ == '__main__':
    """
    Exemplo de uso direto do módulo
    """
    print("="*80)
    print("MÓDULO DE TREINAMENTO - PI4 Machine Learning")
    print("="*80)
    print("\n📚 Exemplo de uso:")
    print("""
    from src.train import treinar_modelos
    
    # Treinar modelos
    trainer = treinar_modelos(
        caminho_dados='data/dados_processados_ml.csv',
        target_class='Chuva',
        target_reg='PRECIPITAÇÃO TOTAL, HORÁRIO (mm)',
        plotar=True,
        salvar_relatorio=True
    )
    """)
    
    # Executar treinamento (se o arquivo existir)
    project_root = Path(__file__).resolve().parent.parent
    caminho_dados = project_root / 'data' / 'dados_processados_ml.csv'
    
    if caminho_dados.exists():
        print("\n✅ Arquivo de dados encontrado!")
        print(f"📁 Caminho: {caminho_dados}")
        print("\n🚀 Iniciando treinamento automático...\n")
        
        trainer = treinar_modelos(
            caminho_dados=str(caminho_dados),
            target_class='Chuva',
            target_reg='PRECIPITAÇÃO TOTAL, HORÁRIO (mm)',
            plotar=True,
            salvar_relatorio=True
        )
        
        if trainer:
            print("\n" + "="*80)
            print("🎉 SISTEMA PRONTO PARA USO!")
            print("="*80)
            print("\n📖 Como usar os modelos salvos:")
            print("""
    from src.train import carregar_modelo, fazer_previsao
    
    # Carregar modelo
    modelo = carregar_modelo('modelo_classificacao')
    
    # Fazer previsão (dados já pré-processados)
    previsoes = fazer_previsao(modelo, df_novos_dados)
            """)
    else:
        print(f"\n⚠️  Arquivo de dados não encontrado: {caminho_dados}")
        print("\n📋 Para treinar os modelos:")
        print("   1. Execute o pré-processamento primeiro:")
        print("      python -m notebooks.exemplo_preprocess")
        print("   2. Depois execute o treinamento:")
        print("      python src/train.py")
        print("\n" + "="*80)
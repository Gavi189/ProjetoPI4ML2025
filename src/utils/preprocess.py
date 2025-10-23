"""
Módulo de Pré-processamento Automatizado para Dados Meteorológicos INMET
Autor: PI4-MachineLearning-2025
Descrição: Pipeline completo de limpeza, transformação e feature engineering
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class DataCleaner:
    """Classe para limpeza e conversão de dados brutos"""
    
    def __init__(self, verbose: bool = True):
        """
        Inicializa o limpador de dados
        
        Args:
            verbose: Se True, imprime informações durante processamento
        """
        self.verbose = verbose
        self.colunas_temporais = ['Data', 'Hora', 'Data Medicao', 'Hora Medicao', 
                                  'DATA (YYYY-MM-DD)', 'HORA (UTC)']
    
    def _log(self, message: str):
        """Imprime mensagem se verbose=True"""
        if self.verbose:
            print(message)
    
    def converter_para_numerico(self, df: pd.DataFrame, 
                                excluir_colunas: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Converte colunas para formato numérico
        
        Args:
            df: DataFrame original
            excluir_colunas: Lista de colunas a não converter
            
        Returns:
            DataFrame com colunas numéricas convertidas
        """
        self._log("🔧 Convertendo colunas para formato numérico...")
        
        df_clean = df.copy()
        
        # Definir colunas a excluir
        if excluir_colunas is None:
            excluir_colunas = self.colunas_temporais
        
        # Colunas a converter
        colunas_para_converter = [col for col in df_clean.columns 
                                  if col not in excluir_colunas]
        
        conversoes_sucesso = 0
        conversoes_falha = []
        
        for col in colunas_para_converter:
            try:
                # Substituir vírgulas por pontos (formato brasileiro)
                df_clean[col] = pd.to_numeric(
                    df_clean[col].astype(str).str.replace(',', '.').str.strip(),
                    errors='coerce'
                )
                
                if df_clean[col].dtype in ['float64', 'int64']:
                    conversoes_sucesso += 1
                else:
                    conversoes_falha.append(col)
            except Exception as e:
                conversoes_falha.append(f"{col} (erro: {str(e)[:50]})")
        
        self._log(f"   ✅ {conversoes_sucesso} colunas convertidas com sucesso")
        if conversoes_falha and self.verbose:
            self._log(f"   ⚠️ {len(conversoes_falha)} colunas não convertidas")
        
        return df_clean
    
    def remover_valores_invalidos(self, df: pd.DataFrame, 
                                  remover_inf: bool = True) -> pd.DataFrame:
        """
        Remove ou substitui valores inválidos (inf, -inf, outliers extremos)
        
        Args:
            df: DataFrame a limpar
            remover_inf: Se True, substitui inf/-inf por NaN
            
        Returns:
            DataFrame limpo
        """
        self._log("🧹 Removendo valores inválidos...")
        
        df_clean = df.copy()
        
        if remover_inf:
            # Substituir inf/-inf por NaN
            df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
            self._log("   ✅ Valores infinitos substituídos por NaN")
        
        # Identificar colunas numéricas
        colunas_numericas = df_clean.select_dtypes(include=[np.number]).columns
        
        total_removidos = 0
        for col in colunas_numericas:
            # Remover outliers extremos (além de 10 desvios padrão)
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            
            if pd.notna(std) and std > 0:
                mask_outliers = np.abs(df_clean[col] - mean) > (10 * std)
                n_outliers = mask_outliers.sum()
                
                if n_outliers > 0:
                    df_clean.loc[mask_outliers, col] = np.nan
                    total_removidos += n_outliers
        
        if total_removidos > 0:
            self._log(f"   ✅ {total_removidos} outliers extremos removidos")
        
        return df_clean
    
    def tratar_valores_nulos(self, df: pd.DataFrame, 
                            metodo: str = 'interpolate',
                            limite_interpolacao: int = 3) -> pd.DataFrame:
        """
        Trata valores nulos usando diferentes estratégias
        
        Args:
            df: DataFrame com valores nulos
            metodo: 'interpolate', 'ffill', 'bfill', 'mean', 'median', 'drop'
            limite_interpolacao: Número máximo de NaNs consecutivos a interpolar
            
        Returns:
            DataFrame com NaNs tratados
        """
        self._log(f"💉 Tratando valores nulos (método: {metodo})...")
        
        df_clean = df.copy()
        colunas_numericas = df_clean.select_dtypes(include=[np.number]).columns
        
        nulos_antes = df_clean[colunas_numericas].isnull().sum().sum()
        
        for col in colunas_numericas:
            if df_clean[col].isnull().sum() > 0:
                if metodo == 'interpolate':
                    df_clean[col] = df_clean[col].interpolate(
                        method='linear', 
                        limit=limite_interpolacao,
                        limit_direction='both'
                    )
                elif metodo == 'ffill':
                    df_clean[col] = df_clean[col].fillna(method='ffill', limit=limite_interpolacao)
                elif metodo == 'bfill':
                    df_clean[col] = df_clean[col].fillna(method='bfill', limit=limite_interpolacao)
                elif metodo == 'mean':
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                elif metodo == 'median':
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                elif metodo == 'drop':
                    # Será tratado depois
                    pass
        
        nulos_depois = df_clean[colunas_numericas].isnull().sum().sum()
        nulos_tratados = nulos_antes - nulos_depois
        
        self._log(f"   ✅ {nulos_tratados} valores nulos tratados")
        
        if metodo == 'drop' and nulos_depois > 0:
            linhas_antes = len(df_clean)
            df_clean = df_clean.dropna(subset=colunas_numericas)
            linhas_removidas = linhas_antes - len(df_clean)
            self._log(f"   ✅ {linhas_removidas} linhas com NaN removidas")
        
        return df_clean


class DateTimeTransformer:
    """Classe para transformação de colunas de data e hora"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def _log(self, message: str):
        if self.verbose:
            print(message)
    
    def identificar_colunas_temporais(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Identifica automaticamente colunas de data e hora
        
        Returns:
            Dicionário com {'data': nome_coluna, 'hora': nome_coluna}
        """
        colunas_data = []
        colunas_hora = []
        
        # Palavras-chave para identificar
        keywords_data = ['DATA', 'DATE', 'DIA', 'DAY']
        keywords_hora = ['HORA', 'HOUR', 'TIME', 'TEMPO']
        
        for col in df.columns:
            col_upper = col.upper()
            
            # Verificar se é coluna de data
            if any(kw in col_upper for kw in keywords_data):
                colunas_data.append(col)
            
            # Verificar se é coluna de hora
            if any(kw in col_upper for kw in keywords_hora) and 'HORARIA' not in col_upper:
                colunas_hora.append(col)
        
        resultado = {}
        if colunas_data:
            resultado['data'] = colunas_data[0]
        if colunas_hora:
            resultado['hora'] = colunas_hora[0]
        
        return resultado
    
    def criar_datetime(self, df: pd.DataFrame, 
                      col_data: Optional[str] = None,
                      col_hora: Optional[str] = None,
                      formato_data: str = '%Y/%m/%d',
                      formato_hora: str = '%H%M') -> pd.DataFrame:
        """
        Cria coluna datetime unificada a partir de data e hora
        
        Args:
            df: DataFrame original
            col_data: Nome da coluna de data (auto-detecta se None)
            col_hora: Nome da coluna de hora (auto-detecta se None)
            formato_data: Formato da data
            formato_hora: Formato da hora
            
        Returns:
            DataFrame com coluna 'datetime' adicionada
        """
        self._log("📅 Criando coluna datetime unificada...")
        
        df_clean = df.copy()
        
        # Auto-detectar colunas se não fornecidas
        if col_data is None or col_hora is None:
            colunas_temp = self.identificar_colunas_temporais(df_clean)
            col_data = col_data or colunas_temp.get('data')
            col_hora = col_hora or colunas_temp.get('hora')
        
        if col_data is None:
            self._log("   ⚠️ Coluna de data não encontrada")
            return df_clean
        
        try:
            # Converter data
            df_clean['data_temp'] = pd.to_datetime(df_clean[col_data], format=formato_data, errors='coerce')
            
            # Se houver hora, combinar
            if col_hora is not None and col_hora in df_clean.columns:
                # Converter hora para string formatada
                df_clean['hora_temp'] = df_clean[col_hora].astype(str).str.zfill(4)
                
                # Criar datetime completo
                df_clean['datetime'] = pd.to_datetime(
                    df_clean['data_temp'].astype(str) + ' ' + 
                    df_clean['hora_temp'].str[:2] + ':' + 
                    df_clean['hora_temp'].str[2:4],
                    errors='coerce'
                )
                
                # Remover colunas temporárias
                df_clean = df_clean.drop(['data_temp', 'hora_temp'], axis=1)
            else:
                df_clean['datetime'] = df_clean['data_temp']
                df_clean = df_clean.drop('data_temp', axis=1)
            
            self._log(f"   ✅ Coluna 'datetime' criada com sucesso")
            
            # Ordenar por datetime
            df_clean = df_clean.sort_values('datetime').reset_index(drop=True)
            self._log(f"   ✅ Dados ordenados por timestamp")
            
        except Exception as e:
            self._log(f"   ❌ Erro ao criar datetime: {str(e)}")
        
        return df_clean


class FeatureEngineer:
    """Classe para criação de features derivadas"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def _log(self, message: str):
        if self.verbose:
            print(message)
    
    def criar_features_temporais(self, df: pd.DataFrame, 
                                col_datetime: str = 'datetime') -> pd.DataFrame:
        """
        Cria features temporais (ano, mês, dia, hora, dia da semana, etc.)
        
        Args:
            df: DataFrame com coluna datetime
            col_datetime: Nome da coluna datetime
            
        Returns:
            DataFrame com features temporais adicionadas
        """
        self._log("🕐 Criando features temporais...")
        
        df_feat = df.copy()
        
        if col_datetime not in df_feat.columns:
            self._log(f"   ⚠️ Coluna '{col_datetime}' não encontrada")
            return df_feat
        
        dt = df_feat[col_datetime]
        
        # Features básicas
        df_feat['ano'] = dt.dt.year
        df_feat['mes'] = dt.dt.month
        df_feat['dia'] = dt.dt.day
        df_feat['hora'] = dt.dt.hour
        df_feat['dia_semana'] = dt.dt.dayofweek  # 0=segunda, 6=domingo
        df_feat['dia_ano'] = dt.dt.dayofyear
        df_feat['semana_ano'] = dt.dt.isocalendar().week
        
        # Features cíclicas (importante para ML)
        df_feat['hora_sin'] = np.sin(2 * np.pi * df_feat['hora'] / 24)
        df_feat['hora_cos'] = np.cos(2 * np.pi * df_feat['hora'] / 24)
        df_feat['mes_sin'] = np.sin(2 * np.pi * df_feat['mes'] / 12)
        df_feat['mes_cos'] = np.cos(2 * np.pi * df_feat['mes'] / 12)
        
        # Estação do ano (Hemisfério Sul)
        def obter_estacao(mes: int) -> str:
            if mes in [12, 1, 2]:
                return 'verao'
            elif mes in [3, 4, 5]:
                return 'outono'
            elif mes in [6, 7, 8]:
                return 'inverno'
            else:
                return 'primavera'
        
        df_feat['estacao'] = df_feat['mes'].apply(obter_estacao)
        
        # Período do dia
        def obter_periodo(hora: int) -> str:
            if 6 <= hora < 12:
                return 'manha'
            elif 12 <= hora < 18:
                return 'tarde'
            elif 18 <= hora < 24:
                return 'noite'
            else:
                return 'madrugada'
        
        df_feat['periodo_dia'] = df_feat['hora'].apply(obter_periodo)
        
        # Fim de semana
        df_feat['fim_semana'] = (df_feat['dia_semana'] >= 5).astype(int)
        
        self._log(f"   ✅ {12} features temporais criadas")
        
        return df_feat
    
    def criar_features_meteorologicas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features derivadas de variáveis meteorológicas
        
        Args:
            df: DataFrame com dados meteorológicos
            
        Returns:
            DataFrame com features meteorológicas adicionadas
        """
        self._log("🌡️ Criando features meteorológicas...")
        
        df_feat = df.copy()
        features_criadas = 0
        
        # Mapear nomes de colunas (case-insensitive)
        col_map = {col.upper(): col for col in df_feat.columns}
        
        # 1. Amplitude térmica (diferença entre máxima e mínima)
        temp_max_key = next((k for k in col_map if 'TEMPERATURA' in k and 'MAX' in k), None)
        temp_min_key = next((k for k in col_map if 'TEMPERATURA' in k and 'MIN' in k), None)
        
        if temp_max_key and temp_min_key:
            df_feat['amplitude_termica'] = (df_feat[col_map[temp_max_key]] - 
                                           df_feat[col_map[temp_min_key]])
            features_criadas += 1
        
        # 2. Ponto de orvalho - temperatura (Spread)
        temp_key = next((k for k in col_map if 'TEMPERATURA' in k and 'BULBO' in k), None)
        orvalho_key = next((k for k in col_map if 'ORVALHO' in k and 'MAX' not in k and 'MIN' not in k), None)
        
        if temp_key and orvalho_key:
            df_feat['spread_temp_orvalho'] = (df_feat[col_map[temp_key]] - 
                                              df_feat[col_map[orvalho_key]])
            features_criadas += 1
        
        # 3. Variação de pressão
        pressao_max_key = next((k for k in col_map if 'PRESSAO' in k and 'MAX' in k), None)
        pressao_min_key = next((k for k in col_map if 'PRESSAO' in k and 'MIN' in k), None)
        
        if pressao_max_key and pressao_min_key:
            df_feat['variacao_pressao'] = (df_feat[col_map[pressao_max_key]] - 
                                          df_feat[col_map[pressao_min_key]])
            features_criadas += 1
        
        # 4. Variação de umidade
        umid_max_key = next((k for k in col_map if 'UMIDADE' in k and 'MAX' in k), None)
        umid_min_key = next((k for k in col_map if 'UMIDADE' in k and 'MIN' in k), None)
        
        if umid_max_key and umid_min_key:
            df_feat['variacao_umidade'] = (df_feat[col_map[umid_max_key]] - 
                                          df_feat[col_map[umid_min_key]])
            features_criadas += 1
        
        # 5. Índice de desconforto térmico (simplificado)
        temp_key = next((k for k in col_map if 'TEMPERATURA' in k and 'BULBO' in k), None)
        umid_key = next((k for k in col_map if 'UMIDADE RELATIVA' in k and 'MAX' not in k and 'MIN' not in k), None)
        
        if temp_key and umid_key:
            # DI = T - 0.55 * (1 - 0.01 * UR) * (T - 14.5)
            T = df_feat[col_map[temp_key]]
            UR = df_feat[col_map[umid_key]]
            df_feat['indice_desconforto'] = T - 0.55 * (1 - 0.01 * UR) * (T - 14.5)
            features_criadas += 1
        
        # 6. Radiação normalizada (se disponível)
        rad_key = next((k for k in col_map if 'RADIACAO' in k), None)
        if rad_key:
            df_feat['radiacao_norm'] = df_feat[col_map[rad_key]] / 1000  # KJ/m² para MJ/m²
            features_criadas += 1
        
        self._log(f"   ✅ {features_criadas} features meteorológicas criadas")
        
        return df_feat
    
    def criar_lag_features(self, df: pd.DataFrame, 
                          colunas: List[str],
                          lags: List[int] = [1, 2, 3, 6, 12, 24]) -> pd.DataFrame:
        """
        Cria features de lag (valores anteriores) para séries temporais
        
        Args:
            df: DataFrame ordenado temporalmente
            colunas: Lista de colunas para criar lags
            lags: Lista de períodos de lag (ex: [1, 2, 3] = 1h, 2h, 3h atrás)
            
        Returns:
            DataFrame com lag features adicionadas
        """
        self._log(f"⏮️ Criando lag features (lags: {lags})...")
        
        df_feat = df.copy()
        features_criadas = 0
        
        for col in colunas:
            if col in df_feat.columns:
                for lag in lags:
                    nome_feature = f'{col}_lag_{lag}h'
                    df_feat[nome_feature] = df_feat[col].shift(lag)
                    features_criadas += 1
        
        self._log(f"   ✅ {features_criadas} lag features criadas")
        
        return df_feat
    
    def criar_rolling_features(self, df: pd.DataFrame,
                              colunas: List[str],
                              windows: List[int] = [3, 6, 12, 24]) -> pd.DataFrame:
        """
        Cria features de rolling (média móvel, std, etc.)
        
        Args:
            df: DataFrame ordenado temporalmente
            colunas: Lista de colunas para criar rolling features
            windows: Janelas de tempo em horas
            
        Returns:
            DataFrame com rolling features adicionadas
        """
        self._log(f"📊 Criando rolling features (windows: {windows})...")
        
        df_feat = df.copy()
        features_criadas = 0
        
        for col in colunas:
            if col in df_feat.columns:
                for window in windows:
                    # Média móvel
                    df_feat[f'{col}_rolling_mean_{window}h'] = (
                        df_feat[col].rolling(window=window, min_periods=1).mean()
                    )
                    
                    # Desvio padrão móvel
                    df_feat[f'{col}_rolling_std_{window}h'] = (
                        df_feat[col].rolling(window=window, min_periods=1).std()
                    )
                    
                    features_criadas += 2
        
        self._log(f"   ✅ {features_criadas} rolling features criadas")
        
        return df_feat


class WeatherPreprocessor:
    """Pipeline completo de pré-processamento"""
    
    def __init__(self, verbose: bool = True):
        """
        Inicializa o pipeline de pré-processamento
        
        Args:
            verbose: Se True, imprime informações durante processamento
        """
        self.verbose = verbose
        self.cleaner = DataCleaner(verbose=verbose)
        self.datetime_transformer = DateTimeTransformer(verbose=verbose)
        self.feature_engineer = FeatureEngineer(verbose=verbose)
        
        # Armazenar estatísticas para uso futuro
        self.stats = {}
    
    def fit_transform(self, df: pd.DataFrame, 
                     criar_target: bool = True,
                     col_precipitacao: Optional[str] = None,
                     criar_lags: bool = False,
                     criar_rolling: bool = False) -> pd.DataFrame:
        """
        Executa pipeline completo de pré-processamento
        
        Args:
            df: DataFrame bruto do INMET
            criar_target: Se True, cria variável binária 'Chuva'
            col_precipitacao: Nome da coluna de precipitação (auto-detecta se None)
            criar_lags: Se True, cria features de lag
            criar_rolling: Se True, cria rolling features
            
        Returns:
            DataFrame completamente pré-processado
        """
        if self.verbose:
            print("=" * 80)
            print("🚀 INICIANDO PIPELINE DE PRÉ-PROCESSAMENTO")
            print("=" * 80)
        
        # 1. Limpeza e conversão
        df_processed = self.cleaner.converter_para_numerico(df)
        df_processed = self.cleaner.remover_valores_invalidos(df_processed)
        df_processed = self.cleaner.tratar_valores_nulos(df_processed, metodo='interpolate')
        
        # 2. Transformação de datetime
        df_processed = self.datetime_transformer.criar_datetime(df_processed)
        
        # 3. Criar variável target (Chuva)
        if criar_target:
            if col_precipitacao is None:
                # Auto-detectar coluna de precipitação
                col_map = {col.upper(): col for col in df_processed.columns}
                precip_key = next((k for k in col_map if 'PRECIPITACAO' in k or 'PRECIPITAÇÃO' in k), None)
                if precip_key:
                    col_precipitacao = col_map[precip_key]
            
            if col_precipitacao and col_precipitacao in df_processed.columns:
                df_processed['Chuva'] = (df_processed[col_precipitacao] > 0).astype(int)
                if self.verbose:
                    taxa_chuva = df_processed['Chuva'].mean() * 100
                    print(f"\n🎯 Variável target 'Chuva' criada: {taxa_chuva:.2f}% de eventos")
        
        # 4. Feature Engineering
        df_processed = self.feature_engineer.criar_features_temporais(df_processed)
        df_processed = self.feature_engineer.criar_features_meteorologicas(df_processed)
        
        # 5. Lag features (opcional)
        if criar_lags:
            colunas_importantes = [col for col in df_processed.columns 
                                  if any(x in col.upper() for x in ['TEMPERATURA', 'UMIDADE', 'PRESSAO'])]
            if colunas_importantes:
                df_processed = self.feature_engineer.criar_lag_features(
                    df_processed, 
                    colunas_importantes[:3],  # Limitar a 3 para não explodir dimensionalidade
                    lags=[1, 3, 6]
                )
        
        # 6. Rolling features (opcional)
        if criar_rolling:
            colunas_importantes = [col for col in df_processed.columns 
                                  if any(x in col.upper() for x in ['TEMPERATURA', 'UMIDADE'])]
            if colunas_importantes:
                df_processed = self.feature_engineer.criar_rolling_features(
                    df_processed,
                    colunas_importantes[:2],  # Limitar a 2
                    windows=[3, 6, 12]
                )
        
        # 7. Remover linhas com muitos NaN gerados por lag/rolling
        if criar_lags or criar_rolling:
            linhas_antes = len(df_processed)
            df_processed = df_processed.dropna()
            linhas_depois = len(df_processed)
            if self.verbose and linhas_antes != linhas_depois:
                print(f"\n🧹 {linhas_antes - linhas_depois} linhas removidas devido a NaN de lag/rolling")
        
        # Armazenar estatísticas
        self.stats['n_registros'] = len(df_processed)
        self.stats['n_features'] = len(df_processed.columns)
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("✅ PRÉ-PROCESSAMENTO CONCLUÍDO")
            print("=" * 80)
            print(f"📊 Registros finais: {self.stats['n_registros']}")
            print(f"📊 Features totais: {self.stats['n_features']}")
            print("=" * 80)
        
        return df_processed
    
    def salvar_dados_processados(self, df: pd.DataFrame, 
                                caminho: str = '../data/dados_processados_ml.csv'):
        """Salva dados processados em CSV"""
        df.to_csv(caminho, index=False, encoding='utf-8')
        if self.verbose:
            print(f"\n💾 Dados processados salvos em: {caminho}")


# Funções auxiliares para uso direto
def preprocessar_dados_inmet(caminho_arquivo: str, 
                            salvar: bool = True,
                            criar_lags: bool = False,
                            criar_rolling: bool = False) -> pd.DataFrame:
    """
    Função de conveniência para pré-processar arquivo INMET completo
    
    Args:
        caminho_arquivo: Caminho para CSV do INMET
        salvar: Se True, salva resultado em CSV
        criar_lags: Se True, cria lag features
        criar_rolling: Se True, cria rolling features
        
    Returns:
        DataFrame pré-processado
    """
    # Importar função de carregamento
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from load_data import carregar_dados_inmet
    
    # Carregar dados
    df = carregar_dados_inmet(caminho_arquivo)
    
    # Pré-processar
    preprocessor = WeatherPreprocessor(verbose=True)
    df_processed = preprocessor.fit_transform(
        df, 
        criar_lags=criar_lags,
        criar_rolling=criar_rolling
    )
    
    # Salvar se solicitado
    if salvar:
        preprocessor.salvar_dados_processados(df_processed)
    
    return df_processed


if __name__ == '__main__':
    """
    Exemplo de uso direto do módulo
    """
    print("Módulo de pré-processamento carregado com sucesso!")
    print("\nExemplo de uso:")
    print("""
    from src.utils.preprocess import WeatherPreprocessor
    from src.utils.load_data import carregar_dados_inmet
    
    # Carregar dados
    df = carregar_dados_inmet('data/dados_inmet_2024.csv')
    
    # Criar preprocessor
    preprocessor = WeatherPreprocessor(verbose=True)
    
    # Processar dados (básico)
    df_processed = preprocessor.fit_transform(df)
    
    # Processar com lag e rolling features (avançado)
    df_processed = preprocessor.fit_transform(
        df, 
        criar_lags=True, 
        criar_rolling=True
    )
    
    # Salvar
    preprocessor.salvar_dados_processados(df_processed)
    """)
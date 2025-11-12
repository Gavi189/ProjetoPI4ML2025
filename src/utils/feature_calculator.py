"""
Módulo para cálculo automático de features derivadas para previsão
Autor: PI4-MachineLearning-2025
Descrição: Calcula features temporais e meteorológicas a partir de inputs básicos
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional


class FeatureCalculator:
    """Calcula features derivadas para entrada em modelos treinados"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def _log(self, message: str):
        if self.verbose:
            print(message)
    
    @staticmethod
    def calcular_features_temporais(hora: int, mes: int, dia_semana: int) -> Dict[str, float]:
        """
        Calcula todas as features temporais derivadas
        
        Args:
            hora: 0-23
            mes: 1-12
            dia_semana: 0-6 (0=segunda, 6=domingo)
            
        Returns:
            Dicionário com todas as features temporais
        """
        features = {
            'hora': float(hora),
            'mes': float(mes),
            'dia_semana': float(dia_semana),
            
            # Features cíclicas
            'hora_sin': float(np.sin(2 * np.pi * hora / 24)),
            'hora_cos': float(np.cos(2 * np.pi * hora / 24)),
            'mes_sin': float(np.sin(2 * np.pi * mes / 12)),
            'mes_cos': float(np.cos(2 * np.pi * mes / 12)),
            
            # Estação do ano (Hemisfério Sul)
            'estacao': FeatureCalculator._obter_estacao(mes),
            
            # Período do dia
            'periodo_dia': FeatureCalculator._obter_periodo(hora),
            
            # Fim de semana
            'fim_semana': 1.0 if dia_semana >= 5 else 0.0,
            
            # Dia e ano (valores default)
            'dia': 15.0,  # Dia do mês (default: meio do mês)
            'dia_ano': float(FeatureCalculator._calcular_dia_ano(mes, 15)),
            'semana_ano': float(FeatureCalculator._calcular_semana_ano(mes, 15)),
            'ano': float(datetime.now().year),
        }
        
        return features
    
    @staticmethod
    def _obter_estacao(mes: int) -> str:
        """Retorna estação (Hemisfério Sul)"""
        if mes in [12, 1, 2]:
            return 'verao'
        elif mes in [3, 4, 5]:
            return 'outono'
        elif mes in [6, 7, 8]:
            return 'inverno'
        else:
            return 'primavera'
    
    @staticmethod
    def _obter_periodo(hora: int) -> str:
        """Retorna período do dia"""
        if 6 <= hora < 12:
            return 'manha'
        elif 12 <= hora < 18:
            return 'tarde'
        elif 18 <= hora < 24:
            return 'noite'
        else:
            return 'madrugada'
    
    @staticmethod
    def _calcular_dia_ano(mes: int, dia: int) -> int:
        """Calcula dia do ano (1-365)"""
        dias_por_mes = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        return sum(dias_por_mes[:mes-1]) + dia
    
    @staticmethod
    def _calcular_semana_ano(mes: int, dia: int) -> int:
        """Calcula semana do ano (1-52)"""
        dia_ano = FeatureCalculator._calcular_dia_ano(mes, dia)
        return (dia_ano - 1) // 7 + 1
    
    @staticmethod
    def calcular_features_meteorologicas(
        temperatura: float,
        umidade: float,
        pressao: float,
        radiacao: float,
        vento: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calcula features meteorológicas derivadas
        
        Args:
            temperatura: °C
            umidade: % (0-100)
            pressao: mB
            radiacao: KJ/m²
            vento: m/s (opcional)
            
        Returns:
            Dicionário com features meteorológicas
        """
        features = {
            'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)': float(temperatura),
            'UMIDADE RELATIVA DO AR, HORARIA (%)': float(umidade),
            'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)': float(pressao),
            'RADIACAO GLOBAL (Kj/m²)': float(radiacao),
        }
        
        # Adicionar vento se fornecido
        if vento is not None:
            # Tentar adicionar com nomes comuns
            features['VENTO VELOCIDADE'] = float(vento)
            features['vento_velocidade'] = float(vento)
        
        # Calcular features derivadas
        # Índice de desconforto térmico simplificado
        T = temperatura
        UR = umidade
        indice_desconforto = T - 0.55 * (1 - 0.01 * UR) * (T - 14.5)
        features['indice_desconforto'] = float(indice_desconforto)
        
        # Radiação normalizada
        features['radiacao_norm'] = float(radiacao / 1000.0)  # KJ/m² para MJ/m²
        
        return features
    
    def criar_entrada_completa(
        self,
        temperatura: float,
        umidade: float,
        pressao: float,
        radiacao: float,
        hora: int,
        mes: int,
        dia_semana: int,
        vento: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Cria DataFrame completo com todas as features para predição
        
        Args:
            temperatura: °C
            umidade: % (0-100)
            pressao: mB
            radiacao: KJ/m²
            hora: 0-23
            mes: 1-12
            dia_semana: 0-6
            vento: m/s (opcional)
            
        Returns:
            DataFrame com todas as features calculadas
        """
        self._log("\n🔧 Calculando features derivadas...")
        
        # Calcular features
        features_temp = self.calcular_features_temporais(hora, mes, dia_semana)
        features_met = self.calcular_features_meteorologicas(
            temperatura, umidade, pressao, radiacao, vento
        )
        
        # Combinar
        dados_combinados = {**features_met, **features_temp}
        
        # Criar DataFrame
        df = pd.DataFrame([dados_combinados])
        
        self._log(f"✅ {len(df.columns)} features calculadas:")
        self._log(f"   • Temporais: hora, mes, dia_semana, estacao, periodo_dia, fim_semana, etc.")
        self._log(f"   • Meteorológicas: temperatura, umidade, pressao, radiacao, indice_desconforto, etc.")
        self._log(f"   • Cíclicas: hora_sin, hora_cos, mes_sin, mes_cos")
        
        return df
    
    def info_features_necessarias(self, metadados: Optional[Dict] = None) -> None:
        """
        Exibe informações sobre features necessárias para o modelo
        
        Args:
            metadados: Dicionário de metadados do modelo (se disponível)
        """
        if metadados:
            feature_names = metadados.get('feature_names', [])
            n_features = len(feature_names)
            
            self._log(f"\n📊 Informações do Modelo:")
            self._log(f"   • Versão: {metadados.get('version', 'N/A')}")
            self._log(f"   • Criado em: {metadados.get('created_at', 'N/A')}")
            self._log(f"   • Total de features esperadas: {n_features}")
            
            # Categorizar features
            temporais = [f for f in feature_names if any(x in f.lower() for x in ['hora', 'mes', 'dia', 'semana', 'ano', 'estacao', 'periodo'])]
            meteorologicas = [f for f in feature_names if any(x in f.lower() for x in ['temperatura', 'umidade', 'pressao', 'radiacao', 'vento', 'precipit', 'orvalho'])]
            derivadas = [f for f in feature_names if any(x in f.lower() for x in ['_sin', '_cos', '_lag', '_rolling', 'desconforto'])]
            
            if temporais:
                self._log(f"\n   🕐 Features Temporais ({len(temporais)}):")
                for f in sorted(temporais)[:10]:
                    self._log(f"      • {f}")
                if len(temporais) > 10:
                    self._log(f"      ... e mais {len(temporais) - 10}")
            
            if meteorologicas:
                self._log(f"\n   🌡️  Features Meteorológicas ({len(meteorologicas)}):")
                for f in sorted(meteorologicas)[:10]:
                    self._log(f"      • {f}")
                if len(meteorologicas) > 10:
                    self._log(f"      ... e mais {len(meteorologicas) - 10}")
            
            if derivadas:
                self._log(f"\n   ✨ Features Derivadas ({len(derivadas)}):")
                for f in sorted(derivadas)[:10]:
                    self._log(f"      • {f}")
                if len(derivadas) > 10:
                    self._log(f"      ... e mais {len(derivadas) - 10}")


def preparar_entrada_previsao(
    temperatura: float,
    umidade: float,
    pressao: float,
    radiacao: float,
    hora: int,
    mes: int,
    dia_semana: int,
    vento: Optional[float] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Função de conveniência para preparar entrada de previsão
    
    Args:
        temperatura: °C
        umidade: % (0-100)
        pressao: mB
        radiacao: KJ/m²
        hora: 0-23
        mes: 1-12
        dia_semana: 0-6
        vento: m/s (opcional)
        verbose: Se True, imprime informações
        
    Returns:
        DataFrame pronto para previsão
    """
    calculator = FeatureCalculator(verbose=verbose)
    return calculator.criar_entrada_completa(
        temperatura, umidade, pressao, radiacao,
        hora, mes, dia_semana, vento
    )


if __name__ == '__main__':
    """
    Exemplo de uso
    """
    print("="*80)
    print("MÓDULO DE CÁLCULO DE FEATURES - PI4 Machine Learning")
    print("="*80)
    
    # Exemplo
    calculator = FeatureCalculator(verbose=True)
    
    df_entrada = calculator.criar_entrada_completa(
        temperatura=25.5,
        umidade=80.0,
        pressao=1013.0,
        radiacao=2000.0,
        hora=14,
        mes=10,
        dia_semana=3,
        vento=3.5
    )
    
    print("\n📊 DataFrame Criado:")
    print(df_entrada)
    print(f"\nColunas: {list(df_entrada.columns)}")
    print(f"Valores: {df_entrada.to_dict('records')[0]}")

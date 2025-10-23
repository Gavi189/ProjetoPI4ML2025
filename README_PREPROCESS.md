# 📦 Módulo de Pré-processamento Automatizado

## 📋 Visão Geral

Módulo completo para pré-processamento de dados meteorológicos do INMET, incluindo:

- ✅ Limpeza e conversão de dados
- ✅ Transformação de datetime
- ✅ Feature engineering automatizado
- ✅ Pipeline reutilizável para treino e previsão

---

## 🏗️ Arquitetura do Módulo

```
src/utils/preprocess.py
│
├── DataCleaner              # Limpeza e conversão
│   ├── converter_para_numerico()
│   ├── remover_valores_invalidos()
│   └── tratar_valores_nulos()
│
├── DateTimeTransformer      # Transformação temporal
│   ├── identificar_colunas_temporais()
│   └── criar_datetime()
│
├── FeatureEngineer          # Criação de features
│   ├── criar_features_temporais()
│   ├── criar_features_meteorologicas()
│   ├── criar_lag_features()
│   └── criar_rolling_features()
│
└── WeatherPreprocessor      # Pipeline completo
    ├── fit_transform()
    └── salvar_dados_processados()
```

---

## 🚀 Uso Rápido

### Opção 1: Pipeline Completo (Recomendado)

```python
from src.utils.preprocess import WeatherPreprocessor
from src.utils.load_data import carregar_dados_inmet

# Carregar dados brutos
df = carregar_dados_inmet('data/dados_inmet_2024.csv')

# Criar preprocessor
preprocessor = WeatherPreprocessor(verbose=True)

# Processar dados
df_processed = preprocessor.fit_transform(df)

# Salvar
preprocessor.salvar_dados_processados(df_processed)
```

### Opção 2: Com Features Avançadas

```python
# Processar com lag e rolling features
df_processed = preprocessor.fit_transform(
    df,
    criar_lags=True,       # Features de 1h, 3h, 6h atrás
    criar_rolling=True     # Médias móveis de 3h, 6h, 12h
)
```

### Opção 3: Uso Modular

```python
from src.utils.preprocess import DataCleaner, DateTimeTransformer, FeatureEngineer

# Componentes separados
cleaner = DataCleaner()
dt_transformer = DateTimeTransformer()
feat_engineer = FeatureEngineer()

# Pipeline customizado
df = cleaner.converter_para_numerico(df)
df = cleaner.tratar_valores_nulos(df, metodo='interpolate')
df = dt_transformer.criar_datetime(df)
df = feat_engineer.criar_features_temporais(df)
```

---

## 📊 Features Criadas Automaticamente

### 🕐 Features Temporais (12 features)

| Feature       | Descrição                                          | Tipo  |
| ------------- | -------------------------------------------------- | ----- |
| `ano`         | Ano (2022, 2023, 2024)                             | int   |
| `mes`         | Mês (1-12)                                         | int   |
| `dia`         | Dia do mês (1-31)                                  | int   |
| `hora`        | Hora do dia (0-23)                                 | int   |
| `dia_semana`  | Dia da semana (0=segunda, 6=domingo)               | int   |
| `dia_ano`     | Dia do ano (1-365)                                 | int   |
| `semana_ano`  | Semana do ano (1-52)                               | int   |
| `hora_sin`    | Codificação cíclica da hora (seno)                 | float |
| `hora_cos`    | Codificação cíclica da hora (cosseno)              | float |
| `mes_sin`     | Codificação cíclica do mês (seno)                  | float |
| `mes_cos`     | Codificação cíclica do mês (cosseno)               | float |
| `estacao`     | Estação do ano (verao, outono, inverno, primavera) | str   |
| `periodo_dia` | Período (manha, tarde, noite, madrugada)           | str   |
| `fim_semana`  | 1 se fim de semana, 0 caso contrário               | int   |

### 🌡️ Features Meteorológicas (até 6 features)

| Feature               | Descrição                     | Fórmula                             |
| --------------------- | ----------------------------- | ----------------------------------- |
| `amplitude_termica`   | Variação de temperatura       | Temp_max - Temp_min                 |
| `spread_temp_orvalho` | Diferença temp-orvalho        | Temp - Orvalho                      |
| `variacao_pressao`    | Variação de pressão           | Pressao_max - Pressao_min           |
| `variacao_umidade`    | Variação de umidade           | Umidade_max - Umidade_min           |
| `indice_desconforto`  | Índice de desconforto térmico | DI = T - 0.55(1 - 0.01UR)(T - 14.5) |
| `radiacao_norm`       | Radiação normalizada          | Radiacao / 1000                     |

### ⏮️ Lag Features (opcional)

```python
# Exemplo: Temperatura 1h, 3h e 6h atrás
- TEMPERATURA_lag_1h
- TEMPERATURA_lag_3h
- TEMPERATURA_lag_6h
```

### 📊 Rolling Features (opcional)

```python
# Exemplo: Médias móveis de temperatura
- TEMPERATURA_rolling_mean_3h
- TEMPERATURA_rolling_std_3h
- TEMPERATURA_rolling_mean_6h
- TEMPERATURA_rolling_std_6h
```

---

## 🔧 Parâmetros de Configuração

### DataCleaner

```python
cleaner = DataCleaner(verbose=True)

# Converter para numérico
df = cleaner.converter_para_numerico(
    df,
    excluir_colunas=['Data', 'Hora']  # Colunas a não converter
)

# Tratar valores nulos
df = cleaner.tratar_valores_nulos(
    df,
    metodo='interpolate',      # 'interpolate', 'ffill', 'bfill', 'mean', 'median', 'drop'
    limite_interpolacao=3      # Máximo de NaNs consecutivos
)
```

### FeatureEngineer

```python
engineer = FeatureEngineer(verbose=True)

# Lag features
df = engineer.criar_lag_features(
    df,
    colunas=['TEMPERATURA', 'UMIDADE'],
    lags=[1, 2, 3, 6, 12, 24]  # Em horas
)

# Rolling features
df = engineer.criar_rolling_features(
    df,
    colunas=['TEMPERATURA', 'UMIDADE'],
    windows=[3, 6, 12, 24]  # Janelas em horas
)
```

---

## 📈 Exemplos de Saída

### Antes do Pipeline

```
Dimensões: 26304 × 19
Colunas numéricas: Strings com vírgulas
Valores nulos: ~15%
Features: Apenas variáveis originais
```

### Depois do Pipeline (Básico)

```
Dimensões: 26304 × 35
Colunas numéricas: Todas em float64
Valores nulos: 0%
Features: Originais + 16 temporais + 6 meteorológicas
Target: 'Chuva' (binário)
```

### Depois do Pipeline (Avançado)

```
Dimensões: 26280 × 65+
Features adicionais: Lags + Rolling
Linhas removidas: ~24 (devido a lag/rolling)
Pronto para modelos de séries temporais
```

---

## ⚙️ Tratamento de Dados Especiais

### 1. Valores Inválidos

- ✅ `inf` e `-inf` → `NaN`
- ✅ Outliers extremos (>10σ) → `NaN`
- ✅ Strings vazias → `NaN`

### 2. Valores Nulos

- ✅ Interpolação linear (padrão)
- ✅ Forward/Backward fill
- ✅ Média/Mediana
- ✅ Drop (remover linhas)

### 3. Conversão Numérica

- ✅ Vírgulas → Pontos decimais
- ✅ Espaços removidos
- ✅ Conversão forçada com `coerce`

### 4. Datetime

- ✅ Auto-detecção de colunas
- ✅ Múltiplos formatos suportados
- ✅ Ordenação temporal automática

---

## 🎯 Casos de Uso

### Caso 1: Preparação para Modelagem Simples

```python
# Para Random Forest, Logistic Regression, SVM
preprocessor = WeatherPreprocessor()
df_processed = preprocessor.fit_transform(
    df,
    criar_lags=False,
    criar_rolling=False
)
```

### Caso 2: Preparação para Séries Temporais

```python
# Para LSTM, ARIMA, Prophet
preprocessor = WeatherPreprocessor()
df_processed = preprocessor.fit_transform(
    df,
    criar_lags=True,
    criar_rolling=True
)
```

### Caso 3: Feature Engineering Customizado

```python
# Criar apenas features específicas
engineer = FeatureEngineer()
df = engineer.criar_features_temporais(df)
df = engineer.criar_lag_features(df, ['TEMPERATURA'], lags=[1, 2, 3])
```

---

## 📊 Métricas de Qualidade

O pipeline registra automaticamente:

```python
print(preprocessor.stats)
# {
#     'n_registros': 26304,
#     'n_features': 35,
#     'taxa_chuva': 0.0617,
#     'completude': 1.0
# }
```

---

## 🧪 Testes e Validação

### Teste Básico

```python
# Verificar se pipeline funciona
df_test = pd.DataFrame({
    'Data': ['2024/01/01', '2024/01/01'],
    'Hora': ['0100', '0200'],
    'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)': ['25,5', '26,0'],
    'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': ['0,0', '1,5']
})

preprocessor = WeatherPreprocessor(verbose=False)
df_result = preprocessor.fit_transform(df_test)

assert 'Chuva' in df_result.columns
assert 'datetime' in df_result.columns
assert 'hora' in df_result.columns
print("✅ Testes básicos passaram!")
```

---

## 🐛 Troubleshooting

### Erro: "Coluna não encontrada"

**Solução:** Verifique os nomes das colunas no seu CSV. O módulo usa auto-detecção, mas você pode especificar manualmente:

```python
# Ver colunas disponíveis
print(df.columns.tolist())

# Especificar coluna de precipitação
df_processed = preprocessor.fit_transform(
    df,
    col_precipitacao='PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'
)
```

### Erro: "Muitos NaNs após lag/rolling"

**Solução:** Reduza o número de lags ou tamanho das janelas:

```python
# Menos agressivo
engineer.criar_lag_features(df, colunas, lags=[1, 3])  # Ao invés de [1,3,6,12,24]
engineer.criar_rolling_features(df, colunas, windows=[3, 6])  # Ao invés de [3,6,12,24]
```

### Erro: "Conversão para numérico falhou"

**Solução:** Verifique formato dos dados. O módulo espera vírgula como separador decimal:

```python
# Se seus dados usam ponto:
df['coluna'] = df['coluna'].str.replace('.', ',')
```

### Erro: "Datetime inválido"

**Solução:** Especifique formato manualmente:

```python
dt_transformer = DateTimeTransformer()
df = dt_transformer.criar_datetime(
    df,
    col_data='Data',
    col_hora='Hora',
    formato_data='%d/%m/%Y',  # Ajustar formato
    formato_hora='%H:%M'       # Ajustar formato
)
```

---

## 🔄 Integração com Outros Módulos

### Com `load_data.py`

```python
from src.utils.load_data import carregar_dados_inmet
from src.utils.preprocess import WeatherPreprocessor

# Carregar e processar em sequência
df = carregar_dados_inmet('data/arquivo.csv')
preprocessor = WeatherPreprocessor()
df_processed = preprocessor.fit_transform(df)
```

### Com Pipeline de Modelagem

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Pré-processar
df_processed = preprocessor.fit_transform(df)

# Separar features e target
X = df_processed.drop(['Chuva', 'datetime'], axis=1)
y = df_processed['Chuva']

# Normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
```

---

## 📝 Boas Práticas

### ✅ DO's

1. **Sempre verifique a ordem temporal**

   ```python
   df = df.sort_values('datetime').reset_index(drop=True)
   ```

2. **Salve o preprocessor para uso em produção**

   ```python
   import pickle
   with open('preprocessor.pkl', 'wb') as f:
       pickle.dump(preprocessor, f)
   ```

3. **Use verbose=True durante desenvolvimento**

   ```python
   preprocessor = WeatherPreprocessor(verbose=True)
   ```

4. **Documente features customizadas**
   ```python
   # Criar arquivo de documentação
   with open('features_customizadas.txt', 'w') as f:
       f.write("Feature X: Calculada como Y + Z\n")
   ```

### ❌ DON'Ts

1. **Não misture dados de diferentes estações**

   ```python
   # Errado
   df = pd.concat([df_brasilia, df_saopaulo])

   # Correto: Processar separadamente
   df1 = preprocessor.fit_transform(df_brasilia)
   df2 = preprocessor.fit_transform(df_saopaulo)
   ```

2. **Não crie lag features antes de ordenar**

   ```python
   # Errado
   df = engineer.criar_lag_features(df, colunas, lags=[1])
   df = df.sort_values('datetime')

   # Correto
   df = df.sort_values('datetime')
   df = engineer.criar_lag_features(df, colunas, lags=[1])
   ```

3. **Não use lag/rolling em dados de teste sem cuidado**
   ```python
   # Risco de data leakage!
   # Use apenas dados de treino para criar lags no teste
   ```

---

## 🎓 Conceitos Importantes

### Codificação Cíclica

Hora e mês são **variáveis cíclicas** (24h volta para 0h, dezembro volta para janeiro). Usamos seno/cosseno para preservar essa relação:

```python
hora_sin = sin(2π × hora / 24)
hora_cos = cos(2π × hora / 24)
```

**Vantagem:** O modelo entende que 23h e 0h são próximas.

### Lag Features

Valores passados podem prever o futuro em séries temporais:

```python
# Temperatura de 1h atrás influencia temperatura atual
TEMPERATURA_lag_1h
```

**Cuidado:** Cria NaN nas primeiras linhas!

### Rolling Features

Capturam tendências de curto/médio prazo:

```python
# Média das últimas 3 horas
TEMPERATURA_rolling_mean_3h
```

**Uso:** Suaviza ruído e detecta padrões.

---

## 📚 Referências e Papers

### Índice de Desconforto Térmico

- Thom, E. C. (1959). "The Discomfort Index". Weatherwise, 12(2), 57-61.
- Fórmula: DI = T - 0.55(1 - 0.01×UR)(T - 14.5)

### Feature Engineering para ML Meteorológico

- Holmstrom, M., Liu, D., & Vo, C. (2016). "Machine learning applied to weather forecasting"
- Rasp, S., et al. (2018). "Neural networks for post-processing ensemble weather forecasts"

### Séries Temporais

- Brownlee, J. (2020). "Deep Learning for Time Series Forecasting"
- Hyndman, R. J., & Athanasopoulos, G. (2018). "Forecasting: Principles and Practice"

---

## 🤝 Contribuindo

Para adicionar novas features ao pipeline:

1. **Criar método na classe FeatureEngineer**

   ```python
   def criar_feature_customizada(self, df: pd.DataFrame) -> pd.DataFrame:
       df_feat = df.copy()
       # Sua lógica aqui
       return df_feat
   ```

2. **Adicionar ao pipeline**

   ```python
   # Em WeatherPreprocessor.fit_transform()
   df_processed = self.feature_engineer.criar_feature_customizada(df_processed)
   ```

3. **Documentar no README**
   - Adicionar à tabela de features
   - Incluir exemplo de uso

---

## 📊 Changelog

### v1.0.0 (2025-01-23)

- ✅ Pipeline completo implementado
- ✅ 12 features temporais
- ✅ 6 features meteorológicas
- ✅ Suporte a lag e rolling features
- ✅ Auto-detecção de colunas
- ✅ Documentação completa

### Próximas Features (v1.1.0)

- [ ] Suporte a múltiplas estações
- [ ] Feature selection automático
- [ ] Exportar pipeline para ONNX
- [ ] Validação de dados em tempo real
- [ ] Suporte a dados faltantes estruturados

---

## 📞 Suporte

Para dúvidas ou problemas:

1. Verifique a seção [Troubleshooting](#-troubleshooting)
2. Execute o notebook `exemplo_preprocess.ipynb`
3. Veja os logs com `verbose=True`
4. Abra uma issue no repositório

---

## 📄 Licença

Este módulo faz parte do projeto PI4-MachineLearning-2025.

---

## 🎉 Exemplos Completos

### Exemplo 1: Pipeline Mínimo

```python
from src.utils.preprocess import WeatherPreprocessor
from src.utils.load_data import carregar_dados_inmet

df = carregar_dados_inmet('data/dados_2024.csv')
preprocessor = WeatherPreprocessor(verbose=False)
df_processed = preprocessor.fit_transform(df)
print(f"✅ {len(df_processed)} registros processados")
```

### Exemplo 2: Pipeline Completo com Validação

```python
import pandas as pd
from src.utils.preprocess import WeatherPreprocessor

# Carregar dados
df = pd.read_csv('data/dados_raw.csv', sep=';', encoding='latin-1')

# Processar
preprocessor = WeatherPreprocessor(verbose=True)
df_processed = preprocessor.fit_transform(
    df,
    criar_lags=True,
    criar_rolling=True
)

# Validar
assert df_processed['Chuva'].isnull().sum() == 0, "Target tem NaN!"
assert 'datetime' in df_processed.columns, "Datetime não criado!"

# Salvar
preprocessor.salvar_dados_processados(df_processed)
print("✅ Pipeline executado com sucesso!")
```

### Exemplo 3: Processamento em Lote

```python
from pathlib import Path
from src.utils.preprocess import WeatherPreprocessor
from src.utils.load_data import carregar_dados_inmet

# Processar múltiplos arquivos
arquivos = list(Path('data').glob('*2022*.csv'))
preprocessor = WeatherPreprocessor(verbose=True)

dfs_processados = []
for arquivo in arquivos:
    print(f"\n📁 Processando: {arquivo.name}")
    df = carregar_dados_inmet(str(arquivo))
    df_proc = preprocessor.fit_transform(df)
    dfs_processados.append(df_proc)

# Concatenar
df_final = pd.concat(dfs_processados, ignore_index=True)
df_final = df_final.sort_values('datetime').reset_index(drop=True)

print(f"\n✅ Total: {len(df_final)} registros de {len(arquivos)} arquivos")
```

---

## 🔍 FAQ

**P: O pipeline funciona com dados de qualquer estação INMET?**  
R: Sim! O módulo usa auto-detecção de colunas e é agnóstico à estação.

**P: Posso usar o pipeline para previsão em tempo real?**  
R: Sim, mas você precisa manter o estado (últimos valores) para criar lag/rolling features.

**P: Como lidar com dados muito desbalanceados (6% de chuva)?**  
R: Use técnicas de balanceamento **após** o pré-processamento (SMOTE, class weights).

**P: O pipeline preserva a ordem temporal?**  
R: Sim! Dados são automaticamente ordenados por `datetime`.

**P: Posso adicionar minhas próprias features?**  
R: Sim! Use o `FeatureEngineer` de forma modular ou estenda a classe.

**P: Quanto tempo demora o processamento?**  
R: ~2-5 segundos para 26k registros (básico), ~10-15 segundos (com lag/rolling).

---

## 🎓 Tutorial Passo a Passo

### Passo 1: Instalação

```bash
# Não há instalação, apenas importe o módulo
cd PI4-MachineLearning-2025
```

### Passo 2: Primeiro Uso

```python
# exemplo_simples.py
from src.utils.preprocess import preprocessar_dados_inmet

df = preprocessar_dados_inmet('data/dados_2024.csv', salvar=True)
print(df.head())
```

### Passo 3: Explorar Features

```python
# Ver todas as features criadas
print("Features temporais:")
print([col for col in df.columns if any(x in col for x in ['ano', 'mes', 'hora'])])

print("\nFeatures meteorológicas:")
print([col for col in df.columns if any(x in col for x in ['amplitude', 'spread', 'variacao'])])
```

### Passo 4: Usar em Modelo

```python
from sklearn.ensemble import RandomForestClassifier

# Separar dados
X = df.drop(['Chuva', 'datetime', 'estacao', 'periodo_dia'], axis=1)
y = df['Chuva']

# Treinar
model = RandomForestClassifier()
model.fit(X, y)
```

---

**🎉 Pronto para usar! Execute o notebook `exemplo_preprocess.ipynb` para ver tudo em ação.**

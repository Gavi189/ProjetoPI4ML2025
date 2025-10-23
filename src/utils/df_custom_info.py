import pandas as pd

def df_custom_info(df: pd.DataFrame):
    """
    Exibe informações resumidas de um DataFrame:
    - Tipo da coluna
    - Valores nulos
    - % de nulos
    - Quantidade de valores únicos
    - Exemplos de valores
    """
    info = []
    for col in df.columns:
        info.append({
            "Coluna": col,
            "Tipo": df[col].dtype,
            "Nulos": df[col].isna().sum(),
            "% Nulos": round(df[col].isna().mean() * 100, 2),
            "Valores Únicos": df[col].nunique(dropna=True),
            "Exemplos": list(df[col].dropna().unique()[:3])
        })
    return pd.DataFrame(info)

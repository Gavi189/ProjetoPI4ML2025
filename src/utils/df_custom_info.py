import pandas as pd

def df_custom_info(df: pd.DataFrame):
    """
    Exibe informações personalizadas de um DataFrame.
    Mostra: tipo, nulos, % de nulos, valores únicos e exemplo de valores.
    """
    info = []
    for col in df.columns:
        dtype = df[col].dtype
        n_missing = df[col].isna().sum()
        perc_missing = (n_missing / len(df)) * 100
        n_unique = df[col].nunique(dropna=True)
        example_values = df[col].dropna().unique()[:3]  # 3 exemplos de valores

        info.append({
            "Coluna": col,
            "Tipo": dtype,
            "Nulos": n_missing,
            "% Nulos": round(perc_missing, 2),
            "Valores Únicos": n_unique,
            "Exemplos": example_values
        })

    return pd.DataFrame(info)
# Exemplo de uso:
# df = pd.read_csv("seu_arquivo.csv")   # Carregue seu DataFrame aqui
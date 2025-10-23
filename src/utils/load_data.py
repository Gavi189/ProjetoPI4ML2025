import pandas as pd

def carregar_dados_inmet(file_path: str) -> pd.DataFrame:
    """
    Carrega arquivos CSV do INMET, ignorando metadados até encontrar o cabeçalho.
    """
    try:
        with open(file_path, "r", encoding="latin-1") as f:
            linhas = f.readlines()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="iso-8859-1") as f:
            linhas = f.readlines()

    header_line = next((i for i, linha in enumerate(linhas) if "Data" in linha and "Hora" in linha), None)

    if header_line is None:
        raise ValueError("Cabeçalho com 'Data' e 'Hora' não encontrado.")

    return pd.read_csv(file_path, sep=";", encoding="latin-1", skiprows=header_line)

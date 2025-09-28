import pandas as pd

def carregar_dados_inmet(file_path: str) -> pd.DataFrame:
    """
    Carrega automaticamente arquivos do INMET (CSV),
    pulando linhas de metadados até encontrar o cabeçalho real.
    Retorna um DataFrame pronto para análise.
    """
    try:
        with open(file_path, "r", encoding="latin-1") as f:
            linhas = f.readlines()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="iso-8859-1") as f:
            linhas = f.readlines()
    
    header_line = None
    for i, linha in enumerate(linhas):
        if "Data" in linha and "Hora" in linha:
            header_line = i
            break

    if header_line is None:
        raise ValueError("Não foi possível encontrar o cabeçalho com 'Data' e 'Hora'.")
    
    df = pd.read_csv(
        file_path,
        sep=";",
        encoding="latin-1",
        skiprows=header_line
    )

    return df

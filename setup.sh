#!/bin/bash
# ============================================================================
# setup.sh - Script de Setup Automático (Linux/Mac)
# ============================================================================

echo "🌦️  Sistema de Previsão Meteorológica - Setup Automático"
echo "=========================================================="
echo ""

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Verificar Python
echo "📌 Verificando Python..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 não encontrado!${NC}"
    echo "Instale Python 3.8+ primeiro: https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo -e "${GREEN}✅ Python $PYTHON_VERSION encontrado${NC}"
echo ""

# Criar ambiente virtual
echo "📦 Criando ambiente virtual..."
python3 -m venv venv

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Ambiente virtual criado${NC}"
else
    echo -e "${RED}❌ Erro ao criar ambiente virtual${NC}"
    exit 1
fi
echo ""

# Ativar ambiente virtual
echo "🔧 Ativando ambiente virtual..."
source venv/bin/activate

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Ambiente virtual ativado${NC}"
else
    echo -e "${RED}❌ Erro ao ativar ambiente virtual${NC}"
    exit 1
fi
echo ""

# Atualizar pip
echo "⬆️  Atualizando pip..."
pip install --upgrade pip --quiet

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ pip atualizado${NC}"
fi
echo ""

# Instalar dependências
echo "📚 Instalando dependências..."
echo "Isso pode levar alguns minutos..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Dependências instaladas com sucesso${NC}"
else
    echo -e "${RED}❌ Erro ao instalar dependências${NC}"
    exit 1
fi
echo ""

# Criar estrutura de diretórios
echo "📁 Criando estrutura de diretórios..."
mkdir -p data/models
mkdir -p data/raw
mkdir -p data/processed
mkdir -p .streamlit
mkdir -p logs

echo -e "${GREEN}✅ Estrutura criada${NC}"
echo ""

# Criar arquivo de configuração do Streamlit
echo "⚙️  Criando configuração do Streamlit..."
cat > .streamlit/config.toml << EOF
[theme]
primaryColor = "#1E88E5"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
port = 8501
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 200

[browser]
gatherUsageStats = false
EOF

echo -e "${GREEN}✅ Configuração criada${NC}"
echo ""

# Verificar instalação
echo "🔍 Verificando instalação..."
python3 -c "import streamlit; import pandas; import sklearn; import plotly" 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Todas as bibliotecas instaladas corretamente${NC}"
else
    echo -e "${YELLOW}⚠️  Algumas bibliotecas podem estar faltando${NC}"
fi
echo ""

# Instruções finais
echo ""
echo "=========================================================="
echo -e "${GREEN}🎉 Setup Concluído com Sucesso!${NC}"
echo "=========================================================="
echo ""
echo "📋 Próximos passos:"
echo ""
echo "1. Ative o ambiente virtual:"
echo -e "   ${YELLOW}source venv/bin/activate${NC}"
echo ""
echo "2. Execute a aplicação:"
echo -e "   ${YELLOW}streamlit run app.py${NC}"
echo ""
echo "3. Acesse no navegador:"
echo -e "   ${YELLOW}http://localhost:8501${NC}"
echo ""
echo "=========================================================="
echo ""
echo "💡 Dicas:"
echo "   - Para desativar o ambiente: deactivate"
echo "   - Para reinstalar dependências: pip install -r requirements.txt"
echo "   - Para ver logs: tail -f logs/app.log"
echo ""
echo "📚 Documentação: README.md"
echo "🚀 Deploy: GUIA_DEPLOY.md"
echo ""
echo "Desenvolvido com ❤️  por PI4-MachineLearning-2025"
echo ""

# ============================================================================
# FIM setup.sh
# ============================================================================


# ============================================================================
# setup.bat - Script de Setup Automático (Windows)
# Salve como: setup.bat
# ============================================================================

@echo off
chcp 65001 >nul
echo.
echo 🌦️  Sistema de Previsão Meteorológica - Setup Automático
echo ==========================================================
echo.

REM Verificar Python
echo 📌 Verificando Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python não encontrado!
    echo Instale Python 3.8+ primeiro: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo ✅ Python %PYTHON_VERSION% encontrado
echo.

REM Criar ambiente virtual
echo 📦 Criando ambiente virtual...
python -m venv venv

if errorlevel 1 (
    echo ❌ Erro ao criar ambiente virtual
    pause
    exit /b 1
)

echo ✅ Ambiente virtual criado
echo.

REM Ativar ambiente virtual
echo 🔧 Ativando ambiente virtual...
call venv\Scripts\activate.bat

if errorlevel 1 (
    echo ❌ Erro ao ativar ambiente virtual
    pause
    exit /b 1
)

echo ✅ Ambiente virtual ativado
echo.

REM Atualizar pip
echo ⬆️  Atualizando pip...
python -m pip install --upgrade pip --quiet

if errorlevel 0 (
    echo ✅ pip atualizado
)
echo.

REM Instalar dependências
echo 📚 Instalando dependências...
echo Isso pode levar alguns minutos...
pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ Erro ao instalar dependências
    pause
    exit /b 1
)

echo ✅ Dependências instaladas com sucesso
echo.

REM Criar estrutura de diretórios
echo 📁 Criando estrutura de diretórios...
mkdir data\models 2>nul
mkdir data\raw 2>nul
mkdir data\processed 2>nul
mkdir .streamlit 2>nul
mkdir logs 2>nul

echo ✅ Estrutura criada
echo.

REM Criar arquivo de configuração do Streamlit
echo ⚙️  Criando configuração do Streamlit...
(
echo [theme]
echo primaryColor = "#1E88E5"
echo backgroundColor = "#FFFFFF"
echo secondaryBackgroundColor = "#F0F2F6"
echo textColor = "#262730"
echo font = "sans serif"
echo.
echo [server]
echo port = 8501
echo enableCORS = false
echo enableXsrfProtection = true
echo maxUploadSize = 200
echo.
echo [browser]
echo gatherUsageStats = false
) > .streamlit\config.toml

echo ✅ Configuração criada
echo.

REM Verificar instalação
echo 🔍 Verificando instalação...
python -c "import streamlit; import pandas; import sklearn; import plotly" 2>nul

if errorlevel 0 (
    echo ✅ Todas as bibliotecas instaladas corretamente
) else (
    echo ⚠️  Algumas bibliotecas podem estar faltando
)
echo.

REM Instruções finais
echo.
echo ==========================================================
echo 🎉 Setup Concluído com Sucesso!
echo ==========================================================
echo.
echo 📋 Próximos passos:
echo.
echo 1. Ative o ambiente virtual (se ainda não ativado):
echo    venv\Scripts\activate
echo.
echo 2. Execute a aplicação:
echo    streamlit run app.py
echo.
echo 3. Acesse no navegador:
echo    http://localhost:8501
echo.
echo ==========================================================
echo.
echo 💡 Dicas:
echo    - Para desativar o ambiente: deactivate
echo    - Para reinstalar dependências: pip install -r requirements.txt
echo    - Para limpar cache: streamlit cache clear
echo.
echo 📚 Documentação: README.md
echo 🚀 Deploy: GUIA_DEPLOY.md
echo.
echo Desenvolvido com ❤️  por PI4-MachineLearning-2025
echo.
pause

REM ============================================================================
REM FIM setup.bat
REM ============================================================================
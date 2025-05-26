#!/bin/bash
# Script para depurar el bot SOL

# Colores para mejor legibilidad
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Depurando el bot SOL ===${NC}"

# Directorio base
BASE_DIR="/home/edisonbautistaruiz2025"
BOT_DIR="${BASE_DIR}/new-trading-bots/src/spot_bots/sol_bot_20m"
LOG_DIR="${BOT_DIR}/logs"
DEBUG_LOG="${LOG_DIR}/debug.log"

# Crear directorio de logs si no existe
echo -e "${YELLOW}Creando directorio de logs...${NC}"
mkdir -p "$LOG_DIR"

# Verificar Python
echo -e "${YELLOW}Verificando Python...${NC}"
echo "Python versión:" > "$DEBUG_LOG"
python3 --version >> "$DEBUG_LOG" 2>&1
echo "" >> "$DEBUG_LOG"

# Verificar dependencias
echo -e "${YELLOW}Verificando dependencias...${NC}"
echo "Dependencias instaladas:" >> "$DEBUG_LOG"
pip3 list | grep -E "scikit-learn|pandas|numpy|python-binance|python-telegram-bot|python-dotenv" >> "$DEBUG_LOG" 2>&1
echo "" >> "$DEBUG_LOG"

# Verificar archivo .env
echo -e "${YELLOW}Verificando archivo .env...${NC}"
echo "Archivo .env:" >> "$DEBUG_LOG"
if [ -f "${BOT_DIR}/.env" ]; then
    echo "El archivo .env existe" >> "$DEBUG_LOG"
    # Verificar variables sin mostrar valores
    echo "Variables en .env:" >> "$DEBUG_LOG"
    grep -o "^[^=]*=" "${BOT_DIR}/.env" >> "$DEBUG_LOG" 2>&1
else
    echo "El archivo .env NO existe" >> "$DEBUG_LOG"
fi
echo "" >> "$DEBUG_LOG"

# Intentar ejecutar el bot
echo -e "${YELLOW}Intentando ejecutar el bot...${NC}"
echo "Intentando ejecutar el bot:" >> "$DEBUG_LOG"
cd "$BOT_DIR"
python3 -c "import sys; print(sys.path)" >> "$DEBUG_LOG" 2>&1
echo "" >> "$DEBUG_LOG"
echo "Importando módulos:" >> "$DEBUG_LOG"
python3 -c "
try:
    import pandas
    print('pandas: OK')
except ImportError as e:
    print(f'pandas: ERROR - {e}')

try:
    import numpy
    print('numpy: OK')
except ImportError as e:
    print(f'numpy: ERROR - {e}')

try:
    import sklearn
    print('scikit-learn: OK')
except ImportError as e:
    print(f'scikit-learn: ERROR - {e}')

try:
    import binance
    print('python-binance: OK')
except ImportError as e:
    print(f'python-binance: ERROR - {e}')

try:
    import telegram
    print('python-telegram-bot: OK')
except ImportError as e:
    print(f'python-telegram-bot: ERROR - {e}')

try:
    import dotenv
    print('python-dotenv: OK')
except ImportError as e:
    print(f'python-dotenv: ERROR - {e}')
" >> "$DEBUG_LOG" 2>&1
echo "" >> "$DEBUG_LOG"

# Intentar ejecutar el script principal
echo "Ejecutando script principal:" >> "$DEBUG_LOG"
python3 main.py --help >> "$DEBUG_LOG" 2>&1 || echo "Error al ejecutar el script principal" >> "$DEBUG_LOG"

echo -e "${GREEN}Depuración completada. Revisa el archivo de log:${NC}"
echo -e "${YELLOW}$DEBUG_LOG${NC}"
echo -e "${YELLOW}Puedes ver el contenido con: cat $DEBUG_LOG${NC}"

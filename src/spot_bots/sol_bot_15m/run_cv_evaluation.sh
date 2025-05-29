#!/bin/bash
# Script para ejecutar la validación cruzada utilizando las mismas variables de entorno que el bot principal

# Obtener el directorio del script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Verificar si start_cloud_simulation.sh existe
if [ ! -f "start_cloud_simulation.sh" ]; then
    echo "Error: No se encontró el script start_cloud_simulation.sh"
    exit 1
fi

# Extraer la ubicación del archivo de credenciales del script start_cloud_simulation.sh
CREDENTIALS_FILE=$(grep -o 'CREDENTIALS_FILE=.*' start_cloud_simulation.sh | head -n 1 | cut -d= -f2 | tr -d '"')

if [ -z "$CREDENTIALS_FILE" ]; then
    # Si no se encuentra, buscar la variable CREDENTIALS_PATH
    CREDENTIALS_FILE=$(grep -o 'CREDENTIALS_PATH=.*' start_cloud_simulation.sh | head -n 1 | cut -d= -f2 | tr -d '"')
fi

if [ -z "$CREDENTIALS_FILE" ]; then
    echo "No se pudo determinar la ubicación del archivo de credenciales"
    exit 1
fi

echo "Archivo de credenciales: $CREDENTIALS_FILE"

# Verificar si el archivo existe
if [ ! -f "$CREDENTIALS_FILE" ]; then
    echo "Error: No se encontró el archivo de credenciales en $CREDENTIALS_FILE"
    exit 1
fi

# Extraer las credenciales del archivo
echo "Extrayendo credenciales del archivo..."
PYTHON_EXTRACT=$(python3 -c "
import json
try:
    with open('$CREDENTIALS_FILE', 'r') as f:
        data = json.load(f)
    
    # Verificar si las credenciales están en la raíz o en 'env'
    env = data.get('env', data)
    
    print(env.get('BINANCE_API_KEY', ''))
    print(env.get('BINANCE_API_SECRET', ''))
    print(env.get('TELEGRAM_BOT_TOKEN', ''))
    print(env.get('TELEGRAM_CHAT_ID', ''))
except Exception as e:
    print('')
    print('')
    print('')
    print('')
    print(f'Error: {str(e)}', file=sys.stderr)
")

# Extraer las credenciales del resultado de Python
BINANCE_API_KEY=$(echo "$PYTHON_EXTRACT" | sed -n '1p')
BINANCE_API_SECRET=$(echo "$PYTHON_EXTRACT" | sed -n '2p')
TELEGRAM_BOT_TOKEN=$(echo "$PYTHON_EXTRACT" | sed -n '3p')
TELEGRAM_CHAT_ID=$(echo "$PYTHON_EXTRACT" | sed -n '4p')

# Si no se pudieron extraer con Python, intentar con grep
if [ -z "$BINANCE_API_KEY" ] || [ -z "$BINANCE_API_SECRET" ]; then
    echo "Intentando extraer credenciales con grep..."
    BINANCE_API_KEY=$(grep -o '"BINANCE_API_KEY"[^,]*' "$CREDENTIALS_FILE" | grep -o '"[^"]*"' | tail -n 1 | tr -d '"')
    BINANCE_API_SECRET=$(grep -o '"BINANCE_API_SECRET"[^,]*' "$CREDENTIALS_FILE" | grep -o '"[^"]*"' | tail -n 1 | tr -d '"')
fi

if [ -z "$TELEGRAM_BOT_TOKEN" ] || [ -z "$TELEGRAM_CHAT_ID" ]; then
    TELEGRAM_BOT_TOKEN=$(grep -o '"TELEGRAM_BOT_TOKEN"[^,]*' "$CREDENTIALS_FILE" | grep -o '"[^"]*"' | tail -n 1 | tr -d '"')
    TELEGRAM_CHAT_ID=$(grep -o '"TELEGRAM_CHAT_ID"[^,]*' "$CREDENTIALS_FILE" | grep -o '"[^"]*"' | tail -n 1 | tr -d '"')
fi

# Exportar las credenciales como variables de entorno
if [ -n "$BINANCE_API_KEY" ] && [ -n "$BINANCE_API_SECRET" ]; then
    export BINANCE_API_KEY="$BINANCE_API_KEY"
    export BINANCE_API_SECRET="$BINANCE_API_SECRET"
    echo "BINANCE_API_KEY configurada: ${BINANCE_API_KEY:0:3}...${BINANCE_API_KEY: -3}"
    echo "BINANCE_API_SECRET configurada: ${BINANCE_API_SECRET:0:3}...${BINANCE_API_SECRET: -3}"
else
    echo "Error: No se pudieron extraer las credenciales de Binance"
    exit 1
fi

if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
    export TELEGRAM_BOT_TOKEN="$TELEGRAM_BOT_TOKEN"
    export TELEGRAM_CHAT_ID="$TELEGRAM_CHAT_ID"
    echo "TELEGRAM_BOT_TOKEN configurado: ${TELEGRAM_BOT_TOKEN:0:3}...${TELEGRAM_BOT_TOKEN: -3}"
    echo "TELEGRAM_CHAT_ID configurado: ${TELEGRAM_CHAT_ID}"
else
    echo "Advertencia: No se pudieron extraer las credenciales de Telegram"
fi

# Configurar otras variables de entorno necesarias
export CLOUD_ENV=true
export MEMORY_LIMIT_MB=2048
export TF_DETERMINISTIC=true
export USE_MULTIPROCESSING=false
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_ENABLE_ONEDNN_OPTS=1
export TF_NUM_INTEROP_THREADS=2
export TF_NUM_INTRAOP_THREADS=2
export TF_ENABLE_AUTO_MIXED_PRECISION=1

# Ejecutar el script de validación cruzada
echo "Ejecutando validación cruzada..."
python3 evaluate_model_cv.py --symbol SOLUSDT --interval 15m --cv-method timeseries --folds 5 --notify

# Verificar el resultado
if [ $? -eq 0 ]; then
    echo "Validación cruzada completada exitosamente"
else
    echo "Error al ejecutar la validación cruzada"
    exit 1
fi

echo "Proceso completado"

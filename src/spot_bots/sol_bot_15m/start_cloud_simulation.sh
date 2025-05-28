#!/bin/bash
# Script optimizado para iniciar el bot SOL en modo simulación de aprendizaje en Google Cloud VM
# Este script está específicamente diseñado para funcionar en el entorno de Google Cloud

# Configuración de variables
BOT_NAME="sol_bot_15m"
SYMBOL="SOLUSDT"
INTERVAL="15m"
SIMULATION_BALANCE=100   # Balance ficticio en USDT
RETRAIN_INTERVAL=15      # Reentrenamiento cada 15 minutos
STATUS_INTERVAL=1        # Actualización de estado cada 1 hora
RISK=0.02                # 2% de riesgo por operación

# Directorio del bot (ruta absoluta en Google Cloud VM)
BOT_DIR="/home/edisonbautistaruiz2025/new-trading-bots/src/spot_bots/sol_bot_15m"

# Crear directorio de logs si no existe
mkdir -p "$BOT_DIR/logs"
LOG_FILE="$BOT_DIR/logs/${BOT_NAME}_cloud_simulation_$(date +%Y%m%d_%H%M%S).log"

# Establecer variables de entorno optimizadas para Google Cloud VM
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

# Verificar que las credenciales estén disponibles
CREDENTIALS_FILE="$HOME/trading-bots-api/credentials.json"
if [ ! -f "$CREDENTIALS_FILE" ]; then
    echo "ERROR: No se encontró el archivo de credenciales"
    echo "Asegúrate de que exista: $CREDENTIALS_FILE"
    echo "Puedes copiarlo desde la API con: cp ~/trading-bots-api/credentials.json $CREDENTIALS_FILE"
    exit 1
fi

# Exportar la ruta de credenciales para que el bot la use
export CREDENTIALS_PATH="$CREDENTIALS_FILE"

# Extraer y exportar las credenciales de Binance y Telegram directamente
if [ -f "$CREDENTIALS_FILE" ]; then
    echo "Extrayendo credenciales desde $CREDENTIALS_FILE"
    
    # Mostrar las primeras líneas del archivo para depuración (sin mostrar valores sensibles)
    echo "Estructura del archivo credentials.json:"
    head -n 20 "$CREDENTIALS_FILE" | grep -v "KEY\|SECRET\|PASSWORD\|TOKEN" | sed 's/".*":.*"/"VALOR_SENSIBLE"/g'
    
    # Verificar si el archivo tiene una estructura JSON válida
    if ! cat "$CREDENTIALS_FILE" | python3 -m json.tool &>/dev/null; then
        echo "ADVERTENCIA: El archivo credentials.json no tiene un formato JSON válido"
    fi
    
    # Intentar diferentes métodos para extraer las credenciales
    echo "Intentando diferentes métodos para extraer credenciales..."
    
    # Método 1: Usar Python para extraer todas las credenciales necesarias
    echo "Método 1: Usando Python"
    PYTHON_EXTRACT=$(python3 -c "
import json
try:
    with open('$CREDENTIALS_FILE', 'r') as f:
        data = json.load(f)
    env = data.get('env', {})
    print(env.get('BINANCE_API_KEY', ''))
    print(env.get('BINANCE_API_SECRET', ''))
    print(env.get('TELEGRAM_BOT_TOKEN', ''))
    print(env.get('TELEGRAM_CHAT_ID', ''))
except Exception as e:
    print('')
    print('')
    print('')
    print('')
    print(f'Error: {str(e)}')
" 2>/dev/null)

    # Extraer cada credencial de la salida de Python
    BINANCE_API_KEY=$(echo "$PYTHON_EXTRACT" | sed -n '1p')
    BINANCE_API_SECRET=$(echo "$PYTHON_EXTRACT" | sed -n '2p')
    TELEGRAM_BOT_TOKEN=$(echo "$PYTHON_EXTRACT" | sed -n '3p')
    TELEGRAM_CHAT_ID=$(echo "$PYTHON_EXTRACT" | sed -n '4p')
    
    # Si el método 1 falla, intentar método 2
    if [ -z "$BINANCE_API_KEY" ] || [ -z "$BINANCE_API_SECRET" ]; then
        echo "Método 2: Usando grep para credenciales de Binance"
        BINANCE_API_KEY=$(grep -o '"BINANCE_API_KEY"[^,]*' "$CREDENTIALS_FILE" | grep -o '"[^"]*"' | tail -n 1 | tr -d '"')
        BINANCE_API_SECRET=$(grep -o '"BINANCE_API_SECRET"[^,]*' "$CREDENTIALS_FILE" | grep -o '"[^"]*"' | tail -n 1 | tr -d '"')
    fi
    
    if [ -z "$TELEGRAM_BOT_TOKEN" ] || [ -z "$TELEGRAM_CHAT_ID" ]; then
        echo "Método 2: Usando grep para credenciales de Telegram"
        TELEGRAM_BOT_TOKEN=$(grep -o '"TELEGRAM_BOT_TOKEN"[^,]*' "$CREDENTIALS_FILE" | grep -o '"[^"]*"' | tail -n 1 | tr -d '"')
        TELEGRAM_CHAT_ID=$(grep -o '"TELEGRAM_CHAT_ID"[^,]*' "$CREDENTIALS_FILE" | grep -o '"[^"]*"' | tail -n 1 | tr -d '"')
    fi
    
    # Verificar y exportar las credenciales de Binance
    if [ -n "$BINANCE_API_KEY" ] && [ -n "$BINANCE_API_SECRET" ]; then
        export BINANCE_API_KEY="$BINANCE_API_KEY"
        export BINANCE_API_SECRET="$BINANCE_API_SECRET"
        echo "Credenciales de Binance configuradas correctamente"
        echo "BINANCE_API_KEY configurada: ${BINANCE_API_KEY:0:3}...${BINANCE_API_KEY: -3}"
        echo "BINANCE_API_SECRET configurada: ${BINANCE_API_SECRET:0:3}...${BINANCE_API_SECRET: -3}"
    else
        echo "ERROR: No se pudieron extraer las credenciales de Binance"
        echo "Por favor, verifica que el archivo credentials.json contiene las claves BINANCE_API_KEY y BINANCE_API_SECRET"
    fi
    
    # Verificar y exportar las credenciales de Telegram
    if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
        export TELEGRAM_BOT_TOKEN="$TELEGRAM_BOT_TOKEN"
        export TELEGRAM_CHAT_ID="$TELEGRAM_CHAT_ID"
        echo "Credenciales de Telegram configuradas correctamente"
        echo "TELEGRAM_BOT_TOKEN configurado: ${TELEGRAM_BOT_TOKEN:0:3}...${TELEGRAM_BOT_TOKEN: -3}"
        echo "TELEGRAM_CHAT_ID configurado: ${TELEGRAM_CHAT_ID}"
    else
        echo "ADVERTENCIA: No se pudieron extraer las credenciales de Telegram"
        echo "Las notificaciones de Telegram no funcionarán"
        echo "Por favor, verifica que el archivo credentials.json contiene las claves TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID"
    fi
else
    echo "ERROR: No se encontró el archivo credentials.json en $CREDENTIALS_FILE"
fi

echo "==================================================================="
echo "  INICIANDO BOT SOL EN MODO SIMULACIÓN DE APRENDIZAJE"
echo "  ENTORNO: GOOGLE CLOUD VM"
echo "==================================================================="
echo "  • Símbolo: $SYMBOL"
echo "  • Intervalo: $INTERVAL"
echo "  • Balance de simulación: $SIMULATION_BALANCE USDT"
echo "  • Reentrenamiento ML: Cada $RETRAIN_INTERVAL minutos"
echo "  • Riesgo por operación: ${RISK}%"
echo "  • Usando credenciales reales de Binance para datos de mercado"
echo "  • Operando con USDT ficticio"
echo "==================================================================="
echo "  Logs: $LOG_FILE"
echo "==================================================================="

# Verificar instalación de dependencias críticas
python3 -c "import tensorflow as tf; print('TensorFlow instalado: versión', tf.__version__)" || {
    echo "ERROR: TensorFlow no está instalado correctamente"
    echo "Instala TensorFlow con: pip install tensorflow"
    exit 1
}

# Iniciar el bot con los parámetros configurados
cd "$BOT_DIR"

# Ejecutar verificación de compatibilidad con Google Cloud VM
echo "Verificando compatibilidad con Google Cloud VM..."
# Comentado porque el script no existe en esta ubicación
# python3 src/utils/check_cloud_compatibility.py --fix

# Limpiar archivos redundantes o temporales
echo "Limpiando archivos redundantes..."
# Comentado porque el script no existe en esta ubicación
# python3 src/utils/cleanup_redundant_files.py --clean

# Iniciar el bot en una sesión de screen
echo "Iniciando bot en sesión screen '$BOT_NAME'..."

# Configurar PYTHONPATH para incluir los directorios necesarios
PYTHONPATH="$BOT_DIR/src:$BOT_DIR:$HOME/new-trading-bots"

# Iniciar el bot en una sesión screen con el PYTHONPATH configurado
screen -dmS $BOT_NAME bash -c "cd $BOT_DIR && PYTHONPATH=$PYTHONPATH python3 main.py \
    --use-ml \
    --retrain-interval $RETRAIN_INTERVAL \
    --interval $INTERVAL \
    --symbol $SYMBOL \
    --simulation \
    --balance $SIMULATION_BALANCE \
    --risk $RISK \
    --status-interval $STATUS_INTERVAL \
    > \"$LOG_FILE\" 2>&1"

echo "Bot iniciado en sesión screen '$BOT_NAME'"
echo "Para ver los logs en tiempo real: screen -r $BOT_NAME"
echo "Para desconectarse de la sesión sin detener el bot: Ctrl+A, D"
echo "Para ver los logs guardados: tail -f $LOG_FILE"
echo ""
echo "Bot iniciado en modo simulación de aprendizaje en Google Cloud VM."
echo "Usando datos de mercado reales con balance ficticio de $SIMULATION_BALANCE USDT."

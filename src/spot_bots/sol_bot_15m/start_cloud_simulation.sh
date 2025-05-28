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

# Exportar la ruta de credenciales como variable de entorno para que el bot la use
export CREDENTIALS_PATH="$CREDENTIALS_FILE"

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
screen -dmS $BOT_NAME bash -c "python3 main.py \
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

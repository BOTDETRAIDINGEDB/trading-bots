#!/bin/bash
# Script para iniciar el bot SOL en modo simulación con credenciales reales de Binance
# pero usando USDT ficticio para operar con datos de mercado reales.
# Este modo permite que el bot aprenda y se optimice sin arriesgar fondos reales.

# Configuración de variables
BOT_NAME="sol_bot_15m"
SYMBOL="SOLUSDT"
INTERVAL="15m"
SIMULATION_BALANCE=100   # Balance ficticio en USDT
RETRAIN_INTERVAL=15      # Reentrenamiento cada 15 minutos
STATUS_INTERVAL=1        # Actualización de estado cada 1 hora
RISK=0.02                # 2% de riesgo por operación

# Directorio del bot (ajustar según la instalación)
BOT_DIR="$(pwd)"

# Crear directorio de logs si no existe
mkdir -p "$BOT_DIR/logs"
LOG_FILE="$BOT_DIR/logs/${BOT_NAME}_simulation_$(date +%Y%m%d_%H%M%S).log"

# Establecer variables de entorno para Google Cloud VM
export CLOUD_ENV=true
export MEMORY_LIMIT_MB=2048
export TF_DETERMINISTIC=true
export USE_MULTIPROCESSING=false
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo "==================================================================="
echo "  INICIANDO BOT SOL EN MODO SIMULACIÓN DE APRENDIZAJE"
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

# === NUEVA LÓGICA DE DETECCIÓN DE ENTORNO CLOUD ===
# Si detectamos que estamos en Google Cloud (por variable de entorno, hostname típico o archivo de marca),
# delegar el inicio a start_cloud_simulation.sh y salir.

IS_CLOUD=false
# Detección básica por variable de entorno o hostname de Google Cloud
if [[ "$CLOUD_ENV" == "true" ]] || grep -qi 'google' /proc/cpuinfo 2>/dev/null || hostname | grep -qi 'gce'; then
    IS_CLOUD=true
fi

if [ "$IS_CLOUD" = true ]; then
    echo "[INFO] Entorno Google Cloud detectado. Delegando a start_cloud_simulation.sh..."
    exec "$BOT_DIR/start_cloud_simulation.sh" "$@"
    exit 0
fi

# Iniciar el bot con los parámetros configurados (entorno local)
cd "$BOT_DIR"

# Verificar si estamos en Windows o Linux/Mac
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows (Git Bash o similar)
    echo "Detectado sistema Windows, iniciando en modo consola..."
    python main.py \
        --use-ml \
        --retrain-interval $RETRAIN_INTERVAL \
        --interval $INTERVAL \
        --symbol $SYMBOL \
        --simulation \
        --balance $SIMULATION_BALANCE \
        --risk $RISK \
        --status-interval $STATUS_INTERVAL \
        > "$LOG_FILE" 2>&1
else
    # Linux/Mac
    echo "Detectado sistema Linux/Mac, iniciando en sesión screen..."
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
fi

echo "Bot iniciado en modo simulación de aprendizaje."
echo "Usando datos de mercado reales con balance ficticio de $SIMULATION_BALANCE USDT."

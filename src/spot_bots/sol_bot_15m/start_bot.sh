#!/bin/bash
# Script principal para iniciar el bot SOL en modo REAL con todas las funcionalidades avanzadas
# Este script está optimizado para operaciones con fondos reales

# Colores para mejor legibilidad
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detectar el sistema operativo y configurar el directorio del bot
if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # macOS o Linux
    BOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    # Windows
    BOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# === NUEVA LÓGICA DE DETECCIÓN DE ENTORNO CLOUD ===
# Si detectamos que estamos en Google Cloud (por variable de entorno, hostname típico o archivo de marca),
# delegar el inicio a start_cloud_simulation.sh (o start_cloud_real.sh en el futuro) y salir.
IS_CLOUD=false
if [[ "$CLOUD_ENV" == "true" ]] || grep -qi 'google' /proc/cpuinfo 2>/dev/null || hostname | grep -qi 'gce'; then
    IS_CLOUD=true
fi

if [ "$IS_CLOUD" = true ]; then
    echo "[INFO] Entorno Google Cloud detectado. Delegando a start_cloud_simulation.sh..."
    exec "$BOT_DIR/start_cloud_simulation.sh" "$@"
    exit 0
fi

cd "$BOT_DIR"

# Configuración de parámetros
SYMBOL="SOLUSDT"
INTERVAL="15m"
RETRAIN_INTERVAL=60  # Reentrenamiento cada 60 minutos en modo real
STATUS_INTERVAL=1    # Actualización de estado cada 1 hora
RISK=0.02            # 2% de riesgo por operación

# Crear directorio de logs si no existe
mkdir -p "$BOT_DIR/logs"
LOG_FILE="$BOT_DIR/logs/sol_bot_real_$(date +%Y%m%d_%H%M%S).log"

# Verificar si hay una instancia del bot ya en ejecución
if pgrep -f "python3 main.py.*$SYMBOL" > /dev/null; then
    echo -e "${YELLOW}ADVERTENCIA: Ya hay una instancia del bot ejecutándose.${NC}"
    echo -e "${YELLOW}Si deseas iniciar una nueva instancia, primero detén la actual con ./stop.sh${NC}"
    exit 1
fi

# Verificar conexión con Telegram antes de iniciar
echo -e "${BLUE}Verificando conexión con Telegram...${NC}"
python3 -c "from src.utils.enhanced_telegram_notifier import EnhancedTelegramNotifier; notifier = EnhancedTelegramNotifier(); print('Conexión con Telegram:', 'OK' if notifier.verify_connection() else 'FALLIDA')"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error al verificar la conexión con Telegram. Verifica tus credenciales.${NC}"
    exit 1
fi

echo -e "${GREEN}==================================================================${NC}"
echo -e "${GREEN}  INICIANDO BOT SOL EN MODO REAL${NC}"
echo -e "${GREEN}==================================================================${NC}"
echo -e "${GREEN}  • Símbolo: $SYMBOL${NC}"
echo -e "${GREEN}  • Intervalo: $INTERVAL${NC}"
echo -e "${GREEN}  • Reentrenamiento ML: Cada $RETRAIN_INTERVAL minutos${NC}"
echo -e "${GREEN}  • Riesgo por operación: ${RISK}%${NC}"
echo -e "${GREEN}  • Usando credenciales reales de Binance${NC}"
echo -e "${GREEN}  • Operando con fondos reales${NC}"
echo -e "${GREEN}==================================================================${NC}"
echo -e "${GREEN}  Logs: $LOG_FILE${NC}"
echo -e "${GREEN}==================================================================${NC}"

# Iniciar servicios adicionales
echo -e "${BLUE}Iniciando servicios adicionales...${NC}"

# Iniciar monitor de riesgos en segundo plano
python3 risk_manager.py --symbol $SYMBOL --interval $INTERVAL &
RISK_MANAGER_PID=$!
echo $RISK_MANAGER_PID > risk_manager.pid

# Iniciar integración con API en segundo plano
python3 api_integration.py --symbol $SYMBOL &
API_PID=$!
echo $API_PID > api_integration.pid

# Iniciar el bot principal
echo -e "${BLUE}Iniciando bot principal...${NC}"
screen -dmS sol_bot bash -c "python3 main.py \
    --use-ml \
    --retrain-interval $RETRAIN_INTERVAL \
    --interval $INTERVAL \
    --symbol $SYMBOL \
    --risk $RISK \
    --status-interval $STATUS_INTERVAL \
    > \"$LOG_FILE\" 2>&1"

# Guardar PID del proceso principal
if [ $? -eq 0 ]; then
    # Obtener PID del proceso de screen
    SCREEN_PID=$(screen -ls | grep sol_bot | awk '{print $1}' | cut -d. -f1)
    echo $SCREEN_PID > sol_bot.pid
    
    echo -e "${GREEN}Bot iniciado exitosamente en sesión screen 'sol_bot'${NC}"
    echo -e "${BLUE}Para ver los logs en tiempo real: ${NC}screen -r sol_bot"
    echo -e "${BLUE}Para desconectarse de la sesión sin detener el bot: ${NC}Ctrl+A, D"
    echo -e "${BLUE}Para ver los logs guardados: ${NC}tail -f $LOG_FILE"
    echo -e "${BLUE}Para detener el bot: ${NC}./stop.sh"
else
    echo -e "${RED}Error al iniciar el bot.${NC}"
    # Limpiar procesos en segundo plano si el bot principal falló
    kill $RISK_MANAGER_PID 2>/dev/null
    kill $API_PID 2>/dev/null
    rm risk_manager.pid api_integration.pid 2>/dev/null
fi

echo -e "${GREEN}Bot SOL iniciado en modo REAL con todas las funcionalidades avanzadas.${NC}"

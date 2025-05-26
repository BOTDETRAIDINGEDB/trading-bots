#!/bin/bash
# Script mejorado para iniciar el bot SOL con todas las nuevas funcionalidades

# Colores para mejor legibilidad
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directorio del bot
BOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$BOT_DIR"

# Cargar variables de entorno
if [ -f ".env" ]; then
    source .env
    echo -e "${GREEN}Variables de entorno cargadas${NC}"
else
    echo -e "${YELLOW}Archivo .env no encontrado, usando valores por defecto${NC}"
fi

# Parámetros por defecto
SYMBOL=${SYMBOL:-"SOLUSDT"}
INTERVAL=${INTERVAL:-"15m"}
LOOKBACK=${LOOKBACK:-90}
BALANCE=${BALANCE:-1000}
RISK=${RISK:-0.02}
SIMULATION=${SIMULATION:-true}
USE_ML=${USE_ML:-true}
RETRAIN_INTERVAL=${RETRAIN_INTERVAL:-1440}

# Parsear argumentos
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --symbol)
            SYMBOL="$2"
            shift
            shift
            ;;
        --interval)
            INTERVAL="$2"
            shift
            shift
            ;;
        --lookback)
            LOOKBACK="$2"
            shift
            shift
            ;;
        --balance)
            BALANCE="$2"
            shift
            shift
            ;;
        --risk)
            RISK="$2"
            shift
            shift
            ;;
        --no-simulation)
            SIMULATION=false
            shift
            ;;
        --no-ml)
            USE_ML=false
            shift
            ;;
        --retrain-interval)
            RETRAIN_INTERVAL="$2"
            shift
            shift
            ;;
        --help)
            echo -e "${BLUE}Uso: $0 [opciones]${NC}"
            echo -e "Opciones:"
            echo -e "  --symbol SYMBOL         Par de trading (default: $SYMBOL)"
            echo -e "  --interval INTERVAL     Intervalo de tiempo (default: $INTERVAL)"
            echo -e "  --lookback DAYS         Días de datos históricos (default: $LOOKBACK)"
            echo -e "  --balance AMOUNT        Balance inicial para simulación (default: $BALANCE)"
            echo -e "  --risk PERCENTAGE       Riesgo por operación (default: $RISK)"
            echo -e "  --no-simulation         Desactivar modo simulación"
            echo -e "  --no-ml                 Desactivar modelo de ML"
            echo -e "  --retrain-interval MIN  Intervalo de reentrenamiento en minutos (default: $RETRAIN_INTERVAL)"
            echo -e "  --help                  Mostrar esta ayuda"
            exit 0
            ;;
        *)
            echo -e "${RED}Opción desconocida: $key${NC}"
            exit 1
            ;;
    esac
done

# Crear directorio de logs si no existe
mkdir -p logs

# Configurar nombre de archivo de log
LOG_FILE="logs/sol_bot_${INTERVAL}_$(date +%Y%m%d).log"

# Configurar nombre de archivo de estado
STATE_FILE="sol_bot_${INTERVAL/m/min}_state.json"

echo -e "${GREEN}=== Iniciando Bot SOL Mejorado ===${NC}"
echo -e "${BLUE}Configuración:${NC}"
echo -e "  Symbol: ${YELLOW}$SYMBOL${NC}"
echo -e "  Interval: ${YELLOW}$INTERVAL${NC}"
echo -e "  Lookback: ${YELLOW}$LOOKBACK días${NC}"
echo -e "  Balance inicial: ${YELLOW}$BALANCE USDT${NC}"
echo -e "  Riesgo por operación: ${YELLOW}$(echo "$RISK * 100" | bc)%${NC}"
echo -e "  Modo simulación: ${YELLOW}$SIMULATION${NC}"
echo -e "  Usar ML: ${YELLOW}$USE_ML${NC}"
echo -e "  Intervalo de reentrenamiento: ${YELLOW}$RETRAIN_INTERVAL minutos${NC}"
echo -e "  Archivo de log: ${YELLOW}$LOG_FILE${NC}"
echo -e "  Archivo de estado: ${YELLOW}$STATE_FILE${NC}"

# Verificar si el modelo de ML existe
if [ "$USE_ML" = true ]; then
    MODEL_FILE="${SYMBOL,,}_model.pkl"
    if [ ! -f "$MODEL_FILE" ]; then
        echo -e "${YELLOW}Modelo de ML no encontrado. Entrenando modelo inicial...${NC}"
        python update_model.py --symbol "$SYMBOL" --interval "$INTERVAL" --lookback "$LOOKBACK"
    else
        echo -e "${GREEN}Modelo de ML encontrado: $MODEL_FILE${NC}"
    fi
fi

# Configurar parámetros para el bot
BOT_ARGS="--symbol $SYMBOL --interval $INTERVAL --lookback $LOOKBACK --balance $BALANCE --risk $RISK"

if [ "$SIMULATION" = true ]; then
    BOT_ARGS="$BOT_ARGS --simulation"
fi

if [ "$USE_ML" = true ]; then
    BOT_ARGS="$BOT_ARGS --use-ml --retrain-interval $RETRAIN_INTERVAL"
fi

# Iniciar el gestor de riesgos
echo -e "${GREEN}Iniciando gestor de riesgos...${NC}"
python risk_manager.py --state-file "$STATE_FILE" &

# Iniciar el bot en segundo plano
echo -e "${GREEN}Iniciando bot SOL...${NC}"
nohup python main.py $BOT_ARGS > "$LOG_FILE" 2>&1 &
BOT_PID=$!
echo -e "${GREEN}Bot iniciado con PID: $BOT_PID${NC}"
echo $BOT_PID > sol_bot.pid

# Configurar monitoreo periódico
echo -e "${GREEN}Configurando monitoreo periódico...${NC}"

# Crear script de monitoreo
cat > monitor_cron.sh << EOF
#!/bin/bash
cd "$BOT_DIR"

# Verificar si el bot sigue en ejecución
if [ -f sol_bot.pid ]; then
    PID=\$(cat sol_bot.pid)
    if ps -p \$PID > /dev/null; then
        echo "Bot en ejecución (PID: \$PID)"
    else
        echo "Bot no está en ejecución. Reiniciando..."
        ./start_enhanced.sh --symbol "$SYMBOL" --interval "$INTERVAL" --lookback "$LOOKBACK" --balance "$BALANCE" --risk "$RISK" $([ "$SIMULATION" = true ] && echo "--simulation") $([ "$USE_ML" = true ] && echo "--use-ml --retrain-interval $RETRAIN_INTERVAL")
    fi
else
    echo "Archivo PID no encontrado. Reiniciando bot..."
    ./start_enhanced.sh --symbol "$SYMBOL" --interval "$INTERVAL" --lookback "$LOOKBACK" --balance "$BALANCE" --risk "$RISK" $([ "$SIMULATION" = true ] && echo "--simulation") $([ "$USE_ML" = true ] && echo "--use-ml --retrain-interval $RETRAIN_INTERVAL")
fi

# Ejecutar monitor de rendimiento
python performance_monitor.py --state-file "$STATE_FILE"

# Ejecutar integración con API
python api_integration.py --state-file "$STATE_FILE"

# Ejecutar gestor de riesgos
python risk_manager.py --state-file "$STATE_FILE"
EOF

chmod +x monitor_cron.sh

# Configurar cron para ejecutar el monitoreo cada hora
(crontab -l 2>/dev/null; echo "0 * * * * $BOT_DIR/monitor_cron.sh >> $BOT_DIR/logs/monitor.log 2>&1") | crontab -

echo -e "${GREEN}Monitoreo configurado para ejecutarse cada hora${NC}"
echo -e "${GREEN}Bot SOL iniciado correctamente${NC}"
echo -e "${YELLOW}Para detener el bot: kill \$(cat sol_bot.pid)${NC}"
echo -e "${YELLOW}Para ver los logs: tail -f $LOG_FILE${NC}"

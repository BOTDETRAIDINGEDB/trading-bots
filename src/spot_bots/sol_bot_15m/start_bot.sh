#!/bin/bash

# Script para iniciar el bot SOL de 15 minutos
# Creado: 25 de mayo de 2025

# Cambiar al directorio del proyecto
cd "$(dirname "$0")/../../.."
PROJECT_DIR=$(pwd)

# Activar el entorno virtual
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Entorno virtual del bot de trading activado."
else
    echo "No se encontró el entorno virtual en $PROJECT_DIR/venv"
    exit 1
fi

# Configurar PYTHONPATH para encontrar los módulos correctamente
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR

# Parámetros por defecto
INTERVAL="15m"
SYMBOL="SOLUSDT"
SIMULATION=true
BALANCE=1000
RISK=0.02
STATUS_INTERVAL=6

# Procesar argumentos
while [[ $# -gt 0 ]]; do
    case $1 in
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        --symbol)
            SYMBOL="$2"
            shift 2
            ;;
        --real)
            SIMULATION=false
            shift
            ;;
        --balance)
            BALANCE="$2"
            shift 2
            ;;
        --risk)
            RISK="$2"
            shift 2
            ;;
        --status-interval)
            STATUS_INTERVAL="$2"
            shift 2
            ;;
        *)
            echo "Argumento desconocido: $1"
            shift
            ;;
    esac
done

# Construir el comando
CMD="python src/spot_bots/sol_bot_15m/main.py --interval $INTERVAL --symbol $SYMBOL"

if $SIMULATION; then
    CMD="$CMD --simulation --balance $BALANCE"
fi

CMD="$CMD --risk $RISK --status-interval $STATUS_INTERVAL"

# Mostrar información
echo "Iniciando bot SOL con los siguientes parámetros:"
echo "- Intervalo: $INTERVAL"
echo "- Símbolo: $SYMBOL"
echo "- Modo: $(if $SIMULATION; then echo "Simulación"; else echo "Real"; fi)"
if $SIMULATION; then
    echo "- Balance inicial: $BALANCE USDT"
fi
echo "- Riesgo por operación: $(echo "$RISK * 100" | bc)%"
echo "- Intervalo de estado: $STATUS_INTERVAL horas"
echo ""

# Ejecutar el bot
echo "Ejecutando: $CMD"
$CMD

# En caso de error o finalización
echo "Bot SOL detenido."

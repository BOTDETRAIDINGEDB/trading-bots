#!/bin/bash
# Script para iniciar el bot SOL en modo simulación con reentrenamiento cada 15 minutos

# Cargar variables de entorno
source ~/.bashrc

# Directorio del bot
BOT_DIR="/home/edisonbautistaruiz2025/new-trading-bots/src/spot_bots/sol_bot_15m"

# Crear directorio de logs si no existe
mkdir -p "$BOT_DIR/logs"

# Iniciar el bot en una sesión de screen con redirección de errores
cd "$BOT_DIR"
screen -dmS sol_bot_15m bash -c "python3 main.py --use-ml --retrain-interval 15 --interval 15m --symbol SOLUSDT --simulation --balance 1000 2> logs/error.log"

echo "Bot SOL iniciado en modo SIMULACIÓN en sesión screen 'sol_bot_15m'"
echo "Para ver los logs: screen -r sol_bot_15m"
echo "Para ver errores: cat $BOT_DIR/logs/error.log"

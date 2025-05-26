#!/bin/bash

# Script para ejecutar el bot XRP con la configuración correcta de PYTHONPATH
# Creado: 25 de mayo de 2025

# Cambiar al directorio del proyecto
cd "$(dirname "$0")"

# Activar el entorno virtual (ajusta la ruta si es necesario)
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "../venv" ]; then
    source ../venv/bin/activate
fi

# Configurar PYTHONPATH para incluir el directorio del bot XRP
export PYTHONPATH=$PYTHONPATH:$(pwd)/src/spot_bots/xrp_bot

# Ejecutar el bot XRP con los parámetros especificados
python src/spot_bots/xrp_bot/main.py --interval 30m --simulation

# Nota: Para ejecutar en una sesión de screen, usa:
# screen -dmS xrp_bot_30m bash -c "./run_xrp_bot.sh; exec bash"

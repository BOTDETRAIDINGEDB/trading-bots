#!/bin/bash

# Activar entorno virtual
source venv/bin/activate

# Iniciar bot en screen con modo simulación
screen -dmS trading_bot bash -c "python main.py --train --symbols SOLUSDT --interval 1h --lookback 90 --simulation --initial_balance 1000; exec bash"

echo "Bot de trading iniciado en modo simulación. Para ver la salida, ejecuta: screen -r trading_bot"

#!/bin/bash

# Activar entorno virtual
source venv/bin/activate

# Iniciar bot en screen con modo simulación
screen -dmS xrp_bot bash -c "python main.py --train --symbols XRPUSDT --interval 30m --lookback 90 --simulation --initial_balance 1000; exec bash"

# Configurar respaldo diario (a las 00:00)
(crontab -l 2>/dev/null; echo "0 0 * * * /home/edisonbautistaruiz2025/trading-bot-xrp/backup_bot.sh") | crontab - 2>/dev/null || echo "No se pudo configurar crontab, ejecutando respaldo manual diario"

# Si crontab no está disponible, configurar un respaldo manual diario usando screen
if [ $? -ne 0 ]; then
    screen -dmS xrp_backup bash -c "while true; do /home/edisonbautistaruiz2025/trading-bot-xrp/backup_bot.sh; sleep 86400; done"
fi

echo "Bot de trading XRP iniciado en modo simulación. Para ver la salida, ejecuta: screen -r xrp_bot"

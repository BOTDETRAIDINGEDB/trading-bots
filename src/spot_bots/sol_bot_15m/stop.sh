#!/bin/bash
# Script para detener el bot SOL
# Autor: Cascade
# Fecha: 2025-05-26

# Configuración
LOG_FILE="bot_stop.log"
echo "$(date): Iniciando proceso de detención del bot" > $LOG_FILE

# === DETENER TODOS LOS SERVICIOS Y PROCESOS RELACIONADOS CON EL BOT SOL ===
# Detener procesos main.py, adaptive_main.py, cloud_monitor.py, sesiones screen, y limpiar archivos temporales

echo "Deteniendo todos los procesos del bot SOL (simulación y real, local y cloud)..." | tee -a $LOG_FILE

# Detener procesos Python principales
for PROC in "main.py" "adaptive_main.py"; do
    PIDS=$(ps aux | grep "[p]ython3 $PROC" | awk '{print $2}')
    for PID in $PIDS; do
        echo "Deteniendo proceso $PROC con PID: $PID" | tee -a $LOG_FILE
        kill $PID
        sleep 1
        if ps -p $PID > /dev/null; then
            echo "Forzando terminación de $PROC con PID: $PID" | tee -a $LOG_FILE
            kill -9 $PID
        fi
    done
done

# Detener cloud_monitor.py si está en ejecución
CLOUD_MON_PIDS=$(ps aux | grep "[c]loud_monitor.py" | awk '{print $2}')
for PID in $CLOUD_MON_PIDS; do
    echo "Deteniendo cloud_monitor.py con PID: $PID" | tee -a $LOG_FILE
    kill $PID
    sleep 1
    if ps -p $PID > /dev/null; then
        kill -9 $PID
    fi
}

echo "Deteniendo sesiones screen relacionadas..." | tee -a $LOG_FILE
for SCR in $(screen -ls | grep -E 'sol_bot|sol_bot_15m' | awk '{print $1}'); do
    echo "Cerrando sesión screen: $SCR" | tee -a $LOG_FILE
    screen -S "$SCR" -X quit
    sleep 1
}

# Eliminar archivos temporales si existen
if [ -f "bot_running.lock" ]; then
        rm bot_running.lock
        echo "Archivo de bloqueo eliminado" >> $LOG_FILE
    fi
    
    echo "$(date): Proceso de detención completado" >> $LOG_FILE
    exit 0
fi

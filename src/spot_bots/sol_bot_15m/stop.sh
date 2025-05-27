#!/bin/bash
# Script para detener el bot SOL
# Autor: Cascade
# Fecha: 2025-05-26

# Configuración
LOG_FILE="bot_stop.log"
echo "$(date): Iniciando proceso de detención del bot" > $LOG_FILE

# Encontrar el PID del proceso del bot
BOT_PID=$(ps aux | grep "[p]ython3 adaptive_main.py" | awk '{print $2}')

if [ -z "$BOT_PID" ]; then
    echo "El bot no está en ejecución" | tee -a $LOG_FILE
    exit 0
else
    echo "Deteniendo bot con PID: $BOT_PID" | tee -a $LOG_FILE
    
    # Intentar detener el bot de forma segura primero
    kill $BOT_PID
    echo "$(date): Señal de terminación enviada al proceso $BOT_PID" >> $LOG_FILE
    
    # Esperar a que el proceso termine (máximo 10 segundos)
    COUNTER=0
    while ps -p $BOT_PID > /dev/null && [ $COUNTER -lt 10 ]; do
        sleep 1
        COUNTER=$((COUNTER+1))
        echo "Esperando que el bot termine... ($COUNTER/10)" | tee -a $LOG_FILE
    done
    
    # Verificar si el proceso terminó
    if ps -p $BOT_PID > /dev/null; then
        echo "El bot no se detuvo correctamente, forzando terminación" | tee -a $LOG_FILE
        kill -9 $BOT_PID
        echo "$(date): Señal SIGKILL enviada al proceso $BOT_PID" >> $LOG_FILE
        
        # Verificar una vez más
        sleep 2
        if ps -p $BOT_PID > /dev/null; then
            echo "ERROR: No se pudo detener el bot" | tee -a $LOG_FILE
            exit 1
        else
            echo "Bot detenido correctamente (forzado)" | tee -a $LOG_FILE
        fi
    else
        echo "Bot detenido correctamente" | tee -a $LOG_FILE
    fi
    
    # Eliminar archivos temporales si existen
    if [ -f "bot_running.lock" ]; then
        rm bot_running.lock
        echo "Archivo de bloqueo eliminado" >> $LOG_FILE
    fi
    
    echo "$(date): Proceso de detención completado" >> $LOG_FILE
    exit 0
fi

#!/bin/bash
# cleanup_bot_sessions.sh - Script para limpiar todas las sesiones del bot y reiniciar limpiamente

echo "===== LIMPIEZA DE SESIONES DEL BOT SOL ====="
echo "Deteniendo todas las sesiones existentes..."

# Detener todas las sesiones del bot principal
echo "Deteniendo sesiones del bot principal..."
for session in $(screen -ls | grep sol_bot_15m | grep -v "sol_bot_15m_adaptive" | awk '{print $1}'); do
    echo "Deteniendo sesión: $session"
    screen -S $session -X quit
done

# Detener todas las sesiones del bot adaptativo
echo "Deteniendo sesiones del bot adaptativo..."
for session in $(screen -ls | grep sol_bot_15m_adaptive | awk '{print $1}'); do
    echo "Deteniendo sesión: $session"
    screen -S $session -X quit
done

echo "Esperando 5 segundos para asegurar que todas las sesiones se hayan cerrado..."
sleep 5

# Verificar que no queden sesiones
echo "Verificando sesiones restantes:"
screen -ls

# Eliminar archivos de control que podrían impedir notificaciones
echo "Eliminando archivos de control..."
if [ -f .last_startup ]; then
    rm -f .last_startup
    echo "Archivo .last_startup eliminado"
fi

# Verificar y mostrar las credenciales configuradas
echo "Verificando credenciales configuradas:"
echo "TELEGRAM_BOT_TOKEN: ${TELEGRAM_BOT_TOKEN:0:3}...${TELEGRAM_BOT_TOKEN: -3}"
echo "TELEGRAM_CHAT_ID: $TELEGRAM_CHAT_ID"

echo "===== REINICIANDO BOT SOL LIMPIO ====="
# Ejecutar el script de inicio
./start_cloud_simulation.sh

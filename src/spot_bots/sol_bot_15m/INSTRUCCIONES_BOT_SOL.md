# Instrucciones para el Bot SOL

## Inicialización del Bot

Para iniciar el bot SOL correctamente, siempre utiliza el script de limpieza:

```bash
cd ~/new-trading-bots/src/spot_bots/sol_bot_15m
./cleanup_bot_sessions.sh
```

Este script realiza las siguientes acciones:
1. Detiene todas las sesiones existentes del bot (principal y adaptativo)
2. Elimina archivos de control que podrían bloquear las notificaciones
3. Configura las variables de entorno necesarias (Binance API, Telegram)
4. Inicia ambos componentes del bot en sesiones screen separadas

> ⚠️ **IMPORTANTE**: Nunca intentes iniciar el bot directamente con `python3 main.py`, ya que esto no configurará las variables de entorno necesarias.

## Verificación del Estado

Para verificar que el bot está funcionando correctamente:

```bash
# Ver las sesiones activas
screen -ls

# Deberías ver dos sesiones:
# - sol_bot_15m (bot principal)
# - sol_bot_15m_adaptive (componente adaptativo)
```

## Monitoreo de Logs

Para monitorear los logs del bot:

```bash
# Ver los últimos logs del bot principal
cat ~/new-trading-bots/src/spot_bots/sol_bot_15m/logs/sol_bot_15m_cloud_simulation_*.log | tail -n 50

# Ver los logs en tiempo real
tail -f ~/new-trading-bots/src/spot_bots/sol_bot_15m/logs/sol_bot_15m_cloud_simulation_*.log
```

## Configuración Actual

El bot está configurado con los siguientes parámetros:
- Símbolo: SOLUSDT
- Intervalo: 15m
- Balance de simulación: 100 USDT
- Reentrenamiento ML: Cada 15 minutos
- Riesgo por operación: 0.02%
- Modo: Simulación de aprendizaje

## Modo de Aprendizaje

El bot opera en modo de aprendizaje hasta que alcanza un win rate del 55%. Durante este período:
- Limita el tamaño de las operaciones al 50% del tamaño normal
- Recopila datos y mejora su modelo de ML
- Envía notificaciones periódicas a Telegram sobre el estado del mercado

## Solución de Problemas

Si el bot no funciona correctamente:

1. Verifica que las credenciales estén configuradas correctamente en `credentials.json`
2. Asegúrate de que el script `cleanup_bot_sessions.sh` tenga permisos de ejecución
3. Revisa los logs para identificar errores específicos
4. Si es necesario, actualiza el código con `git pull origin main` antes de reiniciar

## Actualizaciones de Código

Para actualizar el código del bot:

```bash
cd ~/new-trading-bots
git pull origin main
cd src/spot_bots/sol_bot_15m
./cleanup_bot_sessions.sh
```

Esto asegurará que estás ejecutando la versión más reciente del bot con todas las correcciones y mejoras.

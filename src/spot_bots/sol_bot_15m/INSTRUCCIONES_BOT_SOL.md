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

## Soporte para Posiciones SHORT

El bot ahora soporta posiciones SHORT, lo que le permite operar en mercados bajistas. Las mejoras incluyen:

1. Interpretación correcta de señales de venta (-1)
2. Entrada en posiciones SHORT cuando sea apropiado
3. Gestión adecuada de stop loss, take profit y trailing stop para posiciones cortas
4. Cálculo correcto de ganancias/pérdidas para posiciones SHORT

### Comandos para Actualizar y Ejecutar el Bot con Soporte SHORT

#### Actualización del Código
```bash
# Conectar a la máquina virtual
ssh edisonbautistaruiz2025@iatraidingbots

# Navegar al repositorio
cd ~/new-trading-bots

# Actualizar desde GitHub
git pull origin main

# Verificar que los cambios se aplicaron correctamente
git log -1
```

#### Opciones de Ejecución

**Opción 1: Iniciar con el script estándar (modo REAL)**
```bash
# Navegar al directorio del bot SOL
cd ~/new-trading-bots/src/spot_bots/sol_bot_15m

# Ejecutar el script de inicio
./start_bot.sh
```

**Opción 2: Limpieza y reinicio completo**
```bash
# Navegar al directorio del bot SOL
cd ~/new-trading-bots/src/spot_bots/sol_bot_15m

# Ejecutar el script de limpieza (detendrá todas las sesiones y reiniciará en modo simulación)
./cleanup_bot_sessions.sh
```

**Opción 3: Iniciar manualmente con parámetros específicos**
```bash
# Para modo simulación
screen -dmS sol_bot_sim python3 main.py --symbol SOLUSDT --interval 15m --simulation --use-ml --risk 0.02

# Para modo real con parámetros personalizados
screen -dmS sol_bot python3 main.py --symbol SOLUSDT --interval 15m --use-ml --retrain-interval 60 --risk 0.02 --status-interval 1
```

#### Monitoreo y Control

**Ver logs y monitorear el bot**
```bash
# Ver los logs en tiempo real conectándote a la sesión screen
screen -r sol_bot

# Para desconectarte de la sesión sin detener el bot: Ctrl+A, D

# Ver los logs guardados
tail -f logs/sol_bot_real_*.log
```

**Detener el bot**
```bash
# Detener el bot usando el script de detención
./stop.sh
```

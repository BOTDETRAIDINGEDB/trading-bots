# Modo Simulación de Aprendizaje para Bot SOL

## Introducción

Este documento explica cómo utilizar el modo de simulación de aprendizaje para el bot de trading SOL. Este modo especial permite que el bot opere con datos de mercado reales de Binance, pero utilizando un balance ficticio de USDT, lo que facilita el aprendizaje y la optimización del modelo sin arriesgar fondos reales.

## Características del Modo Simulación de Aprendizaje

- **Datos Reales**: Utiliza la API de Binance con credenciales reales para obtener datos de mercado en tiempo real
- **Balance Ficticio**: Opera con un balance simulado (100 USDT por defecto)
- **Aprendizaje Continuo**: Reentrenamiento automático del modelo ML cada 15 minutos
- **Notificaciones Reales**: Envía notificaciones a Telegram sobre operaciones y estado
- **Registro Detallado**: Mantiene logs completos de todas las decisiones y operaciones
- **Optimizaciones Cloud**: Configurado para rendimiento óptimo en Google Cloud VM

## Requisitos

1. Credenciales válidas de Binance en el archivo `credentials.json`
2. Configuración correcta de Telegram (si se desean notificaciones)
3. Dependencias instaladas (TensorFlow, pandas, numpy, etc.)

## Cómo Iniciar el Bot en Modo Simulación de Aprendizaje

### En Windows

1. Abre una terminal (PowerShell o CMD) en el directorio del bot
2. Ejecuta el siguiente comando:

```powershell
bash start_simulation_learning.sh
```

### En Linux/Mac

1. Abre una terminal en el directorio del bot
2. Asegúrate de que el script tenga permisos de ejecución:

```bash
chmod +x start_simulation_learning.sh
```

3. Ejecuta el script:

```bash
./start_simulation_learning.sh
```

## Parámetros Configurables

Puedes modificar los siguientes parámetros en el script `start_simulation_learning.sh`:

- `SIMULATION_BALANCE`: Balance ficticio en USDT (por defecto: 100)
- `RETRAIN_INTERVAL`: Intervalo de reentrenamiento en minutos (por defecto: 15)
- `RISK`: Porcentaje de riesgo por operación (por defecto: 0.02 = 2%)
- `STATUS_INTERVAL`: Frecuencia de actualizaciones de estado en horas (por defecto: 1)

## Monitoreo del Bot

### Logs

Los logs se guardan en el directorio `logs` con el formato `sol_bot_15m_simulation_YYYYMMDD_HHMMSS.log`. Puedes monitorearlos en tiempo real con:

```bash
tail -f logs/sol_bot_15m_simulation_*.log
```

### Sesión Screen (Linux/Mac)

Si el bot se inició en una sesión screen, puedes:

- Ver la sesión: `screen -r sol_bot_15m`
- Desconectarte sin detener el bot: Presiona `Ctrl+A, D`
- Listar sesiones activas: `screen -ls`

### Notificaciones Telegram

Si has configurado correctamente Telegram, recibirás notificaciones sobre:

- Inicio del bot
- Operaciones de entrada y salida
- Actualizaciones periódicas de estado
- Errores o problemas

## Análisis de Rendimiento

Durante la fase de aprendizaje, es importante analizar el rendimiento del bot para optimizar sus parámetros:

1. **Métricas Clave**:
   - Win Rate (% de operaciones ganadoras)
   - Profit Factor (relación entre ganancias y pérdidas)
   - Drawdown máximo (caída máxima desde un pico)
   - Número total de operaciones

2. **Ajustes Recomendados**:
   - Si el Win Rate es bajo (<40%), considera ajustar los parámetros de entrada
   - Si el Drawdown es alto (>15%), considera reducir el riesgo por operación
   - Si hay pocas operaciones, considera ajustar la sensibilidad de las señales

## Transición a Modo Real

Una vez que el bot haya demostrado un rendimiento consistente en modo simulación durante al menos 2-4 semanas, puedes considerar la transición a modo real:

1. Analiza todas las métricas de rendimiento
2. Comienza con un balance pequeño (10-20% del capital planeado)
3. Utiliza el script `start_bot.sh` en lugar de `start_simulation_learning.sh`
4. Monitorea de cerca las primeras operaciones reales

## Solución de Problemas

### El bot no inicia

- Verifica que todas las dependencias estén instaladas
- Comprueba que el archivo `credentials.json` exista y tenga las credenciales correctas
- Revisa los logs de error en `logs/error.log`

### Errores de API de Binance

- Verifica que las credenciales de Binance sean válidas
- Comprueba que tengas permisos de lectura (para modo simulación)
- Asegúrate de que la IP desde donde se ejecuta el bot esté autorizada en Binance

### Problemas con el modelo ML

- Verifica que TensorFlow esté correctamente instalado
- Comprueba que haya suficiente memoria disponible (mínimo 2GB)
- Revisa los logs para errores específicos de TensorFlow

## Conclusión

El modo simulación de aprendizaje es una herramienta poderosa para optimizar tu bot de trading antes de arriesgar capital real. Utiliza este tiempo para ajustar parámetros, mejorar la estrategia y asegurarte de que el bot funcione de manera confiable y rentable.

---

*Última actualización: Mayo 2025*

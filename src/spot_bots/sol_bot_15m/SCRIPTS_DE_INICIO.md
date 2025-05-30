# Scripts de Inicio del Bot SOL

Este documento describe los diferentes scripts de inicio disponibles para el bot SOL, sus propósitos específicos y cuándo utilizar cada uno.

## Scripts Principales

> **¡NUEVO FLUJO AUTOMATIZADO!**
> 
> Ahora solo necesitas ejecutar `./start_bot.sh` (modo real) o `./start_simulation_learning.sh` (modo simulación). Ambos scripts detectan automáticamente si están en Google Cloud VM y delegan la ejecución al script cloud correspondiente. El usuario no necesita preocuparse por el entorno: el despliegue y la gestión de servicios se realiza automáticamente.

### 1. `start_bot.sh`

**Propósito**: Iniciar el bot en modo REAL con todas las funcionalidades avanzadas.

**Características**:
- Operaciones con fondos reales en Binance
- Reentrenamiento del modelo ML cada 60 minutos
- Integración con el gestor de riesgos
- Integración con la API
- Notificaciones mejoradas de Telegram
- Detección automática del sistema operativo y del entorno cloud
- **Despliegue automático en Google Cloud VM si corresponde**

**Cuándo usar**:
- Cuando el bot ha sido probado exhaustivamente
- Cuando estás listo para operar con fondos reales
- Cuando necesitas todas las funcionalidades avanzadas

**Comando**:
```bash
./start_bot.sh
```

### 2. `start_simulation_learning.sh`

**Propósito**: Iniciar el bot en modo SIMULACIÓN DE APRENDIZAJE.

**Características**:
- Usa credenciales reales de Binance para obtener datos de mercado
- Opera con un balance ficticio de 100 USDT
- Reentrenamiento del modelo ML cada 15 minutos
- Optimizado para aprendizaje y pruebas
- Detección automática del sistema operativo y del entorno cloud
- **Despliegue automático en Google Cloud VM si corresponde**

**Cuándo usar**:
- Durante el desarrollo y prueba de estrategias
- Para permitir que el bot aprenda sin arriesgar fondos reales
- Para evaluar el rendimiento del bot con datos de mercado reales

**Comando**:
```bash
./start_simulation_learning.sh
```

### 3. `start_cloud_simulation.sh`

**Propósito**: Iniciar el bot en modo SIMULACIÓN DE APRENDIZAJE específicamente en Google Cloud VM.

**Características**:
- Configuración optimizada para Google Cloud VM
- Rutas absolutas específicas para el entorno de la nube
- Variables de entorno optimizadas para TensorFlow en la nube
- Verificación automática de dependencias y credenciales
- Ejecución en sesión screen para mantener el bot en segundo plano

**Cuándo usar**:
- Cuando ejecutas el bot en una máquina virtual de Google Cloud
- Para simulaciones de larga duración en la nube
- Para aprovechar los recursos de la nube para el aprendizaje del bot

**Comando**:
```bash
./start_cloud_simulation.sh
```

## Otros scripts útiles

### `stop.sh`

**Propósito**: Detener todas las instancias del bot y servicios relacionados.

**Características**:
- Detiene el bot principal
- Detiene servicios adicionales (gestor de riesgos, integración API)
- Limpia archivos temporales y PIDs

**Comando**:
```bash
./stop.sh
```

### `cleanup_bot_sessions.sh`

**Propósito**: Limpiar todas las sesiones del bot y reiniciar limpiamente.

**Características**:
- Detiene todas las sesiones existentes del bot (principal y adaptativo)
- Elimina archivos de control que podrían bloquear las notificaciones
- Verifica las credenciales configuradas
- Reinicia el bot en modo simulación

**Cuándo usar**:
- Cuando necesitas hacer una limpieza completa de todas las sesiones
- Cuando el bot se ha quedado en un estado inconsistente
- Después de actualizar el código desde GitHub

**Comando**:
```bash
./cleanup_bot_sessions.sh
```

## Guía de Selección

1. **¿Estás en fase de desarrollo y pruebas?**
   - Usa `start_simulation_learning.sh`

2. **¿Estás desplegando en Google Cloud VM?**
   - Usa `start_cloud_simulation.sh`

3. **¿Estás listo para operar con fondos reales?**
   - Usa `start_bot.sh`

## Soporte para Posiciones SHORT

**Nueva funcionalidad**: El bot ahora soporta posiciones SHORT, lo que le permite operar en mercados bajistas.

**Mejoras implementadas**:
- Interpretación correcta de señales de venta (-1)
- Entrada en posiciones SHORT cuando sea apropiado
- Gestión adecuada de stop loss, take profit y trailing stop para posiciones cortas
- Cálculo correcto de ganancias/pérdidas para posiciones SHORT

### Comandos para Actualizar y Ejecutar con Soporte SHORT

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

# Ejecutar el script de limpieza
./cleanup_bot_sessions.sh
```

**Opción 3: Iniciar manualmente con parámetros específicos**
```bash
# Para modo simulación
screen -dmS sol_bot_sim python3 main.py --symbol SOLUSDT --interval 15m --simulation --use-ml --risk 0.02

# Para modo real con parámetros personalizados
screen -dmS sol_bot python3 main.py --symbol SOLUSDT --interval 15m --use-ml --retrain-interval 60 --risk 0.02 --status-interval 1
```

## Notas Importantes

- **NUNCA** ejecutes múltiples instancias del bot para el mismo par de trading (SOLUSDT)
- Siempre utiliza `stop.sh` para detener el bot correctamente antes de iniciar una nueva instancia
- Revisa los logs regularmente para monitorear el rendimiento del bot
- Las credenciales de Binance y Telegram deben estar correctamente configuradas en `credentials.json`

---

*Última actualización: Mayo 2025*

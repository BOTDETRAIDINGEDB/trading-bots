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

## Guía de Selección

1. **¿Estás en fase de desarrollo y pruebas?**
   - Usa `start_simulation_learning.sh`

2. **¿Estás desplegando en Google Cloud VM?**
   - Usa `start_cloud_simulation.sh`

3. **¿Estás listo para operar con fondos reales?**
   - Usa `start_bot.sh`

## Notas Importantes

- **NUNCA** ejecutes múltiples instancias del bot para el mismo par de trading (SOLUSDT)
- Siempre utiliza `stop.sh` para detener el bot correctamente antes de iniciar una nueva instancia
- Revisa los logs regularmente para monitorear el rendimiento del bot
- Las credenciales de Binance y Telegram deben estar correctamente configuradas en `credentials.json`

---

*Última actualización: Mayo 2025*

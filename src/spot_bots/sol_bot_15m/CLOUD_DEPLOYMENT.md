# Despliegue del Bot SOL en Google Cloud

## Introducción

Este documento proporciona instrucciones detalladas para desplegar y ejecutar el bot de trading SOL en Google Cloud Platform (GCP). Incluye tanto la configuración general como instrucciones específicas para el modo de simulación de aprendizaje, que permite que el bot opere con datos de mercado reales pero utilizando un balance ficticio de USDT.

## Requisitos Previos

1. **Cuenta de Google Cloud Platform**
   - Tener una cuenta activa en GCP con facturación habilitada
   - Tener permisos de administrador en el proyecto
   - Tener una VM configurada (recomendado: e2-standard-2 o superior)

2. **Google Cloud SDK** (para configuración inicial)
   - Instalar [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
   - Iniciar sesión con `gcloud auth login`

3. **Dependencias**
   - Python 3.7 o superior
   - TensorFlow 2.4.0 o superior
   - Otras dependencias listadas en `requirements.txt`

4. **Credenciales**
   - API Key y Secret de Binance configurados en `credentials.json`
   - Token de Telegram y Chat ID para notificaciones

## Estructura de Directorios Recomendada

```
/home/edisonbautistaruiz2025/
├── trading-bots/                  # Repositorio principal del bot
│   └── src/
│       └── spot_bots/
│           └── sol_bot_15m/       # Bot SOL de 15 minutos
│               ├── main.py
│               ├── start_cloud_simulation.sh
│               └── src/
│                   └── ...
└── trading-bots-api/              # Repositorio de la API
    └── credentials.json           # Archivo de credenciales
```

## Configuración Inicial de Google Cloud

### 1. Crear una Instancia de VM

1. Ve a la [Consola de Google Cloud](https://console.cloud.google.com/)
2. Navega a Compute Engine > Instancias de VM
3. Haz clic en "Crear instancia"
4. Configura la instancia:
   - Nombre: `trading-bot-vm`
   - Región: Elige la más cercana a ti
   - Tipo de máquina: e2-standard-2 (2 vCPU, 8 GB de memoria)
   - Disco de arranque: Ubuntu 20.04 LTS
   - Tamaño del disco: 20 GB SSD
   - Habilita HTTP y HTTPS
5. Haz clic en "Crear"

### 2. Conectarse a la VM

```bash
gcloud compute ssh [NOMBRE_INSTANCIA] --zone=[ZONA]
```

O usa la consola web de SSH desde la interfaz de Google Cloud.

## Pasos para el Despliegue

### 1. Preparación del Entorno

```bash
# Actualizar sistema
sudo apt-get update
sudo apt-get upgrade -y

# Instalar dependencias del sistema
sudo apt-get install -y python3-pip python3-dev git screen

# Clonar repositorios
git clone https://github.com/[TU_USUARIO]/trading-bots.git
git clone https://github.com/[TU_USUARIO]/trading-bots-api.git

# Instalar dependencias de Python
cd trading-bots
pip3 install -r requirements.txt

# Instalar TensorFlow
pip3 install tensorflow==2.4.0
```

### 2. Configuración de Credenciales

Crea o actualiza el archivo `credentials.json` en el directorio `trading-bots-api`:

```bash
cd ~/trading-bots-api
nano credentials.json
```

Asegúrate de que el archivo contenga las siguientes claves:

```json
{
  "env": {
    "BINANCE_API_KEY": "tu_api_key_de_binance",
    "BINANCE_API_SECRET": "tu_api_secret_de_binance",
    "TELEGRAM_BOT_TOKEN": "tu_token_de_bot_telegram",
    "TELEGRAM_CHAT_ID": "tu_chat_id_de_telegram",
    "JWT_SECRET": "un_secreto_seguro_para_jwt"
  }
}
```

### 3. Preparación del Bot

Configura los permisos de ejecución para los scripts:

```bash
cd ~/trading-bots/src/spot_bots/sol_bot_15m
chmod +x start_cloud_simulation.sh
```

### 4. Verificación de Compatibilidad con Google Cloud VM

Ejecuta el script de verificación de compatibilidad:

```bash
cd ~/trading-bots/src/spot_bots/sol_bot_15m
python3 src/utils/check_cloud_compatibility.py --fix
```

Este script verificará que todos los componentes necesarios estén correctamente configurados para Google Cloud VM y realizará correcciones automáticas si es necesario.

### 5. Limpieza de Archivos Redundantes

Ejecuta el script de limpieza para eliminar archivos innecesarios:

```bash
python3 src/utils/cleanup_redundant_files.py --clean
```

## Modo de Simulación de Aprendizaje

El modo de simulación de aprendizaje permite que el bot opere con datos de mercado reales pero utilizando un balance ficticio de USDT. Esto es ideal para probar estrategias y permitir que el bot aprenda sin arriesgar fondos reales.

### Iniciar el Bot en Modo Simulación

```bash
./start_cloud_simulation.sh
```

Este script:
- Configura las variables de entorno optimizadas para Google Cloud VM
- Verifica la disponibilidad de las credenciales
- Comprueba la instalación de TensorFlow
- Ejecuta los scripts de verificación y limpieza
- Inicia el bot en una sesión de screen con un balance ficticio de 100 USDT

### Configuraciones Principales

El script `start_cloud_simulation.sh` utiliza las siguientes configuraciones:

- **Balance de simulación**: 100 USDT
- **Intervalo de reentrenamiento**: Cada 15 minutos
- **Riesgo por operación**: 2%
- **Variables de entorno optimizadas** para Google Cloud VM:
  - `CLOUD_ENV=true`
  - `MEMORY_LIMIT_MB=2048`
  - `TF_DETERMINISTIC=true`
  - `USE_MULTIPROCESSING=false`
  - `TF_CPP_MIN_LOG_LEVEL=2`
  - `TF_FORCE_GPU_ALLOW_GROWTH=true`

## Monitoreo y Gestión

### Ver Logs en Tiempo Real

```bash
# Conectarse a la sesión screen
screen -r sol_bot_15m

# Desconectarse sin detener el bot
# Presiona Ctrl+A, luego D
```

### Ver Logs Guardados

```bash
tail -f ~/trading-bots/src/spot_bots/sol_bot_15m/logs/sol_bot_15m_cloud_simulation_*.log
```

### Detener el Bot

```bash
# Encontrar la sesión screen
screen -ls

# Terminar la sesión
screen -X -S sol_bot_15m quit
```

## Solución de Problemas

### Error de Memoria

Si el bot se detiene con errores de memoria:

1. Ajusta la variable `MEMORY_LIMIT_MB` en `start_cloud_simulation.sh`
2. Considera aumentar la memoria de la VM en Google Cloud Console

### Problemas de Conexión con Binance

1. Verifica que las credenciales sean correctas
2. Asegúrate de que la IP de la VM esté autorizada en Binance
3. Comprueba los logs para mensajes de error específicos

### Problemas con TensorFlow

```bash
# Verificar la instalación de TensorFlow
python3 -c "import tensorflow as tf; print(tf.__version__)"

# Reinstalar si es necesario
pip3 uninstall -y tensorflow
pip3 install tensorflow==2.4.0
```

### Problemas con las Notificaciones de Telegram

1. Verifica que las credenciales de Telegram sean correctas
2. Asegúrate de que el bot tenga permisos para enviar mensajes
3. Prueba manualmente el envío de mensajes:

```python
from src.utils.enhanced_telegram_notifier import EnhancedTelegramNotifier
notifier = EnhancedTelegramNotifier()
notifier.send_message("Prueba de conexión")
```

## Optimizaciones Adicionales

### Configuración de Firewall

Para mayor seguridad, configura las reglas de firewall para permitir solo el tráfico necesario:

```bash
gcloud compute firewall-rules create allow-trading-bot \
  --direction=INGRESS \
  --priority=1000 \
  --network=default \
  --action=ALLOW \
  --rules=tcp:22,tcp:80,tcp:443 \
  --source-ranges=0.0.0.0/0 \
  --target-tags=trading-bot
```

### Programar Respaldos Automáticos

Configura respaldos automáticos de los modelos entrenados:

```bash
# Crear directorio de respaldos
mkdir -p ~/backups

# Crear script de respaldo
cat > ~/backup_models.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf ~/backups/models_$DATE.tar.gz ~/trading-bots/src/spot_bots/sol_bot_15m/models/
find ~/backups/ -type f -mtime +7 -name "models_*.tar.gz" -delete
EOF

chmod +x ~/backup_models.sh

# Programar con cron (diariamente a las 00:00)
(crontab -l 2>/dev/null; echo "0 0 * * * ~/backup_models.sh") | crontab -
```

## Actualización del Bot

Para actualizar el bot con los últimos cambios:

```bash
cd ~/trading-bots
git pull

# Reinstalar dependencias si es necesario
pip3 install -r requirements.txt

# Reiniciar el bot
cd src/spot_bots/sol_bot_15m
./start_cloud_simulation.sh
```

## Conclusión

El bot SOL ahora está configurado para ejecutarse en modo simulación de aprendizaje en Google Cloud VM. Utilizará credenciales reales de Binance para obtener datos de mercado en tiempo real, pero operará con un balance ficticio de 100 USDT. Esto permite que el bot aprenda y se optimice sin arriesgar fondos reales.

Las notificaciones de Telegram mejoradas te mantendrán informado sobre el estado del bot, las condiciones del mercado y las operaciones realizadas.

---

*Última actualización: Mayo 2025*

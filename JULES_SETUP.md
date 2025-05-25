# Guía de Configuración para Jules

## Mejores Prácticas Profesionales

### 1. Seguridad y Credenciales
- NUNCA subir credenciales reales a GitHub
- Usar siempre variables de entorno para datos sensibles
- Mantener las API keys de prueba privadas
- No compartir credenciales por correo o mensajes

### 2. Desarrollo Profesional
- Seguir los estándares de código PEP 8 para Python
- Documentar todo el código nuevo
- Crear pruebas unitarias para nuevas funciones
- Usar nombres descriptivos para variables y funciones

### 3. Control de Versiones
- Crear ramas separadas para nuevas funciones
- Hacer commits pequeños y descriptivos
- Probar código antes de hacer pull requests
- Mantener el repositorio limpio y organizado

### 4. Estructura del Código
- Seguir la estructura existente del proyecto
- Mantener separación de responsabilidades
- Usar módulos y clases apropiadamente
- Implementar manejo de errores adecuado

## Requisitos Previos
1. Cuenta en GitHub
2. Python 3.8 o superior instalado
3. Git instalado

## Pasos de Configuración

### 1. Clonar el Repositorio
```bash
git clone https://github.com/TU-USUARIO/trading-bots.git
cd trading-bots
```

### 2. Configurar Entorno de Pruebas

#### Binance Testnet
1. Crear cuenta en Binance Testnet: https://testnet.binance.vision/
2. Generar API Key y Secret de prueba
3. Guardar las credenciales de prueba

#### Telegram Bot de Prueba
1. Crear bot de prueba con @BotFather en Telegram
2. Guardar el token del bot
3. Crear un grupo de prueba y obtener el chat_id

### 3. Configurar Variables de Entorno
1. Copiar el archivo `.env.example` a `.env`
2. Llenar con las credenciales de prueba:
```
# Binance API Testnet
BINANCE_API_KEY=tu_api_key_de_prueba
BINANCE_API_SECRET=tu_api_secret_de_prueba

# Telegram Bot de Prueba
TELEGRAM_BOT_TOKEN=tu_bot_token_de_prueba
TELEGRAM_CHAT_ID=tu_chat_id_de_prueba

# Configuración de Trading
TRADE_AMOUNT_USDT=10  # Usar cantidades pequeñas para pruebas
MAX_TRADES=2
STOP_LOSS_PERCENTAGE=2
TAKE_PROFIT_PERCENTAGE=3

# Environment
ENV=development
```

### 4. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 5. Probar los Bots
```bash
# Probar bot de XRP
python src/spot_bots/xrp_bot/main.py
```

## Notas Importantes
- NUNCA uses credenciales reales de producción
- Usa SOLO la cuenta de Binance Testnet para pruebas
- Todas las operaciones deben ser de prueba
- Reporta cualquier error o mejora como Issue en GitHub

## Estructura del Proyecto
```
trading-bots/
├── src/
│   ├── spot_bots/
│   │   ├── xrp_bot/
│   │   └── sol_bot/
│   ├── futures_bots/
│   ├── utils/
│   └── config/
└── tests/
```

## Desarrollo de Nuevos Bots

### 1. Estructura de Nuevos Bots
```
trading-bots/
├── src/
│   ├── spot_bots/
│   │   ├── nuevo_bot/
│   │       ├── strategies/
│   │       ├── main.py
│   │       └── retraining.py
│   ├── utils/
│   └── config/
└── tests/
```

### 2. Manejo de Memoria y Datos
- El código en GitHub NO incluye datos de entrenamiento
- Las memorias se crean automáticamente en producción
- Usar solo datos de prueba para desarrollo
- Mantener respaldos de datos de prueba separados

### 3. Proceso de Desarrollo
1. Desarrollar usando credenciales de prueba
2. Probar exhaustivamente en ambiente de pruebas
3. Documentar comportamiento y configuración
4. Crear pull request para revisión

### 4. Integración con n8n
- Preparar endpoints para comunicación con n8n
- Documentar flujos de datos
- Mantener separación entre pruebas y producción

## Contacto
Para cualquier duda o problema, crear un Issue en GitHub o contactar al administrador del proyecto.

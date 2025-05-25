# Guía de Configuración para Jules

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

# Probar bot de SOL
python src/spot_bots/sol_bot/main.py
```

## Notas Importantes
- NUNCA uses credenciales reales de producción
- Usa SOLO la cuenta de Binance Testnet
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

## Contacto
Para cualquier duda o problema, crear un Issue en GitHub o contactar al administrador del proyecto.

# Bot SOL Adaptativo

Bot de trading para Solana (SOL) con estrategia adaptativa, take profit dinámico y gestión inteligente del capital.

## Características principales

- **Take Profit Dinámico con Stop Loss Fijo**
  - Stop loss fijo al 6% del capital
  - Take profit adaptativo según volatilidad y tendencia
  - Análisis de velas de 15 minutos

- **Gestión Inteligente del Capital**
  - Adaptación automática a cualquier capital disponible
  - Niveles de capital (micro, pequeño, medio, grande)
  - Modo aprendizaje con operaciones más pequeñas

- **Notificaciones Detalladas**
  - Informes de mercado cada 4 horas
  - Alertas de operaciones con información completa
  - Estadísticas de rendimiento

## Estructura del proyecto

```
sol_bot_15m/
├── adaptive_main.py           # Script principal del bot adaptativo
├── adaptive_tp_manager.py     # Gestor de take profit adaptativo
├── risk_manager.py            # Gestor de riesgos
├── sentiment_analyzer.py      # Analizador de sentimiento del mercado
├── performance_monitor.py     # Monitor de rendimiento
├── api_integration.py         # Integración con API
├── start_enhanced.sh          # Script de inicio mejorado
├── src/
│   ├── connectors/            # Conectores a exchanges
│   ├── models/                # Modelos de ML
│   ├── strategies/            # Estrategias de trading
│   │   ├── adaptive_strategy.py  # Estrategia adaptativa
│   │   └── technical_strategy.py # Estrategia técnica original
│   └── utils/                 # Utilidades
│       ├── market_analyzer.py        # Analizador de mercado
│       ├── capital_manager.py        # Gestor de capital
│       └── enhanced_telegram_notifier.py  # Notificador mejorado
```

## Requisitos

- Python 3.8+
- Binance API Key y Secret
- Token de bot de Telegram
- Dependencias: pandas, numpy, scikit-learn, python-binance, python-telegram-bot

## Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/yourusername/trading-bots.git
   cd trading-bots/src/spot_bots/sol_bot_15m
   ```

2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Configurar variables de entorno en un archivo `.env`:
   ```
   BINANCE_API_KEY=tu_api_key
   BINANCE_API_SECRET=tu_api_secret
   TELEGRAM_BOT_TOKEN=tu_token_de_telegram
   TELEGRAM_CHAT_ID=tu_chat_id
   ```

## Uso

### Iniciar el bot adaptativo

```bash
python adaptive_main.py --symbol SOLUSDT --interval 15m --balance 100 --risk 0.02 --stop-loss 0.06 --simulation
```

### Parámetros disponibles

- `--symbol`: Par de trading (default: SOLUSDT)
- `--interval`: Intervalo de tiempo (default: 15m)
- `--lookback`: Días de datos históricos (default: 90)
- `--balance`: Balance inicial para simulación (default: 1000)
- `--risk`: Riesgo por operación (default: 0.02)
- `--stop-loss`: Stop loss fijo (default: 0.06)
- `--simulation`: Operar en modo simulación
- `--no-ml`: No utilizar modelo de ML
- `--retrain-interval`: Intervalo de reentrenamiento en minutos (default: 1440)

### Script de inicio mejorado

```bash
./start_enhanced.sh
```

## Gestión de riesgos

El bot incluye un sistema de gestión de riesgos que:

1. Comienza en modo aprendizaje con posiciones al 50%
2. Sale del modo aprendizaje cuando alcanza un win rate del 55% con al menos 20 operaciones
3. Ajusta el tamaño de las posiciones según el capital disponible
4. Mantiene un stop loss fijo del 6% para limitar pérdidas
5. Adapta el take profit según las condiciones del mercado

## Monitoreo y notificaciones

El bot envía notificaciones detalladas a través de Telegram:

- Inicio del bot con configuración
- Actualizaciones de mercado cada 4 horas
- Entradas y salidas de operaciones con detalles completos
- Actualizaciones de parámetros

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo LICENSE para más detalles.

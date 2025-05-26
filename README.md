# Trading Bots

Este repositorio contiene bots de trading automatizados para criptomonedas, diseñados para operar en el mercado de Binance utilizando estrategias basadas en análisis técnico y aprendizaje automático.

## Bots Disponibles

- **Bot SOL (15m)**: Bot para operar Solana (SOL) con intervalo de 15 minutos y reentrenamiento de modelo ML
- **Bot XRP (30m)**: Bot para operar Ripple (XRP) con intervalo de 30 minutos

## Características Principales

- 📊 **Análisis Técnico Avanzado**: Utiliza múltiples indicadores (RSI, MACD, Bollinger Bands, etc.)
- 🤖 **Aprendizaje Automático**: Implementa modelos de ML para mejorar las señales de trading
- 📱 **Notificaciones en Tiempo Real**: Envía alertas vía Telegram sobre operaciones y estado del bot
- 💰 **Modo Simulación**: Permite probar estrategias sin arriesgar fondos reales
- 🔄 **Reentrenamiento Automático**: Actualiza el modelo ML periódicamente con datos recientes

## Requisitos

- Python 3.8 o superior
- Cuenta en Binance (API Key y Secret)
- Bot de Telegram (para notificaciones)
- Dependencias listadas en `requirements.txt`

## Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/BOTDETRAIDINGEDB/trading-bots.git
   cd trading-bots
   ```

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Configura las variables de entorno:
   - Crea un archivo `.env` en el directorio del bot basado en `.env.example`
   - Añade tus credenciales de Binance y Telegram

## Estructura del Proyecto

```
trading-bots/
├── src/                      # Código fuente principal
│   ├── spot_bots/            # Bots para mercados spot
│   │   ├── sol_bot_15m/      # Bot para SOL con intervalo de 15m
│   │   │   ├── main.py       # Punto de entrada principal
│   │   │   ├── src/          # Módulos del bot
│   │   │   │   ├── data/     # Procesamiento de datos
│   │   │   │   ├── models/   # Modelos de ML
│   │   │   │   ├── strategies/ # Estrategias de trading
│   │   │   │   └── utils/    # Utilidades (Binance, Telegram)
├── scripts/                  # Scripts de utilidad
├── tests/                    # Pruebas unitarias
└── requirements.txt          # Dependencias del proyecto
```

## Uso

### Modo Simulación (Recomendado para Pruebas)

```bash
python src/spot_bots/sol_bot_15m/main.py --use-ml --retrain-interval 15 --interval 15m --symbol SOLUSDT --simulation --balance 1000
```

### Modo Real (Usar con Precaución)

```bash
python src/spot_bots/sol_bot_15m/main.py --use-ml --retrain-interval 15 --interval 15m --symbol SOLUSDT
```

### Parámetros Disponibles

- `--interval`: Intervalo de tiempo para las velas (ej. 15m, 1h, 4h)
- `--symbol`: Par de trading (ej. SOLUSDT, BTCUSDT)
- `--simulation`: Ejecutar en modo simulación sin operar con fondos reales
- `--balance`: Balance inicial para simulación (default: 1000 USDT)
- `--risk`: Riesgo por operación, como porcentaje del balance (default: 0.02 = 2%)
- `--use-ml`: Activar el uso del modelo de aprendizaje automático
- `--retrain-interval`: Intervalo para reentrenar el modelo en minutos

## Estrategia de Trading

El bot utiliza una combinación de análisis técnico y aprendizaje automático:

1. **Indicadores Técnicos**:
   - RSI (Relative Strength Index)
   - MACD (Moving Average Convergence Divergence)
   - Bollinger Bands
   - EMA (Exponential Moving Average)

2. **Modelo de ML**:
   - Algoritmo: Random Forest Classifier
   - Features: Indicadores técnicos y patrones de precio
   - Target: Dirección del precio en el futuro cercano
   - Reentrenamiento: Periódico con datos recientes

3. **Gestión de Riesgo**:
   - Stop Loss dinámico basado en ATR
   - Take Profit basado en niveles de Fibonacci
   - Tamaño de posición proporcional al riesgo configurado

## Scripts de Utilidad

- `scripts/update_vm.sh`: Actualiza el bot en la máquina virtual
- `scripts/debug_bot.sh`: Verifica la configuración y diagnostica problemas
- `scripts/check_telegram.py`: Verifica la configuración de notificaciones
- `scripts/start_bot_simulation.sh`: Inicia el bot en modo simulación

## Advertencia

El trading de criptomonedas implica riesgos significativos. Este bot es una herramienta experimental y no garantiza beneficios. Úsalo bajo tu propia responsabilidad y siempre comienza con el modo de simulación.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, sigue estas pautas:
1. Haz fork del repositorio
2. Crea una rama para tu feature (`git checkout -b feature/amazing-feature`)
3. Haz commit de tus cambios (`git commit -m 'Add amazing feature'`)
4. Push a la rama (`git push origin feature/amazing-feature`)
5. Abre un Pull Request

# Trading Bots

Este repositorio contiene bots de trading para diferentes criptomonedas, incluyendo:

- Bot SOL con intervalo de 15 minutos y reentrenamiento de modelo ML
- Bot XRP con intervalo de 30 minutos

## Estructura del Proyecto

```
trading-bots/
├── src/
│   ├── spot_bots/
│   │   ├── sol_bot_15m/
│   │   │   ├── main.py
│   │   │   ├── src/
│   │   │   │   ├── data/
│   │   │   │   ├── models/
│   │   │   │   ├── strategies/
│   │   │   │   └── utils/
│   │   ├── xrp_bot/
│   │   │   ├── ...
├── scripts/
│   ├── update_vm.sh
│   ├── debug_bot.sh
│   ├── convert_to_15m.sh
│   └── fix_all_references.py
```

## Configuración

1. Crea un archivo `.env` en el directorio del bot basado en `.env.example`
2. Ejecuta el bot con:
   ```
   python main.py --use-ml --retrain-interval 15 --interval 15m --symbol SOLUSDT
   ```

## Scripts de Utilidad

- `update_vm.sh`: Actualiza el bot en la máquina virtual
- `debug_bot.sh`: Verifica la configuración del bot y diagnostica problemas
- `convert_to_15m.sh`: Convierte el bot de 20m a 15m
- `fix_all_references.py`: Corrige todas las referencias de 20m a 15m

# Instrucciones para Reiniciar el Bot SOL

Este documento contiene instrucciones detalladas sobre cómo reiniciar correctamente el bot SOL y un resumen de los cambios realizados para hacerlo más activo en operaciones de trading.

## Método Correcto para Reiniciar el Bot

El script más completo y seguro para reiniciar el bot es `cleanup_bot_sessions.sh`, que realiza las siguientes acciones:

```bash
cd /home/edisonbautistaruiz2025/new-trading-bots/src/spot_bots/sol_bot_15m/
./cleanup_bot_sessions.sh
```

### Lo que hace este script:

1. **Detiene todas las sesiones del bot principal y adaptativo**
2. **Espera 5 segundos** para asegurar que todas las sesiones se hayan cerrado
3. **Verifica que no queden sesiones activas**
4. **Elimina archivos de control** que podrían impedir notificaciones
5. **Verifica las credenciales configuradas**
6. **Reinicia el bot limpiamente** en modo simulación en la nube

## Métodos Alternativos

Si necesitas más control sobre el proceso, puedes usar estos comandos separados:

### Detener el bot:
```bash
cd /home/edisonbautistaruiz2025/new-trading-bots/src/spot_bots/sol_bot_15m/
./stop.sh
```

### Iniciar el bot en modo real:
```bash
./start_bot.sh
```

### Iniciar el bot en modo simulación:
```bash
./start_simulation_learning.sh
```

## Cambios Realizados para Hacer el Bot Más Activo

Se han realizado las siguientes modificaciones para hacer que el bot sea más activo y sensible a oportunidades de trading:

### 1. En `technical_strategy.py`:
- Modificado el método `should_enter_trade` para que entre en operaciones si cualquiera de las señales (ML o técnica) es positiva
- Cambiada la lógica para considerar señales neutrales como potenciales entradas

### 2. En `ml_model.py`:
- Reducido `n_estimators` de 200 a 150 para evitar sobreajuste
- Reducido `max_depth` de 15 a 10 para generalizar mejor
- Reducidos `min_samples_split` y `min_samples_leaf` para hacer el modelo más sensible
- Cambiado `class_weight` de 'balanced' a {0: 1, 1: 2} para dar más peso a señales de compra
- Cambiado `criterion` de 'gini' a 'entropy' para mayor sensibilidad

### 3. Nuevo archivo `bot_params.json`:
- Aumento del `risk_per_trade` de 0.02 a 0.03
- Reducción del `stop_loss_pct` de 0.06 a 0.05
- Reducción del `take_profit_pct` de 0.04 a 0.03
- Reducción del `trailing_percent` de 0.015 a 0.01

### 4. En `feature_config.json`:
- Reducidos los umbrales de clasificación de 0.005 a 0.003 para ser más sensible a movimientos de precio
- Reducido el horizonte de predicción de 3 a 2 para reaccionar más rápido

## Monitoreo del Bot

Después de reiniciar el bot, es importante monitorear su comportamiento:

```bash
# Ver los últimos logs del bot
tail -f sol_bot_15min.log

# Ver el estado actual del bot
cat sol_bot_15min_state.json
```

## Notas Importantes

- Siempre verifica que el bot se haya reiniciado correctamente revisando los logs
- Monitorea el comportamiento del bot durante 24-48 horas después de los cambios
- Si el bot sigue siendo demasiado conservador, se pueden ajustar aún más los parámetros

---

*Última actualización: 28 de mayo de 2025*

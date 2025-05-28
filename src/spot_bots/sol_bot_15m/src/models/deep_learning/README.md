# Módulo LSTM para Trading Bot

Este módulo implementa modelos LSTM/GRU para predicción de mercados financieros, optimizado para funcionar en Google Cloud VM.

## Estructura de Archivos

- `lstm_model.py`: Clase principal DeepTimeSeriesModel
- `lstm_model_part2.py`: Implementación de arquitecturas de modelos
- `lstm_model_part3.py`: Métodos de entrenamiento y evaluación
- `cloud_optimizer.py`: Utilidades para optimizar el rendimiento en Google Cloud VM
- `cloud_init.py`: Script de inicialización para Google Cloud VM

## Configuración para Google Cloud VM

Para ejecutar el bot en una máquina virtual de Google Cloud:

1. Ejecute el script de inicialización:
   ```
   python cloud_init.py
   ```

2. Asegúrese de que las siguientes variables de entorno estén configuradas:
   - `CLOUD_ENV=true`: Activa optimizaciones para entorno cloud
   - `MEMORY_LIMIT_MB=2048`: Límite de memoria en MB (ajustar según la VM)
   - `TF_DETERMINISTIC=true`: Activa modo determinista para reproducibilidad
   - `USE_MULTIPROCESSING=false`: Desactiva multiprocesamiento en entornos con recursos limitados

## Dependencias

- TensorFlow >= 2.4.0
- NumPy >= 1.19.0
- psutil >= 5.8.0 (opcional, para monitoreo de memoria)

## Optimizaciones para Google Cloud VM

Este módulo incluye las siguientes optimizaciones para entornos cloud:

- Manejo eficiente de memoria para evitar errores OOM
- Reintentos automáticos para operaciones críticas
- Monitoreo de recursos
- Configuración optimizada de TensorFlow
- Manejo robusto de errores

## Solución de Problemas

Si experimenta errores en Google Cloud VM:

1. Verifique que las variables de entorno estén configuradas correctamente
2. Aumente el valor de `MEMORY_LIMIT_MB` si hay errores de memoria
3. Ejecute `python cloud_init.py` para reiniciar la configuración
4. Verifique los logs para identificar errores específicos

## Notas Importantes

- Este módulo ha sido optimizado específicamente para funcionar en Google Cloud VM
- Las optimizaciones se activan automáticamente cuando `CLOUD_ENV=true`
- Para entornos de desarrollo local, no es necesario configurar estas variables

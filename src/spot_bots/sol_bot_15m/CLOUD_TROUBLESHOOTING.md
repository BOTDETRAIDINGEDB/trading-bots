# Guía de Solución de Problemas para Google Cloud

Esta guía proporciona soluciones a problemas comunes que pueden surgir durante el despliegue y operación del bot SOL con aprendizaje profundo en Google Cloud Platform.

## Problemas de Despliegue

### Error: "Failed to deploy to App Engine"

**Síntomas:**
- El script `deploy_to_cloud.py` falla con un error relacionado con el despliegue a App Engine.

**Soluciones:**
1. Verificar permisos:
   ```bash
   gcloud projects get-iam-policy TU-PROYECTO-ID
   ```
   Asegúrate de que tu cuenta tenga el rol `roles/appengine.appAdmin`.

2. Verificar cuota:
   - Revisa la [consola de cuotas](https://console.cloud.google.com/iam-admin/quotas) para asegurarte de que no has excedido los límites.

3. Verificar configuración:
   - Asegúrate de que el archivo `cloud_config.yaml` esté correctamente formateado.
   - Ejecuta `python check_cloud_compatibility.py` para verificar la configuración.

### Error: "ERROR: (gcloud.app.deploy) Error Response: [9] Cloud build has status: FAILURE"

**Síntomas:**
- El despliegue falla durante la fase de construcción.

**Soluciones:**
1. Verificar logs de construcción:
   ```bash
   gcloud builds list --filter="source.repoSource.repoName:appengine"
   gcloud builds log [BUILD_ID]
   ```

2. Problemas comunes:
   - **Dependencias faltantes**: Asegúrate de que `requirements.txt` incluya todas las dependencias.
   - **Versiones incompatibles**: Verifica que las versiones de las dependencias sean compatibles.
   - **Timeout**: Si la construcción toma demasiado tiempo, aumenta el timeout en la configuración.

## Problemas de Almacenamiento

### Error: "Access Denied" al acceder a Cloud Storage

**Síntomas:**
- El bot no puede leer o escribir en el bucket de Cloud Storage.
- Logs muestran errores de permisos.

**Soluciones:**
1. Verificar permisos de la cuenta de servicio:
   ```bash
   gcloud projects add-iam-policy-binding TU-PROYECTO-ID \
       --member="serviceAccount:TU-PROYECTO-ID@appspot.gserviceaccount.com" \
       --role="roles/storage.admin"
   ```

2. Verificar existencia del bucket:
   ```bash
   gsutil ls -b gs://TU-BUCKET
   ```
   Si no existe, créalo:
   ```bash
   gsutil mb -p TU-PROYECTO-ID -l us-central1 gs://TU-BUCKET
   ```

3. Verificar configuración:
   - Asegúrate de que `STORAGE_BUCKET` en `cloud_config.yaml` coincida con el nombre del bucket.

### Error: "No such file or directory" al cargar/guardar modelos

**Síntomas:**
- El bot no puede encontrar archivos de modelo o datos.

**Soluciones:**
1. Verificar estructura de directorios:
   ```bash
   gsutil ls -r gs://TU-BUCKET/
   ```
   Asegúrate de que existan las carpetas `models/` y `data/`.

2. Verificar script de inicio:
   - Asegúrate de que el script `postStart` en `cloud_config.yaml` esté copiando correctamente los archivos.

3. Solución manual:
   ```bash
   # Crear estructura de directorios
   gsutil mkdir -p gs://TU-BUCKET/models/
   gsutil mkdir -p gs://TU-BUCKET/data/
   
   # Subir archivos locales
   gsutil -m cp -r ./models/* gs://TU-BUCKET/models/
   gsutil -m cp -r ./data/* gs://TU-BUCKET/data/
   ```

## Problemas de Memoria

### Error: "OOM" (Out of Memory)

**Síntomas:**
- El bot se detiene inesperadamente.
- Logs muestran errores de memoria insuficiente.

**Soluciones:**
1. Aumentar recursos en `cloud_config.yaml`:
   ```yaml
   resources:
     cpu: 2
     memory_gb: 8  # Aumentar este valor
   ```

2. Optimizar uso de memoria:
   - Implementa la liberación de memoria después de cada predicción.
   - Utiliza el módulo `tensorflow_cloud_config.py` para configurar TensorFlow.
   - Ejecuta `python test_cloud_integration.py --test memory` para identificar fugas de memoria.

3. Reducir tamaño del modelo:
   - Considera usar un modelo más pequeño o técnicas de cuantización.
   - Implementa carga parcial del modelo si es posible.

### Advertencia: "High memory usage"

**Síntomas:**
- El bot funciona pero los logs muestran advertencias de uso alto de memoria.

**Soluciones:**
1. Monitorear uso de memoria:
   ```bash
   python cloud_monitor.py --project-id TU-PROYECTO-ID --get-metrics
   ```

2. Implementar limpieza periódica:
   - Asegúrate de llamar a `cleanup_tensorflow_memory()` después de cada predicción.
   - Considera implementar un recolector de basura periódico.

## Problemas de TensorFlow

### Error: "Could not create cudnn handle"

**Síntomas:**
- El bot falla al inicializar TensorFlow con GPU.

**Soluciones:**
1. Verificar disponibilidad de GPU:
   - Asegúrate de que estás usando una instancia con GPU habilitada.
   - Verifica que las bibliotecas CUDA estén instaladas.

2. Configurar TensorFlow para CPU:
   ```bash
   # Añadir a variables de entorno
   export CUDA_VISIBLE_DEVICES=-1
   ```
   O en `cloud_config.yaml`:
   ```yaml
   env_variables:
     CUDA_VISIBLE_DEVICES: -1
   ```

3. Optimizar configuración:
   - Utiliza el módulo `tensorflow_cloud_config.py` para configurar TensorFlow.

### Error: "Failed to load model"

**Síntomas:**
- El bot no puede cargar el modelo LSTM.

**Soluciones:**
1. Verificar formato del modelo:
   - Asegúrate de que el modelo esté guardado en formato compatible (SavedModel o HDF5).
   - Verifica la versión de TensorFlow usada para guardar y cargar.

2. Verificar estructura del modelo:
   ```bash
   # Para modelos SavedModel
   saved_model_cli show --dir ./models/lstm_model --all
   ```

3. Solución manual:
   - Reconstruye el modelo y guárdalo nuevamente.
   - Asegúrate de guardar también los metadatos y configuración.

## Problemas de Logging

### Error: "Failed to send logs to Cloud Logging"

**Síntomas:**
- El bot funciona pero no se ven logs en Cloud Logging.

**Soluciones:**
1. Verificar permisos:
   ```bash
   gcloud projects add-iam-policy-binding TU-PROYECTO-ID \
       --member="serviceAccount:TU-PROYECTO-ID@appspot.gserviceaccount.com" \
       --role="roles/logging.logWriter"
   ```

2. Verificar configuración:
   - Asegúrate de que la configuración de logging en `cloud_config.yaml` sea correcta.

3. Implementar logging local como respaldo:
   - Configura el logging para escribir también a archivos locales.

## Problemas de Rendimiento

### Problema: "Slow prediction times"

**Síntomas:**
- Las predicciones toman mucho más tiempo que en entorno local.

**Soluciones:**
1. Optimizar TensorFlow:
   - Utiliza precisión mixta (`mixed_float16`).
   - Habilita optimizaciones XLA.
   - Configura correctamente el número de hilos.

2. Monitorear rendimiento:
   ```bash
   python cloud_monitor.py --project-id TU-PROYECTO-ID --full-report
   ```

3. Ajustar recursos:
   - Aumenta CPU y memoria en `cloud_config.yaml`.
   - Considera usar una instancia con GPU.

### Problema: "High latency"

**Síntomas:**
- Alta latencia en respuestas del bot.

**Soluciones:**
1. Optimizar configuración de App Engine:
   ```yaml
   automatic_scaling:
     min_instances: 1  # Mantener al menos una instancia activa
     target_cpu_utilization: 0.6
   ```

2. Implementar caché:
   - Almacena en caché resultados frecuentes.
   - Implementa predicciones por lotes.

## Problemas de Secretos

### Error: "Failed to access secret"

**Síntomas:**
- El bot no puede acceder a secretos en Secret Manager.

**Soluciones:**
1. Verificar permisos:
   ```bash
   gcloud projects add-iam-policy-binding TU-PROYECTO-ID \
       --member="serviceAccount:TU-PROYECTO-ID@appspot.gserviceaccount.com" \
       --role="roles/secretmanager.secretAccessor"
   ```

2. Verificar existencia del secreto:
   ```bash
   gcloud secrets list --project TU-PROYECTO-ID
   ```

3. Verificar configuración:
   - Asegúrate de que los nombres de los secretos en `cloud_config.yaml` coincidan con los creados.

## Diagnóstico General

### Herramienta de Verificación de Compatibilidad

Ejecuta la herramienta de verificación de compatibilidad para diagnosticar problemas:

```bash
python check_cloud_compatibility.py
```

Esta herramienta verificará:
- Dependencias instaladas
- Configuración de TensorFlow
- Compatibilidad con Google Cloud Storage
- Configuración de archivos

### Herramienta de Monitoreo

Utiliza la herramienta de monitoreo para diagnosticar problemas en tiempo real:

```bash
python cloud_monitor.py --project-id TU-PROYECTO-ID --full-report
```

### Pruebas de Integración

Ejecuta pruebas de integración para identificar problemas específicos:

```bash
python test_cloud_integration.py --test all
```

## Contacto y Soporte

Si encuentras problemas que no están cubiertos en esta guía:

1. Consulta la [documentación oficial de Google Cloud](https://cloud.google.com/docs)
2. Revisa los [foros de Google Cloud](https://cloud.google.com/community)
3. Abre un issue en el repositorio del proyecto

## Recursos Adicionales

- [Optimización de TensorFlow para producción](https://www.tensorflow.org/guide/performance/overview)
- [Mejores prácticas para App Engine](https://cloud.google.com/appengine/docs/standard/python3/building-app/writing-web-service)
- [Guía de Secret Manager](https://cloud.google.com/secret-manager/docs/best-practices)
- [Guía de Cloud Storage](https://cloud.google.com/storage/docs/best-practices)

#!/usr/bin/env python3
"""
Módulo que contiene la parte 3 de la implementación de DeepTimeSeriesModel.
Contiene métodos para entrenamiento y evaluación del modelo.
"""

import os
import sys
import logging
import traceback
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import json
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import time
import gc
import tempfile

# Configurar logging
logger = logging.getLogger(__name__)

def train(self, X_train: np.ndarray, y_train: np.ndarray, 
          X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
          class_weights: Optional[Dict] = None) -> Dict:
    """
    Entrena el modelo con los datos proporcionados.
    
    Args:
        X_train: Datos de entrenamiento (secuencias)
        y_train: Etiquetas de entrenamiento
        X_val: Datos de validación (opcional)
        y_val: Etiquetas de validación (opcional)
        class_weights: Pesos por clase para datos desbalanceados
        
    Returns:
        Diccionario con métricas de entrenamiento
    """
    try:
        # Verificar que hay datos suficientes
        if len(X_train) < 100:
            logger.warning("Datos insuficientes para entrenar el modelo")
            return {}
        
        # Configurar callbacks con manejo de errores
        try:
            # Obtener configuración con valores por defecto seguros
            config = self.config.get(self.model_type, {}) if hasattr(self, 'config') and self.config else {}
            patience = config.get("patience", 15)
            epochs = min(config.get("epochs", 100), 500)  # Limitar epochs para evitar entrenamientos excesivos
            batch_size = min(config.get("batch_size", 64), 256)  # Limitar batch size para evitar OOM
            
            # Crear directorio para checkpoints con manejo de errores
            checkpoint_dir = os.path.dirname(self.model_path)
            try:
                # Usar función segura para crear directorios
                if 'ensure_directory_exists' in globals():
                    ensure_directory_exists(checkpoint_dir)
                else:
                    os.makedirs(checkpoint_dir, exist_ok=True)
            except Exception as e:
                logger.warning(f"Error al crear directorio para checkpoints: {str(e)}. Usando directorio temporal.")
                import tempfile
                checkpoint_dir = tempfile.mkdtemp(prefix="model_checkpoints_")
                self.model_path = os.path.join(checkpoint_dir, os.path.basename(self.model_path))
            
            # Usar formato SavedModel para mayor compatibilidad con TF 2.x
            checkpoint_path = os.path.join(checkpoint_dir, f"{os.path.basename(self.model_path)}_{self.model_type}")
            
            # Configurar callbacks básicos
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience//3, min_lr=1e-6),
            ]
            
            # Añadir ModelCheckpoint con manejo de errores
            try:
                callbacks.append(ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1,
                    save_format='tf'  # Usar formato SavedModel en lugar de HDF5 para mejor compatibilidad
                ))
            except Exception as e:
                logger.warning(f"Error al configurar ModelCheckpoint: {str(e)}. Usando configuración básica.")
                # Intentar con formato HDF5 como fallback
                try:
                    callbacks.append(ModelCheckpoint(
                        filepath=f"{checkpoint_path}.h5",
                        monitor='val_loss',
                        save_best_only=True,
                        verbose=1
                    ))
                except Exception as e2:
                    logger.warning(f"Error al configurar ModelCheckpoint alternativo: {str(e2)}. Continuando sin checkpoint.")
            
            # Añadir TensorBoard para monitoreo en cloud si estamos en entorno cloud
            if os.environ.get('CLOUD_ENV') == 'true':
                try:
                    import tensorflow as tf
                    log_dir = os.path.join(checkpoint_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
                    if 'ensure_directory_exists' in globals():
                        ensure_directory_exists(log_dir)
                    else:
                        os.makedirs(log_dir, exist_ok=True)
                    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))
                    logger.info(f"TensorBoard configurado en {log_dir}")
                except Exception as e:
                    logger.warning(f"Error al configurar TensorBoard: {str(e)}. Continuando sin TensorBoard.")
                    
            # Añadir callback para monitoreo de memoria en entornos cloud
            if os.environ.get('CLOUD_ENV') == 'true':
                try:
                    class MemoryMonitorCallback(tf.keras.callbacks.Callback):
                        def on_epoch_end(self, epoch, logs=None):
                            try:
                                import psutil
                                process = psutil.Process(os.getpid())
                                memory_info = process.memory_info()
                                memory_mb = memory_info.rss / (1024 * 1024)
                                logger.info(f"Epoch {epoch+1} - Uso de memoria: {memory_mb:.2f} MB")
                                # Si el uso de memoria es excesivo, detener entrenamiento
                                memory_limit = float(os.environ.get('MEMORY_LIMIT_MB', '0'))
                                if memory_limit > 0 and memory_mb > memory_limit * 0.9:
                                    logger.warning(f"Uso de memoria excesivo ({memory_mb:.2f} MB). Deteniendo entrenamiento.")
                                    self.model.stop_training = True
                            except Exception as e:
                                logger.warning(f"Error en monitoreo de memoria: {str(e)}")
                    
                    callbacks.append(MemoryMonitorCallback())
                    logger.info("Monitor de memoria configurado")
                except Exception as e:
                    logger.warning(f"Error al configurar monitor de memoria: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error al configurar callbacks: {str(e)}")
            # Usar callbacks mínimos en caso de error
            callbacks = [EarlyStopping(monitor='val_loss', patience=10)]
        
        # Preparar datos de validación
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        # Si no hay datos de validación, usar split de entrenamiento
        if validation_data is None:
            validation_split = 0.2
        else:
            validation_split = 0.0
        
        # Configurar TensorFlow para entorno cloud si es necesario
        if os.environ.get('CLOUD_ENV') == 'true':
            try:
                import tensorflow as tf
                # Limitar uso de memoria
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    logger.info(f"Detectadas {len(gpus)} GPUs")
                    for gpu in gpus:
                        try:
                            tf.config.experimental.set_memory_growth(gpu, True)
                            logger.info(f"Configurado memory growth para GPU: {gpu}")
                        except Exception as e:
                            logger.warning(f"Error al configurar memory growth para GPU: {str(e)}")
                
                # Configurar para determinismo si se requiere
                if os.environ.get('TF_DETERMINISTIC', 'false').lower() == 'true':
                    logger.info("Configurando TensorFlow para ejecución determinista")
                    tf.random.set_seed(42)
                    np.random.seed(42)
                    tf.config.threading.set_inter_op_parallelism_threads(1)
                    tf.config.threading.set_intra_op_parallelism_threads(1)
            except Exception as e:
                logger.warning(f"Error al configurar TensorFlow para entorno cloud: {str(e)}")
        
        # Monitorear memoria antes de entrenar
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / (1024 * 1024)
            logger.info(f"Memoria antes de entrenar: {memory_before:.2f} MB")
        except Exception as e:
            logger.warning(f"Error al monitorear memoria: {str(e)}")
        
        # Entrenar modelo con manejo de errores optimizado para Google Cloud VM
        start_time = time.time()
        try:
            # Configurar opciones de entrenamiento según el entorno
            is_cloud = os.environ.get('CLOUD_ENV', 'false').lower() == 'true'
            use_multiprocessing = os.environ.get('USE_MULTIPROCESSING', 'false').lower() == 'true'
            
            # En Google Cloud VM, limitar el número de workers y desactivar multiprocesamiento si está configurado
            if is_cloud:
                workers = min(os.cpu_count() or 1, 2)  # Limitar a máximo 2 workers en cloud
                logger.info(f"Entrenando en entorno cloud con {workers} workers y multiprocesamiento {'activado' if use_multiprocessing else 'desactivado'}")
            else:
                workers = os.cpu_count() or 1
            
            # Usar un bloque try-except para capturar errores durante el entrenamiento
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                validation_split=validation_split,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=2,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                max_queue_size=10
            )
            
            # Registrar tiempo de entrenamiento
            training_time = time.time() - start_time
            logger.info(f"Entrenamiento completado en {training_time:.2f} segundos")
            
            # Monitorear memoria después de entrenar
            try:
                memory_after = process.memory_info().rss / (1024 * 1024)
                logger.info(f"Memoria después de entrenar: {memory_after:.2f} MB (Incremento: {memory_after - memory_before:.2f} MB)")
            except Exception:
                pass
                
        except Exception as e:
            logger.error(f"Error durante el entrenamiento: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            
            # Intentar liberar memoria de manera más robusta
            try:
                # Primero intentar usar el optimizador de cloud si está disponible
                try:
                    from cloud_optimizer import cleanup_memory
                    cleanup_memory()
                    logger.info("Memoria liberada usando cloud_optimizer")
                except ImportError:
                    # Si no está disponible, usar método estándar
                    import gc
                    gc.collect()
                    if 'tf' in globals() or 'tf' in locals():
                        tf.keras.backend.clear_session()
                    logger.info("Memoria liberada usando método estándar")
            except Exception as e:
                logger.warning(f"Error al liberar memoria: {str(e)}")
                # Intentar último recurso
                try:
                    import gc
                    gc.collect()
                except Exception:
                    pass
            
            return {}
        
        # Guardar métricas de entrenamiento
        self.train_history = {
            'loss': history.history.get('loss', []),
            'val_loss': history.history.get('val_loss', []),
            'accuracy': history.history.get('accuracy', []),
            'val_accuracy': history.history.get('val_accuracy', []),
            'epochs': len(history.history.get('loss', [])),
            'training_time': training_time,
            'date': datetime.now().isoformat()
        }
        
        self.last_trained = datetime.now().isoformat()
        
        # Guardar modelo entrenado
        try:
            # Guardar en formato SavedModel (TF 2.x)
            savedmodel_path = f"{self.model_path}_{self.model_type}_savedmodel"
            self.model.save(savedmodel_path, save_format='tf')
            logger.info(f"Modelo guardado en formato SavedModel: {savedmodel_path}")
            
            # También guardar en formato HDF5 para compatibilidad
            h5_path = f"{self.model_path}_{self.model_type}.h5"
            self.model.save(h5_path, save_format='h5')
            logger.info(f"Modelo guardado en formato HDF5: {h5_path}")
            
            # Guardar metadatos
            metadata_path = f"{self.model_path}_{self.model_type}_metadata.json"
            metadata = {
                'train_history': self.train_history,
                'model_type': self.model_type,
                'sequence_length': self.sequence_length,
                'num_features': self.num_features,
                'num_classes': self.num_classes,
                'config': self.config
            }
            
            # Usar función segura para guardar JSON si está disponible
            if 'safe_json_dump' in globals():
                safe_json_dump(metadata, metadata_path)
            else:
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4, default=lambda o: o.isoformat() if isinstance(o, datetime) else None)
            
            logger.info(f"Metadatos guardados en: {metadata_path}")
            
        except Exception as e:
            logger.error(f"Error al guardar modelo: {str(e)}")
        
        # Liberar memoria después de entrenar
        try:
            import gc
            gc.collect()
            if 'tf' in globals() or 'tf' in locals():
                tf.keras.backend.clear_session()
        except Exception:
            pass
        
        return self.train_history
        
    except Exception as e:
        logger.error(f"Error general en entrenamiento: {str(e)}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return {}

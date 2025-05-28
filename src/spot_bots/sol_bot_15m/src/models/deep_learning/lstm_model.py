#!/usr/bin/env python3
"""
Módulo de modelos LSTM/GRU para predicción de mercados financieros.
Implementa arquitecturas avanzadas de redes neuronales recurrentes.

Este módulo es el punto de entrada principal para usar modelos LSTM/GRU.
"""

# Importaciones necesarias
import os
import sys
import json
import logging
import traceback
import numpy as np
from datetime import datetime
import time
import gc
import tempfile

# Configurar logging


# Configurar logging
logger = logging.getLogger(__name__)

# Intentar importar TensorFlow con manejo de errores
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.layers import Bidirectional, TimeDistributed, Attention, Concatenate
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l1_l2
    from typing import Dict, List, Tuple, Optional, Union, Any
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow no está disponible. Algunas funcionalidades estarán limitadas.")

# Configurar logging


# Implementación principal para mantener compatibilidad con código existente
# La implementación completa se encuentra en lstm_model_unified.py

# Importar utilidades propias
try:
    from utils.cloud_utils import ensure_directory_exists, safe_json_dump, safe_json_load, json_serializable
except ImportError:
    # Definir funciones básicas si no se pueden importar las utilidades
    def ensure_directory_exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        return True
        
    def safe_json_dump(data, file_path, indent=4):
        try:
            directory = os.path.dirname(file_path)
            os.makedirs(directory, exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=indent)
            return True
        except Exception as e:
            print(f"Error al guardar JSON: {str(e)}")
            return False
            
    def safe_json_load(file_path, default=None):
        try:
            if not os.path.exists(file_path):
                return default
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error al cargar JSON: {str(e)}")
            return default
            
    def json_serializable(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        raise TypeError(f"Tipo no serializable: {type(obj)}")

# Configurar logging


# Los métodos de los archivos auxiliares se importarán después de definir la clase DeepTimeSeriesModel
# Estos métodos serán asignados dinámicamente a la clase en __init__.py:
# - _build_bilstm_model de lstm_model_part2.py
# - _build_attention_model de lstm_model_part2.py
# - train de lstm_model_part3.py

# Configurar TensorFlow para ser compatible con entornos con y sin GPU
try:
    # Configurar para usar menos memoria y crecer según sea necesario
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU configurada para crecimiento de memoria: {gpu}")
            except RuntimeError as e:
                logger.warning(f"Error al configurar GPU {gpu}: {str(e)}")
        
        # Limitar memoria GPU si está en entorno cloud
        if os.environ.get('CLOUD_ENV') == 'true':
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
                )
                logger.info("GPU limitada a 4GB para entorno cloud")
            except Exception as e:
                logger.warning(f"No se pudo limitar memoria GPU: {str(e)}")
                
        logger.info(f"GPU disponible para entrenamiento: {len(gpus)} dispositivos")
    else:
        logger.info("No se detectó GPU, usando CPU para entrenamiento")
        
    # Configurar para usar operaciones deterministas (reproducibilidad)
    if os.environ.get('TF_DETERMINISTIC') == 'true':
        tf.config.experimental.enable_op_determinism()
        logger.info("TensorFlow configurado para operaciones deterministas")
        
except Exception as e:
    logger.warning(f"Error al configurar TensorFlow: {str(e)}")
    logger.info("Continuando con configuración por defecto de TensorFlow")
    
# Configurar para usar menos memoria en CPU
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(4)

class DeepTimeSeriesModel:
    """
    Modelo de series temporales basado en redes neuronales profundas LSTM/GRU.
    Soporta arquitecturas avanzadas y entrenamiento optimizado.
    """
    
    def __init__(self, 
                 model_type: str = 'lstm',
                 sequence_length: int = 60,
                 num_features: int = 20,
                 num_classes: int = 3,
                 model_path: str = 'models/deep_learning/lstm_model',
                 config_path: str = 'config/lstm_config.json'):
        """
        Inicializa el modelo de series temporales profundo.
        
        Args:
            model_type: Tipo de modelo ('lstm', 'gru', 'bilstm', 'attention')
            sequence_length: Longitud de secuencia para entrada
            num_features: Número de características de entrada
            num_classes: Número de clases para clasificación (3 para compra/mantener/venta)
            model_path: Ruta base donde guardar/cargar el modelo
            config_path: Ruta al archivo de configuración del modelo
        """
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.num_classes = num_classes
        self.model_path = model_path
        self.config_path = config_path
        
        # Crear directorios si no existen
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Cargar configuración
        self.config = self._load_config()
        
        # Inicializar modelo
        self.model = self._build_model()
        
        # Métricas de entrenamiento
        self.train_history = None
        self.last_trained = None
        
        logger.info(f"Modelo {model_type} inicializado con {num_features} características y secuencia de {sequence_length}")
    
    def _load_config(self) -> Dict:
        """
        Carga la configuración del modelo desde archivo JSON.
        
        Returns:
            Diccionario con configuración del modelo
        """
        default_config = {
            "lstm": {
                "units": [128, 64, 32],
                "dropout": 0.3,
                "recurrent_dropout": 0.3,
                "learning_rate": 0.001,
                "batch_size": 64,
                "epochs": 100,
                "patience": 15,
                "l1_reg": 0.0001,
                "l2_reg": 0.0001
            },
            "gru": {
                "units": [128, 64, 32],
                "dropout": 0.3,
                "recurrent_dropout": 0.3,
                "learning_rate": 0.001,
                "batch_size": 64,
                "epochs": 100,
                "patience": 15,
                "l1_reg": 0.0001,
                "l2_reg": 0.0001
            },
            "bilstm": {
                "units": [128, 64],
                "dropout": 0.3,
                "recurrent_dropout": 0.3,
                "learning_rate": 0.001,
                "batch_size": 64,
                "epochs": 100,
                "patience": 15,
                "l1_reg": 0.0001,
                "l2_reg": 0.0001
            },
            "attention": {
                "lstm_units": 128,
                "attention_units": 64,
                "dropout": 0.3,
                "learning_rate": 0.001,
                "batch_size": 64,
                "epochs": 100,
                "patience": 15,
                "l1_reg": 0.0001,
                "l2_reg": 0.0001
            }
        }
        
        # Verificar si hay configuración en variables de entorno
        env_config = {}
        for key in os.environ:
            if key.startswith('MODEL_'):
                # Extraer nombre de parámetro (MODEL_LSTM_UNITS -> lstm_units)
                param_parts = key[6:].lower().split('_')
                if len(param_parts) >= 2:
                    model_type = param_parts[0]
                    param_name = '_'.join(param_parts[1:])
                    
                    # Crear estructura si no existe
                    if model_type not in env_config:
                        env_config[model_type] = {}
                    
                    # Convertir valor a tipo adecuado
                    value = os.environ[key]
                    try:
                        # Intentar convertir a número si es posible
                        if '.' in value:
                            env_config[model_type][param_name] = float(value)
                        else:
                            env_config[model_type][param_name] = int(value)
                    except ValueError:
                        # Si no es número, mantener como string
                        env_config[model_type][param_name] = value
        
        # Si hay configuración en variables de entorno, actualizar valores por defecto
        if env_config:
            logger.info(f"Usando configuración de variables de entorno: {env_config}")
            for model_type, params in env_config.items():
                if model_type in default_config:
                    default_config[model_type].update(params)
                else:
                    default_config[model_type] = params
        
        # Intentar cargar desde archivo
        try:
            # Usar función segura para cargar JSON
            config = safe_json_load(self.config_path, default=None)
            
            if config:
                logger.info(f"Configuración cargada desde {self.config_path}")
                return config
            else:
                logger.warning(f"Archivo de configuración no encontrado o inválido: {self.config_path}")
                
                # Asegurar que el directorio existe
                ensure_directory_exists(os.path.dirname(self.config_path))
                
                # Guardar configuración por defecto de manera segura
                if safe_json_dump(default_config, self.config_path):
                    logger.info(f"Configuración por defecto guardada en {self.config_path}")
                else:
                    logger.warning(f"No se pudo guardar configuración por defecto en {self.config_path}")
                
                return default_config
                
        except Exception as e:
            logger.error(f"Error al cargar configuración: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return default_config
    
    def _build_model(self) -> Model:
        """
        Construye el modelo según el tipo especificado.
        
        Returns:
            Modelo Keras compilado
        """
        try:
            # Verificar si ya existe un modelo cargado y si es necesario reconstruirlo
            if self.model is not None:
                logger.info("Ya existe un modelo cargado. Se usará el modelo existente.")
                return self.model
                
            # Construir modelo según tipo
            model_builders = {
                'lstm': self._build_lstm_model,
                'gru': self._build_gru_model,
                'bilstm': self._build_bilstm_model,
                'attention': self._build_attention_model
            }
            
            # Obtener constructor adecuado o usar LSTM por defecto
            builder = model_builders.get(self.model_type.lower())
            if builder is None:
                logger.warning(f"Tipo de modelo '{self.model_type}' no reconocido. Usando LSTM por defecto.")
                builder = self._build_lstm_model
                self.model_type = 'lstm'  # Actualizar tipo de modelo
            
            # Construir modelo
            model = builder()
            
            # Obtener parámetros de compilación
            model_config = self.config.get(self.model_type, {})
            learning_rate = model_config.get("learning_rate", 0.001)
            optimizer_name = model_config.get("optimizer", "adam").lower()
            
            # Seleccionar optimizador
            if optimizer_name == "adam":
                optimizer = Adam(learning_rate=learning_rate)
            elif optimizer_name == "rmsprop":
                from tensorflow.keras.optimizers import RMSprop
                optimizer = RMSprop(learning_rate=learning_rate)
            elif optimizer_name == "sgd":
                from tensorflow.keras.optimizers import SGD
                optimizer = SGD(learning_rate=learning_rate)
            else:
                logger.warning(f"Optimizador '{optimizer_name}' no reconocido. Usando Adam.")
                optimizer = Adam(learning_rate=learning_rate)
            
            # Compilar modelo según tipo de problema
            if self.num_classes == 1:  # Regresión
                model.compile(
                    optimizer=optimizer, 
                    loss='mse', 
                    metrics=['mae', 'mse']
                )
                logger.info("Modelo compilado para regresión")
            else:  # Clasificación
                model.compile(
                    optimizer=optimizer, 
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy', 'sparse_categorical_accuracy']
                )
                logger.info(f"Modelo compilado para clasificación con {self.num_classes} clases")
            
            # Registrar resumen del modelo
            model.summary(print_fn=logger.info)
            
            # Registrar parámetros del modelo
            trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
            non_trainable_params = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
            logger.info(f"Parámetros totales: {trainable_params + non_trainable_params:,}")
            logger.info(f"Parámetros entrenables: {trainable_params:,}")
            logger.info(f"Parámetros no entrenables: {non_trainable_params:,}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error al construir modelo: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            
            # Intentar construir un modelo mínimo como fallback
            logger.warning("Intentando construir modelo mínimo como fallback...")
            try:
                # Modelo simple como fallback
                fallback_model = Sequential([
                    LSTM(32, input_shape=(self.sequence_length, self.num_features)),
                    Dense(16, activation='relu'),
                    Dense(self.num_classes, activation='softmax' if self.num_classes > 1 else 'linear')
                ])
                
                # Compilar modelo fallback
                fallback_model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy' if self.num_classes > 1 else 'mse',
                    metrics=['accuracy'] if self.num_classes > 1 else ['mae']
                )
                
                logger.info("Modelo fallback construido correctamente")
                return fallback_model
                
            except Exception as fallback_error:
                logger.error(f"Error al construir modelo fallback: {str(fallback_error)}")
                raise RuntimeError("No se pudo construir ningún modelo válido") from fallback_error
    
    def _build_lstm_model(self) -> Model:
        """
        Construye un modelo LSTM multicapa.
        
        Returns:
            Modelo Keras compilado
        """
        config = self.config.get("lstm", {})
        units = config.get("units", [128, 64, 32])
        dropout_rate = config.get("dropout", 0.3)
        recurrent_dropout = config.get("recurrent_dropout", 0.3)
        l1_reg = config.get("l1_reg", 0.0001)
        l2_reg = config.get("l2_reg", 0.0001)
        
        model = Sequential()
        
        # Primera capa LSTM (devuelve secuencias para capas intermedias)
        model.add(LSTM(units[0], 
                       input_shape=(self.sequence_length, self.num_features),
                       return_sequences=len(units) > 1,
                       dropout=dropout_rate,
                       recurrent_dropout=recurrent_dropout,
                       kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
        
        model.add(BatchNormalization())
        
        # Capas LSTM intermedias
        for i in range(1, len(units) - 1):
            model.add(LSTM(units[i],
                          return_sequences=True,
                          dropout=dropout_rate,
                          recurrent_dropout=recurrent_dropout,
                          kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
            model.add(BatchNormalization())
        
        # Última capa LSTM (no devuelve secuencias)
        if len(units) > 1:
            model.add(LSTM(units[-1],
                          return_sequences=False,
                          dropout=dropout_rate,
                          recurrent_dropout=recurrent_dropout,
                          kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
            model.add(BatchNormalization())
        
        # Capa de salida
        model.add(Dense(self.num_classes, activation='softmax'))
        
        return model
        
    def save(self, custom_path: Optional[str] = None) -> bool:
        """
        Guarda el modelo completo con su configuración.
        
        Args:
            custom_path: Ruta personalizada para guardar (opcional)
            
        Returns:
            True si se guardó correctamente, False en caso contrario
        """
        try:
            # Verificar si el modelo existe
            if self.model is None:
                logger.error("No hay modelo para guardar")
                return False
            
            # Determinar ruta
            save_path = custom_path or f"{self.model_path}_{self.model_type}.h5"
            
            # Asegurar que el directorio existe
            save_dir = os.path.dirname(save_path)
            if not ensure_directory_exists(save_dir):
                logger.error(f"No se pudo crear el directorio {save_dir}")
                return False
            
            # Guardar modelo con manejo de errores
            try:
                # Intentar guardar en formato HDF5
                self.model.save(save_path, save_format='h5')
                logger.info(f"Modelo guardado en formato HDF5: {save_path}")
            except Exception as h5_error:
                logger.warning(f"Error al guardar en formato HDF5: {str(h5_error)}")
                
                # Intentar guardar en formato SavedModel como alternativa
                try:
                    savedmodel_path = f"{os.path.splitext(save_path)[0]}_savedmodel"
                    self.model.save(savedmodel_path, save_format='tf')
                    logger.info(f"Modelo guardado en formato SavedModel: {savedmodel_path}")
                    save_path = savedmodel_path  # Actualizar ruta para metadatos
                except Exception as tf_error:
                    logger.error(f"Error al guardar en formato SavedModel: {str(tf_error)}")
                    return False
            
            # Preparar metadatos
            metadata = {
                'model_type': self.model_type,
                'sequence_length': self.sequence_length,
                'num_features': self.num_features,
                'num_classes': self.num_classes,
                'last_trained': self.last_trained.isoformat() if self.last_trained else None,
                'save_timestamp': datetime.now().isoformat(),
                'tensorflow_version': tf.__version__,
                'config': self.config,
                'model_format': 'savedmodel' if save_path.endswith('_savedmodel') else 'h5'
            }
            
            # Guardar metadatos de manera segura
            metadata_path = f"{os.path.splitext(save_path)[0]}_metadata.json"
            if not safe_json_dump(metadata, metadata_path):
                logger.warning(f"No se pudieron guardar los metadatos en {metadata_path}")
            
            # Guardar una copia de la configuración
            config_backup_path = os.path.join(save_dir, f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            if not safe_json_dump(self.config, config_backup_path):
                logger.warning(f"No se pudo guardar la copia de configuración en {config_backup_path}")
            
            logger.info(f"Modelo guardado correctamente en {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error al guardar modelo: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return False
    
    def load(self, custom_path: Optional[str] = None) -> bool:
        """
        Carga el modelo completo con su configuración, con soporte para Google Cloud Storage.
        
        Args:
            custom_path: Ruta personalizada para cargar (opcional)
            
        Returns:
            True si se cargó correctamente, False en caso contrario
        """
        try:
            # Verificar si estamos en entorno cloud
            is_cloud_env = os.environ.get('CLOUD_ENV') == 'true'
            bucket_name = os.environ.get('STORAGE_BUCKET')
            
            # Si estamos en entorno cloud, intentar descargar el modelo desde Cloud Storage
            if is_cloud_env and bucket_name and not custom_path:
                try:
                    logger.info(f"Intentando cargar modelo desde Cloud Storage: {bucket_name}")
                    
                    # Determinar rutas locales
                    local_model_dir = os.path.dirname(self.model_path)
                    ensure_directory_exists(local_model_dir)
                    
                    # Intentar usar google-cloud-storage
                    try:
                        from google.cloud import storage
                        client = storage.Client()
                        bucket = client.bucket(bucket_name)
                        
                        # Buscar modelos en el bucket
                        model_prefix = f"models/{os.path.basename(self.model_path)}_{self.model_type}"
                        blobs = list(bucket.list_blobs(prefix=model_prefix))
                        
                        if blobs:
                            # Ordenar por fecha de última modificación (más reciente primero)
                            blobs.sort(key=lambda x: x.updated, reverse=True)
                            latest_blob = blobs[0]
                            
                            # Determinar formato y ruta local
                            is_savedmodel_zip = latest_blob.name.endswith('_savedmodel.zip')
                            is_h5 = latest_blob.name.endswith('.h5')
                            
                            if is_savedmodel_zip:
                                # Descargar y descomprimir SavedModel
                                import tempfile
                                import zipfile
                                import shutil
                                
                                # Crear archivo temporal para el zip
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
                                    temp_zip = temp_file.name
                                
                                # Descargar a archivo temporal
                                latest_blob.download_to_filename(temp_zip)
                                logger.info(f"Modelo descargado a archivo temporal: {temp_zip}")
                                
                                # Preparar ruta para descomprimir
                                savedmodel_path = f"{os.path.splitext(self.model_path)[0]}_{self.model_type}_savedmodel"
                                if os.path.exists(savedmodel_path):
                                    shutil.rmtree(savedmodel_path, ignore_errors=True)
                                
                                # Descomprimir
                                with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                                    zip_ref.extractall(os.path.dirname(savedmodel_path))
                                
                                # Eliminar archivo temporal
                                os.unlink(temp_zip)
                                
                                logger.info(f"Modelo SavedModel descomprimido en: {savedmodel_path}")
                                custom_path = savedmodel_path
                                
                            elif is_h5:
                                # Descargar modelo H5 directamente
                                h5_path = f"{self.model_path}_{self.model_type}.h5"
                                latest_blob.download_to_filename(h5_path)
                                logger.info(f"Modelo H5 descargado a: {h5_path}")
                                custom_path = h5_path
                            
                            # Descargar metadatos si existen
                            metadata_blob_name = latest_blob.name.replace('.zip', '_metadata.json').replace('.h5', '_metadata.json')
                            metadata_blob = bucket.blob(metadata_blob_name)
                            
                            if metadata_blob.exists():
                                metadata_path = f"{os.path.splitext(self.model_path)[0]}_{self.model_type}_metadata.json"
                                metadata_blob.download_to_filename(metadata_path)
                                logger.info(f"Metadatos descargados a: {metadata_path}")
                        else:
                            logger.warning(f"No se encontraron modelos en Cloud Storage con prefijo: {model_prefix}")
                    
                    except ImportError:
                        # Alternativa: usar gsutil
                        logger.warning("No se pudo importar google-cloud-storage. Intentando con gsutil...")
                        try:
                            import subprocess
                            import glob
                            
                            # Buscar último modelo en GCS
                            gs_path = f"gs://{bucket_name}/models/"
                            result = subprocess.run(["gsutil", "ls", "-l", f"{gs_path}*{self.model_type}*"], 
                                                    capture_output=True, text=True)
                            
                            if result.returncode == 0 and result.stdout.strip():
                                # Parsear salida para encontrar el modelo más reciente
                                lines = result.stdout.strip().split('\n')
                                # Ordenar por fecha (gsutil ls -l muestra fecha en la columna 2)
                                lines.sort(key=lambda x: x.split()[1] if len(x.split()) > 1 else '', reverse=True)
                                latest_model = lines[0].split()[-1] if lines and len(lines[0].split()) > 0 else None
                                
                                if latest_model:
                                    # Descargar modelo
                                    local_path = os.path.join(local_model_dir, os.path.basename(latest_model))
                                    subprocess.run(["gsutil", "cp", latest_model, local_path])
                                    logger.info(f"Modelo descargado desde GCS a: {local_path}")
                                    custom_path = local_path
                                    
                                    # Descargar metadatos si existen
                                    metadata_gs_path = latest_model.replace('.zip', '_metadata.json').replace('.h5', '_metadata.json')
                                    metadata_local_path = local_path.replace('.zip', '_metadata.json').replace('.h5', '_metadata.json')
                                    subprocess.run(["gsutil", "cp", metadata_gs_path, metadata_local_path], 
                                                  capture_output=True)
                            else:
                                logger.warning(f"No se encontraron modelos en GCS o error en gsutil: {result.stderr}")
                        except Exception as gsutil_error:
                            logger.warning(f"Error al usar gsutil: {str(gsutil_error)}")
                
                except Exception as cloud_error:
                    logger.warning(f"Error al cargar desde Cloud Storage: {str(cloud_error)}. Intentando carga local.")
            
            # Determinar ruta local
            load_path = custom_path or f"{self.model_path}_{self.model_type}.h5"
            
            # Verificar si existe el modelo en formato H5
            is_h5_model = os.path.exists(load_path) and load_path.endswith('.h5')
            
            # Verificar si existe el modelo en formato SavedModel
            savedmodel_path = f"{os.path.splitext(load_path)[0]}_savedmodel"
            is_savedmodel = os.path.exists(savedmodel_path) and os.path.isdir(savedmodel_path)
            
            # Si no existe ninguno de los formatos, salir
            if not is_h5_model and not is_savedmodel:
                logger.warning(f"No se encontró modelo en {load_path} ni en {savedmodel_path}")
                return False
            
            # Cargar modelo según formato disponible
            try:
                # Configurar TensorFlow para limitar uso de memoria
                if is_cloud_env:
                    try:
                        # Limitar uso de memoria
                        gpus = tf.config.experimental.list_physical_devices('GPU')
                        if gpus:
                            for gpu in gpus:
                                try:
                                    tf.config.experimental.set_memory_growth(gpu, True)
                                    logger.info(f"Memory growth configurado para GPU: {gpu}")
                                except Exception as e:
                                    logger.warning(f"Error al configurar memory growth: {str(e)}")
                    except Exception as tf_config_error:
                        logger.warning(f"Error al configurar TensorFlow: {str(tf_config_error)}")
                
                # Monitorear memoria antes de cargar
                try:
                    import psutil
                    process = psutil.Process(os.getpid())
                    memory_before = process.memory_info().rss / (1024 * 1024)
                    logger.info(f"Memoria antes de cargar modelo: {memory_before:.2f} MB")
                except Exception:
                    pass
                
                # Cargar modelo
                if is_h5_model:
                    # Intentar cargar modelo H5
                    try:
                        self.model = load_model(load_path, compile=True)
                        logger.info(f"Modelo cargado desde formato HDF5: {load_path}")
                        model_path = load_path
                    except Exception as h5_error:
                        logger.error(f"Error al cargar modelo H5: {str(h5_error)}")
                        raise
                elif is_savedmodel:
                    # Intentar cargar modelo SavedModel
                    try:
                        self.model = load_model(savedmodel_path, compile=True)
                        logger.info(f"Modelo cargado desde formato SavedModel: {savedmodel_path}")
                        model_path = savedmodel_path
                    except Exception as tf_error:
                        logger.error(f"Error al cargar modelo SavedModel: {str(tf_error)}")
                        raise
                
                # Monitorear memoria después de cargar
                try:
                    memory_after = process.memory_info().rss / (1024 * 1024)
                    logger.info(f"Memoria después de cargar modelo: {memory_after:.2f} MB (Incremento: {memory_after - memory_before:.2f} MB)")
                except Exception:
                    pass
                    
            except Exception as load_error:
                logger.error(f"No se pudo cargar el modelo: {str(load_error)}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                return False
            
            # Cargar metadatos
            metadata_path = f"{os.path.splitext(load_path)[0]}_metadata.json"
            metadata = safe_json_load(metadata_path, default={})
            
            if metadata:
                # Actualizar atributos
                self.model_type = metadata.get('model_type', self.model_type)
                self.sequence_length = metadata.get('sequence_length', self.sequence_length)
                self.num_features = metadata.get('num_features', self.num_features)
                self.num_classes = metadata.get('num_classes', self.num_classes)
                
                # Convertir last_trained a datetime
                last_trained_str = metadata.get('last_trained')
                if last_trained_str:
                    try:
                        self.last_trained = datetime.fromisoformat(last_trained_str)
                    except ValueError:
                        logger.warning(f"Formato de fecha inválido: {last_trained_str}")
                        self.last_trained = None
                
                # Actualizar configuración
                if 'config' in metadata:
                    self.config = metadata['config']
                    
                logger.info(f"Metadatos cargados desde {metadata_path}")
            else:
                logger.warning(f"No se encontraron metadatos en {metadata_path}")
            
            # Verificar compatibilidad del modelo
            input_shape = self.model.input_shape
            if input_shape[1:] != (self.sequence_length, self.num_features):
                logger.warning(f"La forma de entrada del modelo cargado {input_shape[1:]} no coincide con la esperada ({self.sequence_length}, {self.num_features})")
            
            output_shape = self.model.output_shape
            if output_shape[-1] != self.num_classes:
                logger.warning(f"La forma de salida del modelo cargado {output_shape[-1]} no coincide con la esperada ({self.num_classes})")
            
            logger.info(f"Modelo cargado correctamente desde {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error al cargar modelo: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _build_gru_model(self) -> Model:
        """
        Construye un modelo GRU multicapa.
        
        Returns:
            Modelo Keras compilado
        """
        config = self.config.get("gru", {})
        units = config.get("units", [128, 64, 32])
        dropout_rate = config.get("dropout", 0.3)
        recurrent_dropout = config.get("recurrent_dropout", 0.3)
        l1_reg = config.get("l1_reg", 0.0001)
        l2_reg = config.get("l2_reg", 0.0001)
        
        model = Sequential()
        
        # Primera capa GRU (devuelve secuencias para capas intermedias)
        model.add(GRU(units[0], 
                      input_shape=(self.sequence_length, self.num_features),
                      return_sequences=len(units) > 1,
                      dropout=dropout_rate,
                      recurrent_dropout=recurrent_dropout,
                      kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
        
        model.add(BatchNormalization())
        
        # Capas GRU intermedias
        for i in range(1, len(units) - 1):
            model.add(GRU(units[i],
                         return_sequences=True,
                         dropout=dropout_rate,
                         recurrent_dropout=recurrent_dropout,
                         kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
            model.add(BatchNormalization())
        
        # Última capa GRU (no devuelve secuencias)
        if len(units) > 1:
            model.add(GRU(units[-1],
                         return_sequences=False,
                         dropout=dropout_rate,
                         recurrent_dropout=recurrent_dropout,
                         kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
            model.add(BatchNormalization())
        
        # Capa de salida
        model.add(Dense(self.num_classes, activation='softmax'))
        
        return model

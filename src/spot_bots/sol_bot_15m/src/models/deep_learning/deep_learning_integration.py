#!/usr/bin/env python3
"""
Módulo de integración de aprendizaje profundo para el bot SOL.
Permite utilizar modelos LSTM/GRU para mejorar las decisiones de trading.
Diseñado para ser compatible con entornos cloud y robusto ante fallos.
"""

import os
import sys
import logging
import traceback
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import time
import pytz
import warnings
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

# Suprimir advertencias innecesarias
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Ajustar el PYTHONPATH para encontrar los módulos correctamente
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Importar utilidades propias
try:
    from utils.cloud_utils import (
        ensure_directory_exists, 
        safe_json_dump, 
        safe_json_load, 
        get_memory_usage,
        setup_logging,
        retry_with_backoff
    )
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
            
    def get_memory_usage():
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            return {
                'rss': mem_info.rss / (1024 * 1024),  # MB
                'vms': mem_info.vms / (1024 * 1024),  # MB
                'percent': process.memory_percent()
            }
        except:
            return {'rss': 0, 'vms': 0, 'percent': 0}
    
    def setup_logging(level=logging.INFO):
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def retry_with_backoff(func=None, stop_max_attempt_number=3, wait_exponential_multiplier=1000, wait_exponential_max=10000):
        """Decorador para reintentar funciones con backoff exponencial"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                attempt = 0
                while attempt < stop_max_attempt_number:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        attempt += 1
                        if attempt == stop_max_attempt_number:
                            raise
                        wait_time = min(
                            wait_exponential_multiplier * (2 ** (attempt - 1)),
                            wait_exponential_max
                        )
                        time.sleep(wait_time / 1000.0)  # Convertir a segundos
            return wrapper
        return decorator if func is None else decorator(func)

# Configurar logging
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar módulos propios
try:
    from models.deep_learning.data_loader import MultiTimeframeDataLoader
    from models.deep_learning.data_processor import DeepLearningDataProcessor
    from models.deep_learning.lstm_model import DeepTimeSeriesModel
except ImportError as e:
    logger.error(f"Error al importar módulos de deep learning: {str(e)}")
    logger.error(f"Asegúrate de que los módulos están en el PYTHONPATH: {sys.path}")
    raise

class DeepLearningIntegration:
    """
    Integración de modelos de aprendizaje profundo para el bot SOL.
    Proporciona predicciones y señales basadas en modelos LSTM/GRU.
    Compatible con entornos cloud y robusto ante fallos.
    """
    
    def __init__(self, 
                 symbol: str = 'SOL/USDT',
                 timeframes: List[str] = None,
                 base_timeframe: str = '15m',
                 model_type: str = 'lstm',
                 models_dir: Optional[str] = None,
                 prediction_threshold: float = 0.65,
                 retrain_interval_minutes: int = 1440,
                 use_cloud_storage: bool = False,
                 cloud_bucket: Optional[str] = None,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 exchange: str = 'binance'):
        """
        Inicializa la integración de aprendizaje profundo con soporte para entornos cloud.
        
        Args:
            symbol: Par de trading
            timeframes: Lista de intervalos temporales a utilizar
            base_timeframe: Timeframe base para sincronización
            model_type: Tipo de modelo ('lstm', 'gru', 'bilstm', 'attention')
            models_dir: Directorio donde se encuentran los modelos
            prediction_threshold: Umbral de confianza para generar señales
            retrain_interval_minutes: Intervalo de reentrenamiento en minutos
            use_cloud_storage: Si es True, intenta usar almacenamiento en la nube
            cloud_bucket: Nombre del bucket de almacenamiento en la nube
            api_key: API key del exchange (opcional, puede usar variables de entorno)
            api_secret: API secret del exchange (opcional, puede usar variables de entorno)
            exchange: Nombre del exchange a utilizar
        """
        try:
            # Configurar valores por defecto
            if timeframes is None:
                timeframes = ['5m', '15m', '1h', '4h']
                
            # Configurar directorio de modelos
            if models_dir is None:
                # Usar variables de entorno si están disponibles
                models_dir = os.environ.get('MODELS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models', 'deep_learning'))
            
            # Asegurar que el directorio de modelos existe
            ensure_directory_exists(models_dir)
            
            # Configurar subdirectorios
            data_dir = os.path.join(models_dir, 'data')
            logs_dir = os.path.join(models_dir, 'logs')
            config_dir = os.path.join(models_dir, 'config')
            
            # Asegurar que los subdirectorios existen
            ensure_directory_exists(data_dir)
            ensure_directory_exists(logs_dir)
            ensure_directory_exists(config_dir)
            
            # Guardar parámetros
            self.symbol = symbol
            self.timeframes = timeframes
            self.base_timeframe = base_timeframe
            self.model_type = model_type
            self.models_dir = models_dir
            self.prediction_threshold = prediction_threshold
            self.retrain_interval_minutes = retrain_interval_minutes
            self.use_cloud_storage = use_cloud_storage
            self.cloud_bucket = cloud_bucket
            
            # Configuración de cloud
            if use_cloud_storage and cloud_bucket:
                logger.info(f"Configurado para usar almacenamiento en la nube: {cloud_bucket}")
                # Aquí se podría inicializar un cliente de almacenamiento en la nube
                
            # Inicializar componentes con manejo de errores
            try:
                self.data_loader = MultiTimeframeDataLoader(
                    symbol=symbol,
                    timeframes=timeframes,
                    data_dir=data_dir,
                    api_key=api_key,
                    api_secret=api_secret,
                    exchange=exchange
                )
                logger.info(f"Data loader inicializado para {symbol} en timeframes {timeframes}")
            except Exception as e:
                logger.error(f"Error al inicializar data loader: {str(e)}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                # Crear un data loader con valores por defecto
                self.data_loader = MultiTimeframeDataLoader(
                    symbol=symbol,
                    timeframes=timeframes,
                    data_dir=data_dir
                )
            
            try:
                self.data_processor = DeepLearningDataProcessor(
                    sequence_length=60,
                    prediction_horizon=3,
                    scaler_path=os.path.join(models_dir, 'dl_scaler.pkl'),
                    base_dir=models_dir,
                    use_cloud_storage=use_cloud_storage,
                    cloud_bucket=cloud_bucket
                )
                logger.info("Data processor inicializado")
            except Exception as e:
                logger.error(f"Error al inicializar data processor: {str(e)}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                # Crear un data processor con valores por defecto
                self.data_processor = DeepLearningDataProcessor(
                    sequence_length=60,
                    prediction_horizon=3,
                    scaler_path=os.path.join(models_dir, 'dl_scaler.pkl')
                )
            
            # Cargar modelo con manejo de errores
            self.model = None
            self.last_trained = None
            self.model_loaded = self.load_model()
            
            # Última predicción
            self.last_prediction = None
            self.last_prediction_time = None
            
            # Configuración de monitoreo
            self.metrics = {
                'predictions_count': 0,
                'successful_predictions': 0,
                'failed_predictions': 0,
                'last_memory_usage': get_memory_usage()
            }
            
            # Guardar configuración
            self._save_config()
            
            logger.info(f"Integración de aprendizaje profundo inicializada para {symbol} usando modelo {model_type}")
            
        except Exception as e:
            logger.error(f"Error al inicializar integración de aprendizaje profundo: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            # Inicializar con valores mínimos para evitar errores
            self.symbol = symbol
            self.timeframes = timeframes
            self.base_timeframe = base_timeframe
            self.model_type = model_type
            self.models_dir = models_dir if models_dir else 'models/deep_learning'
            self.prediction_threshold = prediction_threshold
            self.retrain_interval_minutes = retrain_interval_minutes
            self.model = None
            self.last_trained = None
            self.last_prediction = None
            self.last_prediction_time = None
            self.metrics = {'predictions_count': 0, 'successful_predictions': 0, 'failed_predictions': 0}
            
            # Intentar inicializar componentes básicos
            try:
                self.data_loader = MultiTimeframeDataLoader(symbol=symbol, timeframes=timeframes)
                self.data_processor = DeepLearningDataProcessor()
            except Exception:
                # Si todo falla, crear objetos vacíos
                self.data_loader = None
                self.data_processor = None
    
    def _save_config(self) -> bool:
        """
        Guarda la configuración actual en un archivo JSON.
        
        Returns:
            True si se guardó correctamente, False en caso contrario
        """
        try:
            config = {
                'symbol': self.symbol,
                'timeframes': self.timeframes,
                'base_timeframe': self.base_timeframe,
                'model_type': self.model_type,
                'prediction_threshold': self.prediction_threshold,
                'retrain_interval_minutes': self.retrain_interval_minutes,
                'use_cloud_storage': self.use_cloud_storage if hasattr(self, 'use_cloud_storage') else False,
                'cloud_bucket': self.cloud_bucket if hasattr(self, 'cloud_bucket') else None,
                'last_trained': self.last_trained.isoformat() if self.last_trained else None,
                'metrics': self.metrics if hasattr(self, 'metrics') else {},
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'version': '1.0.0',
                    'python_version': sys.version,
                    'platform': sys.platform
                }
            }
            
            config_path = os.path.join(self.models_dir, 'config', 'dl_integration_config.json')
            
            # Guardar configuración usando la función segura
            if safe_json_dump(config, config_path):
                logger.debug(f"Configuración guardada en {config_path}")
                
                # Si estamos en entorno cloud, también guardar en cloud storage
                if hasattr(self, 'use_cloud_storage') and self.use_cloud_storage and hasattr(self, 'cloud_bucket') and self.cloud_bucket:
                    try:
                        # Aquí se implementaría la lógica para guardar en GCS
                        logger.debug(f"Guardando configuración en cloud storage: {self.cloud_bucket}")
                    except Exception as cloud_error:
                        logger.warning(f"Error al guardar configuración en cloud storage: {str(cloud_error)}")
                
                return True
            else:
                logger.warning(f"No se pudo guardar la configuración en {config_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error al guardar configuración: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return False
    
    def load_model(self) -> bool:
        """
        Carga el modelo de aprendizaje profundo con soporte optimizado para Google Cloud.
        Implementa estrategias avanzadas de manejo de errores y gestión de recursos.
        
        Returns:
            True si se cargó correctamente, False en caso contrario
        """
        try:
            # Monitorear uso de memoria antes de cargar
            initial_memory = get_memory_usage()
            logger.info(f"Memoria antes de cargar modelo: {initial_memory.get('rss', 0):.2f} MB")
            
            # Verificar si estamos en entorno cloud
            is_cloud_env = os.environ.get('CLOUD_ENV', 'false').lower() == 'true'
            
            # Si estamos en entorno cloud, intentar cargar desde Cloud Storage
            if is_cloud_env and hasattr(self, 'use_cloud_storage') and self.use_cloud_storage and hasattr(self, 'cloud_bucket') and self.cloud_bucket:
                logger.info(f"Intentando cargar modelo desde Cloud Storage: {self.cloud_bucket}")
                model_loaded = self._load_model_from_cloud()
                if model_loaded:
                    return True
                logger.warning("No se pudo cargar desde Cloud Storage. Intentando carga local.")
            
            # Buscar modelo en el directorio local con múltiples formatos
            model_base_path = os.path.join(self.models_dir, f"models/lstm_model_{self.model_type}")
            h5_path = f"{model_base_path}.h5"
            savedmodel_path = f"{model_base_path}_savedmodel"
            metadata_path = f"{model_base_path}_metadata.json"
            
            # Verificar si existe alguna versión del modelo
            h5_exists = os.path.exists(h5_path)
            savedmodel_exists = os.path.exists(savedmodel_path) and os.path.isdir(savedmodel_path)
            
            if not h5_exists and not savedmodel_exists:
                logger.warning(f"No se encontró modelo en {h5_path} ni en {savedmodel_path}")
                return False
            
            # Cargar metadatos para obtener configuración
            metadata = {}
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Obtener timestamp de último entrenamiento
                    last_trained_str = metadata.get('last_trained')
                    if last_trained_str:
                        try:
                            self.last_trained = datetime.fromisoformat(last_trained_str)
                        except ValueError:
                            logger.warning(f"Formato de fecha inválido en metadatos: {last_trained_str}")
                            self.last_trained = None
                    
                    logger.info(f"Metadatos cargados desde {metadata_path}")
                except Exception as metadata_error:
                    logger.warning(f"Error al cargar metadatos: {str(metadata_error)}")
                    metadata = {}
            
            # Obtener parámetros del modelo (desde metadatos o valores por defecto)
            sequence_length = metadata.get('sequence_length', 60)
            num_features = metadata.get('num_features', 20)
            num_classes = metadata.get('num_classes', 3)
            
            # Implementar reintentos con backoff exponencial
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    # Si no es el primer intento, liberar memoria y esperar
                    if attempt > 0:
                        self._cleanup_memory()
                        wait_time = 2 ** attempt  # Backoff exponencial: 2, 4, 8 segundos
                        logger.info(f"Reintentando carga (intento {attempt+1}/{max_attempts}) después de {wait_time}s")
                        time.sleep(wait_time)
                    
                    # Inicializar modelo con configuración optimizada
                    model_path = os.path.join(self.models_dir, 'models/lstm_model')
                    
                    # Configurar opciones específicas para entorno cloud
                    if is_cloud_env:
                        # Añadir opciones específicas para cloud
                        self._configure_tensorflow_for_cloud()
                    
                    # Crear instancia del modelo
                    self.model = DeepTimeSeriesModel(
                        model_type=self.model_type,
                        sequence_length=sequence_length,
                        num_features=num_features,
                        num_classes=num_classes,
                        model_path=model_path
                    )
                    
                    # Cargar pesos con manejo de errores
                    if h5_exists:
                        logger.info(f"Intentando cargar modelo desde formato H5: {h5_path}")
                        success = self.model.load(custom_path=h5_path)
                    elif savedmodel_exists:
                        logger.info(f"Intentando cargar modelo desde formato SavedModel: {savedmodel_path}")
                        success = self.model.load(custom_path=savedmodel_path)
                    else:
                        success = False
                    
                    if success:
                        # Monitorear uso de memoria después de cargar
                        final_memory = get_memory_usage()
                        memory_increase = final_memory.get('rss', 0) - initial_memory.get('rss', 0)
                        logger.info(f"Modelo cargado correctamente. Incremento de memoria: {memory_increase:.2f} MB")
                        
                        # Verificar si el uso de memoria es alto
                        memory_limit = int(os.environ.get('MEMORY_LIMIT_MB', '2048'))
                        memory_percent = (final_memory.get('rss', 0) / memory_limit) * 100 if memory_limit > 0 else 0
                        
                        if memory_percent > 80:
                            logger.warning(f"Uso de memoria alto: {final_memory.get('rss', 0):.2f}/{memory_limit} MB ({memory_percent:.1f}%)")
                        
                        return True
                    else:
                        logger.error(f"Error al cargar modelo (intento {attempt+1}/{max_attempts})")
                        if attempt == max_attempts - 1:
                            return False
                
                except tf.errors.ResourceExhaustedError as resource_error:
                    logger.warning(f"Error de recursos agotados (intento {attempt+1}): {str(resource_error)}")
                    if attempt == max_attempts - 1:
                        logger.error("Se agotaron los intentos de carga debido a recursos insuficientes")
                        return False
                
                except Exception as load_error:
                    logger.error(f"Error al cargar modelo (intento {attempt+1}): {str(load_error)}")
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                    if attempt == max_attempts - 1:
                        return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error al cargar modelo: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _load_model_from_cloud(self) -> bool:
        """Intenta cargar el modelo desde Google Cloud Storage."""
        try:
            if not self.cloud_bucket:
                return False
                
            # Determinar rutas locales
            local_model_dir = os.path.join(self.models_dir, 'models')
            ensure_directory_exists(local_model_dir)
            
            # Intentar usar google-cloud-storage
            try:
                from google.cloud import storage
                client = storage.Client()
                bucket = client.bucket(self.cloud_bucket)
                
                # Buscar modelos en el bucket
                model_prefix = f"models/lstm_model_{self.model_type}"
                blobs = list(bucket.list_blobs(prefix=model_prefix))
                
                if not blobs:
                    logger.warning(f"No se encontraron modelos en Cloud Storage con prefijo: {model_prefix}")
                    return False
                    
                # Ordenar por fecha de última modificación (más reciente primero)
                blobs.sort(key=lambda x: x.updated, reverse=True)
                latest_blob = blobs[0]
                
                # Determinar formato y ruta local
                is_savedmodel = latest_blob.name.endswith('_savedmodel') or latest_blob.name.endswith('_savedmodel/')
                is_h5 = latest_blob.name.endswith('.h5')
                
                if not (is_savedmodel or is_h5):
                    # Buscar específicamente archivos de modelo
                    model_blobs = [b for b in blobs if b.name.endswith('.h5') or b.name.endswith('_savedmodel/')]
                    if model_blobs:
                        latest_blob = model_blobs[0]
                        is_savedmodel = latest_blob.name.endswith('_savedmodel') or latest_blob.name.endswith('_savedmodel/')
                        is_h5 = latest_blob.name.endswith('.h5')
                    else:
                        logger.warning("No se encontraron archivos de modelo válidos en Cloud Storage")
                        return False
                
                # Descargar según formato
                custom_path = None
                if is_h5:
                    # Descargar modelo H5 directamente
                    h5_path = os.path.join(local_model_dir, f"lstm_model_{self.model_type}.h5")
                    latest_blob.download_to_filename(h5_path)
                    logger.info(f"Modelo H5 descargado desde Cloud Storage a: {h5_path}")
                    custom_path = h5_path
                elif is_savedmodel:
                    # Para SavedModel, necesitamos descargar todo el directorio
                    # Primero, encontrar todos los archivos del SavedModel
                    savedmodel_prefix = latest_blob.name.rstrip('/')
                    savedmodel_blobs = list(bucket.list_blobs(prefix=savedmodel_prefix))
                    
                    # Crear directorio local
                    local_savedmodel_path = os.path.join(local_model_dir, f"lstm_model_{self.model_type}_savedmodel")
                    ensure_directory_exists(local_savedmodel_path)
                    
                    # Descargar cada archivo
                    for blob in savedmodel_blobs:
                        # Obtener ruta relativa dentro del SavedModel
                        rel_path = blob.name[len(savedmodel_prefix):].lstrip('/')
                        if not rel_path:  # Saltar el directorio raíz
                            continue
                            
                        # Crear subdirectorios si es necesario
                        local_file_path = os.path.join(local_savedmodel_path, rel_path)
                        ensure_directory_exists(os.path.dirname(local_file_path))
                        
                        # Descargar archivo
                        blob.download_to_filename(local_file_path)
                    
                    logger.info(f"Modelo SavedModel descargado desde Cloud Storage a: {local_savedmodel_path}")
                    custom_path = local_savedmodel_path
                
                # Descargar metadatos si existen
                metadata_blob_name = latest_blob.name.replace('.h5', '_metadata.json')
                if not metadata_blob_name.endswith('_metadata.json'):
                    metadata_blob_name = f"{metadata_blob_name.rstrip('/')}_metadata.json"
                    
                metadata_blob = bucket.blob(metadata_blob_name)
                
                if metadata_blob.exists():
                    metadata_path = os.path.join(local_model_dir, f"lstm_model_{self.model_type}_metadata.json")
                    metadata_blob.download_to_filename(metadata_path)
                    logger.info(f"Metadatos descargados desde Cloud Storage a: {metadata_path}")
                
                # Si tenemos una ruta válida, intentar cargar el modelo
                if custom_path:
                    # Inicializar modelo
                    self.model = DeepTimeSeriesModel(
                        model_type=self.model_type,
                        sequence_length=60,  # Se actualizará al cargar metadatos
                        num_features=20,     # Se actualizará al cargar metadatos
                        num_classes=3,       # Se actualizará al cargar metadatos
                        model_path=os.path.join(self.models_dir, 'models/lstm_model')
                    )
                    
                    # Cargar modelo
                    success = self.model.load(custom_path=custom_path)
                    return success
                
                return False
                
            except ImportError:
                # Alternativa: usar gsutil si está disponible
                logger.warning("No se pudo importar google-cloud-storage. Intentando con gsutil...")
                return self._load_model_from_cloud_gsutil()
                
        except Exception as e:
            logger.error(f"Error al cargar modelo desde Cloud Storage: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _load_model_from_cloud_gsutil(self) -> bool:
        """Intenta cargar el modelo desde Cloud Storage usando gsutil."""
        try:
            # Verificar si gsutil está disponible
            import subprocess
            result = subprocess.run(["gsutil", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning("gsutil no está disponible en el sistema")
                return False
            
            # Determinar rutas
            local_model_dir = os.path.join(self.models_dir, 'models')
            ensure_directory_exists(local_model_dir)
            gs_path = f"gs://{self.cloud_bucket}/models/"
            
            # Buscar último modelo en GCS
            result = subprocess.run(["gsutil", "ls", "-l", f"{gs_path}*{self.model_type}*"], 
                                   capture_output=True, text=True)
            
            if result.returncode != 0 or not result.stdout.strip():
                logger.warning(f"No se encontraron modelos en GCS o error en gsutil: {result.stderr}")
                return False
            
            # Parsear salida para encontrar el modelo más reciente
            lines = result.stdout.strip().split('\n')
            # Ordenar por fecha (gsutil ls -l muestra fecha en la columna 2)
            lines.sort(key=lambda x: x.split()[1] if len(x.split()) > 1 else '', reverse=True)
            latest_model = lines[0].split()[-1] if lines and len(lines[0].split()) > 0 else None
            
            if not latest_model:
                logger.warning("No se pudo determinar el modelo más reciente en GCS")
                return False
            
            # Determinar formato
            is_h5 = latest_model.endswith('.h5')
            is_savedmodel = '_savedmodel' in latest_model
            
            # Descargar modelo
            local_path = None
            if is_h5:
                local_path = os.path.join(local_model_dir, f"lstm_model_{self.model_type}.h5")
                subprocess.run(["gsutil", "cp", latest_model, local_path])
                logger.info(f"Modelo H5 descargado desde GCS a: {local_path}")
            elif is_savedmodel:
                # Para SavedModel, descargar todo el directorio
                local_path = os.path.join(local_model_dir, f"lstm_model_{self.model_type}_savedmodel")
                ensure_directory_exists(local_path)
                subprocess.run(["gsutil", "-m", "cp", "-r", latest_model, local_path])
                logger.info(f"Modelo SavedModel descargado desde GCS a: {local_path}")
            else:
                logger.warning(f"Formato de modelo no reconocido: {latest_model}")
                return False
            
            # Descargar metadatos si existen
            metadata_gs_path = latest_model.replace('.h5', '_metadata.json')
            if not metadata_gs_path.endswith('_metadata.json'):
                metadata_gs_path = f"{metadata_gs_path.rstrip('/')}_metadata.json"
                
            metadata_local_path = os.path.join(local_model_dir, f"lstm_model_{self.model_type}_metadata.json")
            subprocess.run(["gsutil", "cp", metadata_gs_path, metadata_local_path], capture_output=True)
            
            # Inicializar y cargar modelo
            if local_path:
                self.model = DeepTimeSeriesModel(
                    model_type=self.model_type,
                    sequence_length=60,  # Se actualizará al cargar metadatos
                    num_features=20,     # Se actualizará al cargar metadatos
                    num_classes=3,       # Se actualizará al cargar metadatos
                    model_path=os.path.join(self.models_dir, 'models/lstm_model')
                )
                
                # Cargar modelo
                success = self.model.load(custom_path=local_path)
                return success
            
            return False
            
        except Exception as e:
            logger.error(f"Error al cargar modelo desde GCS con gsutil: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _configure_tensorflow_for_cloud(self):
        """Configura TensorFlow para optimizar rendimiento en entornos cloud."""
        try:
            # Configurar TensorFlow para limitar uso de recursos
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                # Configurar crecimiento de memoria para GPUs
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        logger.info(f"Memory growth configurado para GPU: {gpu}")
                    except Exception as e:
                        logger.warning(f"Error al configurar memory growth: {str(e)}")
            
            # Limitar paralelismo para reducir uso de CPU
            tf.config.threading.set_inter_op_parallelism_threads(2)
            tf.config.threading.set_intra_op_parallelism_threads(2)
            
            # Configurar nivel de log para reducir mensajes
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
            
            logger.info("TensorFlow configurado para entorno cloud")
        except Exception as e:
            logger.warning(f"Error al configurar TensorFlow: {str(e)}")
    
    def _cleanup_memory(self):
        """Libera memoria después de operaciones intensivas."""
        try:
            # Limpiar sesión de Keras
            tf.keras.backend.clear_session()
            
            # Forzar recolección de basura
            import gc
            gc.collect()
            
            # Monitorear uso de memoria
            memory_usage = get_memory_usage()
            logger.debug(f"Memoria después de limpieza: {memory_usage.get('rss', 0):.2f} MB")
        except Exception as e:
            logger.warning(f"Error al liberar memoria: {str(e)}")
    
    def should_retrain(self) -> bool:
        """
        Determina si el modelo debe ser reentrenado.
        
        Returns:
            True si el modelo debe ser reentrenado, False en caso contrario
        """
        # Si nunca se ha entrenado, entrenar
        if self.last_trained is None:
            return True
        
        # Calcular tiempo transcurrido desde el último entrenamiento en minutos
        elapsed_minutes = (datetime.now() - self.last_trained).total_seconds() / 60
        
        # Reentrenar según intervalo configurado
        if elapsed_minutes >= self.retrain_interval_minutes:
            logger.info(f"Han pasado {elapsed_minutes:.1f} minutos desde el último entrenamiento. Reentrenando...")
            return True
        
        return False
    
    def retrain_model(self) -> bool:
        """
        Reentrenar el modelo con datos actualizados.
        
        Returns:
            True si se reentrenó correctamente, False en caso contrario
        """
        try:
            # Importar entrenador
            from models.deep_learning.model_trainer import DeepLearningTrainer
            
            # Inicializar entrenador
            trainer = DeepLearningTrainer(
                symbol=self.symbol,
                timeframes=self.timeframes,
                base_timeframe=self.base_timeframe,
                model_type=self.model_type,
                output_dir=self.models_dir
            )
            
            # Entrenar modelo
            metrics = trainer.train_model(force_update=True)
            
            if metrics:
                logger.info(f"Modelo reentrenado correctamente. Métricas: {metrics}")
                
                # Actualizar modelo cargado
                self.load_model()
                
                return True
            else:
                logger.error("Error al reentrenar modelo")
                return False
            
        except Exception as e:
            logger.error(f"Error al reentrenar modelo: {str(e)}")
            return False
    
    def get_prediction(self, market_data: pd.DataFrame) -> Dict:
        """
        Obtiene predicción del modelo para los datos de mercado actuales.
        
        Args:
            market_data: DataFrame con datos de mercado recientes
            
        Returns:
            Diccionario con predicción
        """
        try:
            # Verificar si hay datos suficientes
            if market_data.empty or len(market_data) < self.model.sequence_length:
                logger.warning(f"Datos insuficientes para predicción. Se necesitan al menos {self.model.sequence_length} registros.")
                return {}
            
            # Verificar si se debe reentrenar
            if self.should_retrain():
                logger.info("Reentrenando modelo...")
                self.retrain_model()
            
            # Actualizar datos multi-timeframe
            self.data_loader.update_data()
            
            # Obtener datos sincronizados
            data_dict = self.data_loader.get_synchronized_data(base_timeframe=self.base_timeframe)
            
            # Verificar si hay datos
            if not data_dict or self.base_timeframe not in data_dict:
                logger.error(f"No se pudieron obtener datos sincronizados para {self.base_timeframe}")
                return {}
            
            # Preparar datos multi-timeframe
            combined_df = self.data_processor.prepare_multi_timeframe_data(
                data_dict, 
                base_timeframe=self.base_timeframe
            )
            
            # Añadir características técnicas
            enhanced_df = self.data_processor.add_technical_features(combined_df)
            
            # Obtener datos para predicción (última secuencia disponible)
            X = self.data_processor.prepare_prediction_data(enhanced_df)
            
            if len(X) == 0:
                logger.error("No se pudieron preparar datos para predicción")
                return {}
            
            # Realizar predicción
            raw_predictions = self.model.predict(X)
            
            # Procesar predicción
            prediction = {}
            
            if self.model.num_classes > 1:  # Clasificación
                # Obtener clase con mayor probabilidad
                pred_class = np.argmax(raw_predictions[0])
                
                # Mapear clase a dirección
                direction_map = {
                    0: "NEUTRAL",
                    1: "ALCISTA",
                    -1: "BAJISTA"
                }
                
                # Obtener probabilidades
                probabilities = {
                    "BAJISTA": float(raw_predictions[0][0]) if raw_predictions[0].shape[0] > 1 else 0.0,
                    "NEUTRAL": float(raw_predictions[0][1]) if raw_predictions[0].shape[0] > 2 else 0.0,
                    "ALCISTA": float(raw_predictions[0][2 if raw_predictions[0].shape[0] > 2 else 1])
                }
                
                direction = direction_map.get(pred_class, "NEUTRAL")
                confidence = float(np.max(raw_predictions[0]))
                
                prediction = {
                    "direction": direction,
                    "confidence": confidence,
                    "probabilities": probabilities,
                    "signal": self._get_signal(direction, confidence),
                    "timestamp": datetime.now().isoformat()
                }
                
            else:  # Regresión
                # Para regresión, el valor predicho es directamente el cambio porcentual
                pred_value = float(raw_predictions[0][0])
                
                # Determinar dirección basada en el valor
                if pred_value > 0.005:  # Umbral positivo
                    direction = "ALCISTA"
                elif pred_value < -0.005:  # Umbral negativo
                    direction = "BAJISTA"
                else:
                    direction = "NEUTRAL"
                
                prediction = {
                    "direction": direction,
                    "predicted_change": pred_value,
                    "signal": self._get_signal_regression(pred_value),
                    "timestamp": datetime.now().isoformat()
                }
            
            # Guardar última predicción
            self.last_prediction = prediction
            self.last_prediction_time = datetime.now()
            
            logger.info(f"Predicción generada: {prediction}")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error al obtener predicción: {str(e)}")
            import traceback
            logger.debug(f"Traceback completo: {traceback.format_exc()}")
            return {}
    
    def _get_signal(self, direction: str, confidence: float) -> int:
        """
        Convierte dirección y confianza en señal de trading.
        
        Args:
            direction: Dirección predicha ('ALCISTA', 'BAJISTA', 'NEUTRAL')
            confidence: Confianza de la predicción (0-1)
            
        Returns:
            Señal de trading (1: compra, -1: venta, 0: mantener)
        """
        # Solo generar señal si la confianza supera el umbral
        if confidence < self.prediction_threshold:
            return 0
        
        if direction == "ALCISTA":
            return 1
        elif direction == "BAJISTA":
            return -1
        else:
            return 0
    
    def _get_signal_regression(self, predicted_change: float) -> int:
        """
        Convierte cambio porcentual predicho en señal de trading.
        
        Args:
            predicted_change: Cambio porcentual predicho
            
        Returns:
            Señal de trading (1: compra, -1: venta, 0: mantener)
        """
        # Umbrales para generar señales
        buy_threshold = 0.005  # 0.5%
        sell_threshold = -0.005  # -0.5%
        
        if predicted_change > buy_threshold:
            return 1
        elif predicted_change < sell_threshold:
            return -1
        else:
            return 0
    
    def enhance_trading_decision(self, 
                                technical_signal: int, 
                                market_data: pd.DataFrame,
                                risk_level: float = 0.5) -> Dict:
        """
        Mejora la decisión de trading combinando señal técnica con predicción de ML.
        
        Args:
            technical_signal: Señal basada en indicadores técnicos (1: compra, -1: venta, 0: mantener)
            market_data: DataFrame con datos de mercado recientes
            risk_level: Nivel de riesgo (0-1), influye en el peso de la predicción de ML
            
        Returns:
            Diccionario con decisión mejorada
        """
        try:
            # Obtener predicción de ML
            ml_prediction = self.get_prediction(market_data)
            
            if not ml_prediction:
                logger.warning("No se pudo obtener predicción de ML, usando solo señal técnica")
                return {
                    "signal": technical_signal,
                    "confidence": 0.5,
                    "ml_contribution": 0.0,
                    "final_decision": "TÉCNICA",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Obtener señal de ML
            ml_signal = ml_prediction.get("signal", 0)
            
            # Calcular peso de ML según nivel de riesgo
            ml_weight = risk_level
            technical_weight = 1 - ml_weight
            
            # Si las señales coinciden, aumentar confianza
            if technical_signal == ml_signal and technical_signal != 0:
                confidence = 0.5 + (0.5 * ml_prediction.get("confidence", 0.5))
                final_signal = technical_signal
                decision_type = "CONFIRMADA"
            
            # Si las señales son opuestas, usar la de mayor confianza
            elif technical_signal != 0 and ml_signal != 0 and technical_signal != ml_signal:
                ml_confidence = ml_prediction.get("confidence", 0.5)
                
                if ml_confidence > 0.8:  # Alta confianza de ML
                    final_signal = ml_signal
                    confidence = ml_confidence
                    decision_type = "ML"
                else:
                    final_signal = technical_signal
                    confidence = 0.5
                    decision_type = "TÉCNICA"
            
            # Si una señal es neutral, usar la otra
            elif technical_signal == 0 and ml_signal != 0:
                ml_confidence = ml_prediction.get("confidence", 0.5)
                
                if ml_confidence > self.prediction_threshold:
                    final_signal = ml_signal
                    confidence = ml_confidence
                    decision_type = "ML"
                else:
                    final_signal = 0
                    confidence = ml_confidence
                    decision_type = "NEUTRAL"
            
            elif technical_signal != 0 and ml_signal == 0:
                final_signal = technical_signal
                confidence = 0.5
                decision_type = "TÉCNICA"
            
            # Si ambas son neutrales
            else:
                final_signal = 0
                confidence = 0.5
                decision_type = "NEUTRAL"
            
            # Crear resultado
            result = {
                "technical_signal": technical_signal,
                "ml_signal": ml_signal,
                "ml_confidence": ml_prediction.get("confidence", 0.5),
                "ml_direction": ml_prediction.get("direction", "NEUTRAL"),
                "final_signal": final_signal,
                "confidence": confidence,
                "decision_type": decision_type,
                "risk_level": risk_level,
                "ml_weight": ml_weight,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Decisión mejorada: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error al mejorar decisión de trading: {str(e)}")
            return {
                "signal": technical_signal,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_adaptive_parameters(self, 
                              market_data: pd.DataFrame, 
                              current_params: Dict) -> Dict:
        """
        Obtiene parámetros adaptativos basados en predicciones de ML.
        
        Args:
            market_data: DataFrame con datos de mercado recientes
            current_params: Parámetros actuales
            
        Returns:
            Diccionario con parámetros adaptados
        """
        try:
            # Obtener predicción de ML
            ml_prediction = self.get_prediction(market_data)
            
            if not ml_prediction:
                logger.warning("No se pudo obtener predicción de ML, manteniendo parámetros actuales")
                return current_params
            
            # Copiar parámetros actuales
            new_params = current_params.copy()
            
            # Obtener dirección y confianza
            direction = ml_prediction.get("direction", "NEUTRAL")
            
            if "confidence" in ml_prediction:
                confidence = ml_prediction.get("confidence", 0.5)
            else:
                predicted_change = ml_prediction.get("predicted_change", 0)
                confidence = min(1.0, abs(predicted_change) * 10)  # Escalar cambio a confianza
            
            # Ajustar take profit según dirección y confianza
            if direction == "ALCISTA" and confidence > 0.7:
                # Mercado fuertemente alcista, aumentar take profit
                new_params["take_profit_pct"] = current_params.get("take_profit_pct", 0.03) * 1.2
                logger.info(f"Mercado fuertemente alcista, aumentando take profit a {new_params['take_profit_pct']:.4f}")
            
            elif direction == "BAJISTA" and confidence > 0.7:
                # Mercado fuertemente bajista, reducir take profit
                new_params["take_profit_pct"] = current_params.get("take_profit_pct", 0.03) * 0.8
                logger.info(f"Mercado fuertemente bajista, reduciendo take profit a {new_params['take_profit_pct']:.4f}")
            
            # Ajustar tamaño de posición según confianza
            if confidence > 0.8:
                # Alta confianza, aumentar tamaño de posición
                new_params["position_size_pct"] = current_params.get("position_size_pct", 0.1) * 1.1
                logger.info(f"Alta confianza, aumentando tamaño de posición a {new_params['position_size_pct']:.4f}")
            
            elif confidence < 0.6:
                # Baja confianza, reducir tamaño de posición
                new_params["position_size_pct"] = current_params.get("position_size_pct", 0.1) * 0.9
                logger.info(f"Baja confianza, reduciendo tamaño de posición a {new_params['position_size_pct']:.4f}")
            
            # Mantener stop loss fijo al 6% como solicitado por el usuario
            new_params["stop_loss_pct"] = 0.06
            
            return new_params
            
        except Exception as e:
            logger.error(f"Error al obtener parámetros adaptativos: {str(e)}")
            return current_params

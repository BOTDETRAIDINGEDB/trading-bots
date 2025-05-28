#!/usr/bin/env python3
"""
Módulo de preprocesamiento de datos para modelos de aprendizaje profundo.
Transforma datos crudos en características adecuadas para redes neuronales.
"""

import os
import sys
import logging
import traceback
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import pickle
from datetime import datetime, timedelta
import json
import time
import warnings
import gc
import tempfile
from contextlib import contextmanager

# Ajustar el PYTHONPATH para encontrar los módulos correctamente
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Importar utilidades propias
try:
    from utils.cloud_utils import ensure_directory_exists, safe_json_dump, safe_json_load
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

# Configurar logging
logger = logging.getLogger(__name__)

# Suprimir advertencias innecesarias
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class DeepLearningDataProcessor:
    """
    Procesador de datos para modelos de aprendizaje profundo.
    Prepara secuencias de datos para entrenar y predecir con modelos LSTM/GRU.
    Optimizado para entornos Google Cloud con gestión eficiente de memoria.
    """
    
    def __init__(self, 
                 sequence_length: int = 60,
                 prediction_horizon: int = 3,
                 scaler_path: Optional[str] = None,
                 base_dir: Optional[str] = None,
                 feature_config_path: Optional[str] = None,
                 use_cloud_storage: bool = False,
                 cloud_bucket: Optional[str] = None):
        """
        Inicializa el procesador de datos con soporte para entornos cloud.
        
        Args:
            sequence_length: Longitud de secuencia para LSTM/GRU
            prediction_horizon: Horizonte de predicción (cuántas velas adelante)
            scaler_path: Ruta donde guardar/cargar el escalador
            base_dir: Directorio base para almacenar archivos (si es None, usa directorio actual)
            feature_config_path: Ruta al archivo de configuración de características
            use_cloud_storage: Si es True, utiliza Google Cloud Storage
            cloud_bucket: Nombre del bucket de almacenamiento en la nube
        """
        try:
            # Configurar directorios base
            if base_dir is None:
                # Usar variables de entorno si están disponibles
                base_dir = os.environ.get('MODELS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models'))
            
            # Asegurar que el directorio base existe
            ensure_directory_exists(base_dir)
            
            # Configurar rutas
            config_dir = os.path.join(base_dir, 'config')
            models_dir = os.path.join(base_dir, 'models')
            
            # Asegurar que los directorios existen
            ensure_directory_exists(config_dir)
            ensure_directory_exists(models_dir)
            
            # Establecer rutas por defecto si no se proporcionan
            if feature_config_path is None:
                feature_config_path = os.path.join(config_dir, 'feature_config.json')
            
            if scaler_path is None:
                scaler_path = os.path.join(models_dir, 'dl_scaler.pkl')
            
            # Guardar parámetros
            self.sequence_length = sequence_length
            self.prediction_horizon = prediction_horizon
            self.scaler_type = 'robust'  # Tipo de escalador por defecto
            self.use_cloud_storage = use_cloud_storage
            self.cloud_bucket = cloud_bucket
            self.feature_config_path = feature_config_path
            self.scaler_path = scaler_path
            self.base_dir = base_dir
            self.use_cloud_storage = use_cloud_storage
            self.cloud_bucket = cloud_bucket
            
            # Configuración de cloud
            if use_cloud_storage and cloud_bucket:
                logger.info(f"Configurado para usar almacenamiento en la nube: {cloud_bucket}")
                # Aquí se podría inicializar un cliente de almacenamiento en la nube
            
            # Inicializar escalador según el tipo especificado
            if scaler_type == 'minmax':
                self.scaler = MinMaxScaler(feature_range=(-1, 1))  # Rango -1 a 1 para redes neuronales
            elif scaler_type == 'standard':
                self.scaler = StandardScaler()
            elif self.scaler_type == 'robust':
                self.scaler = RobustScaler()  # Mejor para datos con outliers
            else:
                logger.warning(f"Tipo de escalador '{self.scaler_type}' no reconocido. Usando RobustScaler por defecto.")
                self.scaler = RobustScaler()
            
            # Configurar opciones específicas para entorno cloud
            self.is_cloud_env = os.environ.get('CLOUD_ENV', 'false').lower() == 'true'
            
            if self.is_cloud_env:
                # Configurar opciones específicas para cloud
                self._configure_for_cloud()
            
            # Cargar configuración de características
            self.feature_config = self._load_feature_config()
            
            # Inicializar escalador
            self.scaler = self._load_scaler()
            
            # Inicializar caché para optimizar rendimiento
            self.cache = {}
            self.memory_usage_threshold_mb = int(os.environ.get('MEMORY_LIMIT_MB', '2048')) * 0.8
            
            logger.info(f"Procesador de datos inicializado con secuencia de {sequence_length} y horizonte de {prediction_horizon}")
            if self.use_cloud_storage:
                logger.info(f"Usando almacenamiento en la nube: {self.cloud_bucket}")
            
        except Exception as e:
            logger.error(f"Error al inicializar procesador de datos: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")            
            # Inicializar con valores mínimos para evitar errores
            self.sequence_length = sequence_length
            self.prediction_horizon = prediction_horizon
            self.feature_config_path = feature_config_path
            self.scaler_path = scaler_path
            self.feature_config = {}
            self.scaler = None
            self.cache = {}
            self.is_cloud_env = False
    
    def _configure_for_cloud(self):
        """Configura opciones específicas para entorno cloud."""
        try:
            logger.info("Configurando procesador de datos para entorno cloud")
            
            # Verificar si estamos en un entorno con memoria limitada
            memory_limit_mb = int(os.environ.get('MEMORY_LIMIT_MB', '2048'))
            logger.info(f"Límite de memoria configurado: {memory_limit_mb} MB")
            
            # Ajustar umbral de uso de memoria para limpieza automática
            self.memory_usage_threshold_mb = memory_limit_mb * 0.8
            
            # Configurar opciones para optimizar rendimiento en cloud
            os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
            
            # Verificar disponibilidad de Google Cloud Storage
            if self.use_cloud_storage and self.cloud_bucket:
                try:
                    # Intentar importar biblioteca de Google Cloud Storage
                    from google.cloud import storage
                    client = storage.Client()
                    bucket = client.bucket(self.cloud_bucket)
                    logger.info(f"Conexión a Google Cloud Storage establecida: {self.cloud_bucket}")
                except ImportError:
                    logger.warning("No se pudo importar google-cloud-storage. Verificando gsutil...")
                    try:
                        import subprocess
                        result = subprocess.run(["gsutil", "--version"], capture_output=True, text=True)
                        if result.returncode == 0:
                            logger.info("gsutil disponible para operaciones de almacenamiento")
                        else:
                            logger.warning("gsutil no está disponible. Deshabilitando almacenamiento en la nube.")
                            self.use_cloud_storage = False
                    except Exception:
                        logger.warning("Error al verificar gsutil. Deshabilitando almacenamiento en la nube.")
                        self.use_cloud_storage = False
                except Exception as e:
                    logger.warning(f"Error al conectar con Google Cloud Storage: {str(e)}")
                    self.use_cloud_storage = False
        except Exception as e:
            logger.warning(f"Error al configurar para entorno cloud: {str(e)}")
    
    @contextmanager
    def _memory_efficient_context(self):
        """Contexto para operaciones con uso eficiente de memoria."""
        try:
            # Monitorear memoria antes de la operación
            initial_memory = self._get_memory_usage()
            
            # Ejecutar operación
            yield
            
            # Monitorear memoria después de la operación
            final_memory = self._get_memory_usage()
            memory_increase = final_memory - initial_memory
            
            # Si el incremento es significativo, limpiar memoria
            if memory_increase > 100:  # Más de 100 MB
                self._cleanup_memory()
                logger.debug(f"Memoria liberada después de operación. Incremento: {memory_increase:.2f} MB")
        except Exception as e:
            logger.warning(f"Error en contexto de memoria eficiente: {str(e)}")
            raise
    
    def _get_memory_usage(self) -> float:
        """Obtiene el uso actual de memoria en MB."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convertir a MB
        except Exception:
            return 0.0
    
    def _cleanup_memory(self):
        """Libera memoria después de operaciones intensivas."""
        try:
            # Limpiar caché si existe
            if hasattr(self, 'cache'):
                self.cache.clear()
            
            # Forzar recolección de basura
            gc.collect()
            
            logger.debug("Memoria liberada")
        except Exception as e:
            logger.warning(f"Error al liberar memoria: {str(e)}")
        self.feature_config = self._load_feature_config()
        
        # Cargar escalador si existe
        self._load_scaler()
        
        logger.info(f"Inicializado procesador de datos con secuencia={sequence_length}, horizonte={prediction_horizon}")
    
    def _load_feature_config(self) -> Dict[str, Any]:
        """
        Carga la configuración de características desde archivo JSON con soporte para Google Cloud.
        Si el archivo no existe, crea uno con la configuración por defecto.
        
        Returns:
            Diccionario con configuración de características
        """
        # Configuración por defecto
        default_config = {
            "price_features": ["open", "high", "low", "close"],
            "volume_features": ["volume"],
            "technical_indicators": [
                "sma_20", "sma_50", "sma_200", "ema_9", "ema_21",
                "rsi_14", "macd", "macd_signal", "macd_histogram",
                "bb_upper", "bb_middle", "bb_lower", "atr_14"
            ],
            "derived_features": [
                "price_change", "volume_change", "volatility",
                "price_to_sma_ratio", "price_to_bb_ratio"
            ],
            "target_type": "classification",  # 'classification' o 'regression'
            "target_threshold": 0.005,  # 0.5% para clasificación
            "feature_importance_threshold": 0.01,  # Umbral para selección de características
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "version": "1.0.0",
                "description": "Configuración de características para modelo de aprendizaje profundo"
            }
        }
        
        try:
            # Verificar si estamos en entorno cloud
            if self.use_cloud_storage and self.cloud_bucket:
                try:
                    # Intentar cargar desde Google Cloud Storage
                    from google.cloud import storage
                    client = storage.Client()
                    bucket = client.bucket(self.cloud_bucket)
                    blob_name = f"models/config/{os.path.basename(self.feature_config_path)}"
                    blob = bucket.blob(blob_name)
                    
                    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
                        temp_path = temp_file.name
                    
                    # Descargar archivo a temporal
                    if blob.exists():
                        blob.download_to_filename(temp_path)
                        logger.info(f"Configuración descargada desde cloud storage: {blob_name}")
                        
                        # Cargar desde archivo temporal
                        config = safe_json_load(temp_path, default=None)
                        if config is not None:
                            # Limpiar archivo temporal
                            try:
                                os.unlink(temp_path)
                            except Exception:
                                pass
                                
                            # Validar la configuración
                            required_keys = ["price_features", "volume_features", "technical_indicators", "target_type"]
                            if all(key in config for key in required_keys):
                                logger.info("Configuración válida cargada desde cloud storage")
                                return config
                    
                    logger.info(f"No se encontró configuración válida en cloud storage: {blob_name}")
                    # Continuamos con la implementación local como fallback
                except Exception as cloud_error:
                    logger.warning(f"Error al cargar desde cloud storage: {str(cloud_error)}. Usando almacenamiento local.")
            
            # Cargar desde almacenamiento local
            if os.path.exists(self.feature_config_path):
                # Usar la función segura para cargar JSON
                config = safe_json_load(self.feature_config_path, default=None)
                if config is not None:
                    logger.info(f"Configuración de características cargada desde {self.feature_config_path}")
                    
                    # Validar la configuración cargada
                    required_keys = ["price_features", "volume_features", "technical_indicators", "target_type"]
                    if all(key in config for key in required_keys):
                        return config
                    else:
                        logger.warning("La configuración cargada no contiene todas las claves requeridas. Usando configuración por defecto.")
                        return default_config
                else:
                    logger.warning(f"No se pudo cargar la configuración desde {self.feature_config_path}. Usando configuración por defecto.")
            else:
                logger.warning(f"Archivo de configuración no encontrado: {self.feature_config_path}")
            
            # Si llegamos aquí, necesitamos crear la configuración por defecto
            # Crear directorio si no existe
            ensure_directory_exists(os.path.dirname(self.feature_config_path))
            
            # Actualizar timestamp en la configuración por defecto
            default_config["metadata"]["created_at"] = datetime.now().isoformat()
            
            # Guardar configuración por defecto usando la función segura
            if safe_json_dump(default_config, self.feature_config_path):
                logger.info(f"Configuración por defecto guardada en {self.feature_config_path}")
                
                # Si estamos en entorno cloud, también guardar en cloud storage
                if self.use_cloud_storage and self.cloud_bucket:
                    try:
                        from google.cloud import storage
                        client = storage.Client()
                        bucket = client.bucket(self.cloud_bucket)
                        blob_name = f"models/config/{os.path.basename(self.feature_config_path)}"
                        blob = bucket.blob(blob_name)
                        
                        # Guardar en archivo temporal primero
                        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
                            temp_path = temp_file.name
                            safe_json_dump(default_config, temp_path)
                        
                        # Subir archivo a Google Cloud Storage
                        blob.upload_from_filename(temp_path)
                        
                        # Limpiar archivo temporal
                        try:
                            os.unlink(temp_path)
                        except Exception:
                            pass
                            
                        logger.info(f"Configuración guardada en cloud storage: {blob_name}")
                    except Exception as e:
                        logger.warning(f"Error al guardar en cloud storage: {str(e)}. Configuración guardada solo localmente.")
            else:
                logger.warning(f"No se pudo guardar la configuración por defecto en {self.feature_config_path}")
            
            return default_config
            
        except Exception as e:
            logger.error(f"Error al cargar configuración de características: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return default_config
    
    def _load_scaler(self) -> bool:
        """
        Carga el escalador desde archivo si existe, con soporte para Google Cloud Storage.
        
        Returns:
            True si se cargó correctamente, False en caso contrario
        """
        try:
            # Verificar si estamos en entorno cloud
            if self.use_cloud_storage and self.cloud_bucket:
                try:
                    # Intentar cargar desde Google Cloud Storage
                    from google.cloud import storage
                    client = storage.Client()
                    bucket = client.bucket(self.cloud_bucket)
                    blob_name = f"models/scalers/{os.path.basename(self.scaler_path)}"
                    blob = bucket.blob(blob_name)
                    
                    # Verificar si existe el escalador en cloud storage
                    if blob.exists():
                        # Crear archivo temporal para descargar
                        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                            temp_path = temp_file.name
                        
                        # Descargar a archivo temporal
                        blob.download_to_filename(temp_path)
                        logger.info(f"Escalador descargado desde cloud storage: {blob_name}")
                        
                        # Cargar desde archivo temporal
                        with open(temp_path, 'rb') as f:
                            loaded_scaler = pickle.load(f)
                        
                        # Limpiar archivo temporal
                        try:
                            os.unlink(temp_path)
                        except Exception:
                            pass
                        
                        # Verificar que el escalador es válido
                        if hasattr(loaded_scaler, 'fit_transform') and hasattr(loaded_scaler, 'transform'):
                            self.scaler = loaded_scaler
                            logger.info("Escalador válido cargado desde cloud storage")
                            return True
                        else:
                            logger.warning("El escalador descargado desde cloud storage no es válido")
                    else:
                        logger.info(f"No se encontró escalador en cloud storage: {blob_name}")
                    # Si no se encuentra en cloud o no es válido, continuamos con la implementación local
                except Exception as cloud_error:
                    logger.warning(f"Error al cargar escalador desde cloud storage: {str(cloud_error)}. Usando almacenamiento local.")
            
            # Cargar desde almacenamiento local
            if os.path.exists(self.scaler_path):
                try:
                    # Intentar cargar el escalador
                    with open(self.scaler_path, 'rb') as f:
                        loaded_scaler = pickle.load(f)
                    
                    # Verificar que el escalador es válido
                    if hasattr(loaded_scaler, 'fit_transform') and hasattr(loaded_scaler, 'transform'):
                        self.scaler = loaded_scaler
                        logger.info(f"Escalador cargado desde {self.scaler_path}")
                        return True
                    else:
                        logger.warning("El objeto cargado no parece ser un escalador válido. Usando uno nuevo.")
                except (pickle.UnpicklingError, EOFError) as e:
                    logger.error(f"Error al deserializar el escalador: {str(e)}")
                except Exception as e:
                    logger.error(f"Error al cargar escalador: {str(e)}")
                    logger.debug(f"Traceback: {traceback.format_exc()}")
            else:
                logger.info(f"No se encontró archivo de escalador en {self.scaler_path}")
            
            # Si llegamos aquí, no se pudo cargar el escalador
            logger.info("Se usará un nuevo escalador")
            return False
            
        except Exception as e:
            logger.error(f"Error inesperado al cargar escalador: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _save_scaler(self) -> bool:
        """
        Guarda el escalador en archivo con soporte para Google Cloud Storage.
        
        Returns:
            True si se guardó correctamente, False en caso contrario
        """
        try:
            # Verificar que el escalador está inicializado
            if self.scaler is None:
                logger.warning("No hay escalador para guardar")
                return False
            
            # Crear directorio si no existe
            ensure_directory_exists(os.path.dirname(self.scaler_path))
            
            # Guardar localmente
            local_save_success = False
            try:
                with open(self.scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
                logger.info(f"Escalador guardado en {self.scaler_path}")
                local_save_success = True
            except Exception as local_error:
                logger.error(f"Error al guardar escalador localmente: {str(local_error)}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
            
            # Guardar en cloud si está configurado
            cloud_save_success = False
            if self.use_cloud_storage and self.cloud_bucket:
                try:
                    # Importar biblioteca de Google Cloud Storage
                    from google.cloud import storage
                    client = storage.Client()
                    bucket = client.bucket(self.cloud_bucket)
                    blob_name = f"models/scalers/{os.path.basename(self.scaler_path)}"
                    blob = bucket.blob(blob_name)
                    
                    # Crear archivo temporal para subir
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        temp_path = temp_file.name
                    
                    # Guardar en archivo temporal
                    with open(temp_path, 'wb') as f:
                        pickle.dump(self.scaler, f)
                    
                    # Subir archivo a Google Cloud Storage
                    blob.upload_from_filename(temp_path)
                    
                    # Limpiar archivo temporal
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass
                        
                    logger.info(f"Escalador guardado en cloud storage: {blob_name}")
                    cloud_save_success = True
                    return local_save_success or cloud_save_success
                except Exception as cloud_error:
                    logger.warning(f"Error al guardar escalador en cloud storage: {str(cloud_error)}")
                    return local_save_success
            
            return local_save_success
            
        except Exception as e:
            logger.error(f"Error inesperado al guardar escalador: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return False
    
    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade características técnicas derivadas al DataFrame.
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            DataFrame con características adicionales
        """
        # Crear copia para no modificar el original
        result = df.copy()
        
        try:
            # Características de cambio de precio
            result['price_change'] = result['close'].pct_change()
            result['high_low_ratio'] = result['high'] / result['low']
            result['close_open_ratio'] = result['close'] / result['open']
            
            # Características de volumen
            result['volume_change'] = result['volume'].pct_change()
            result['relative_volume'] = result['volume'] / result['volume'].rolling(20).mean()
            
            # Volatilidad
            result['volatility'] = result['close'].rolling(window=20).std() / result['close']
            
            # Características de tendencia
            if 'sma_20' in result.columns and 'sma_50' in result.columns:
                result['trend_strength'] = (result['sma_20'] / result['sma_50']) - 1
            
            # Características de momentum
            if 'rsi_14' in result.columns:
                result['rsi_change'] = result['rsi_14'].diff()
                result['rsi_divergence'] = (result['close'].pct_change() > 0) & (result['rsi_14'].diff() < 0)
            
            # Características de Bollinger Bands
            if all(col in result.columns for col in ['bb_upper', 'bb_lower', 'close']):
                result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
                result['bb_position'] = (result['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'])
            
            # Características multi-timeframe (si están disponibles)
            # Estas se añadirían desde fuera con datos de otros timeframes
            
            logger.info(f"Añadidas {len(result.columns) - len(df.columns)} características técnicas")
            
            # Eliminar filas con NaN
            result.dropna(inplace=True)
            
            return result
            
        except Exception as e:
            logger.error(f"Error al añadir características técnicas: {str(e)}")
            # En caso de error, devolver el DataFrame original
            return df
    
    def prepare_sequences(self, df: pd.DataFrame, for_training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara secuencias de datos para entrenar o predecir con modelos LSTM/GRU.
        Optimizado para entornos con memoria limitada como Google Cloud.
        
        Args:
            df: DataFrame con datos de precios y características
            for_training: Si es True, prepara datos para entrenamiento (incluye target)
                         Si es False, prepara datos solo para predicción
        
        Returns:
            Tupla con (X, y) donde:
                X: Array de secuencias de entrada (samples, sequence_length, features)
                y: Array de valores objetivo (samples, 1) o None si for_training=False
        """
        # Usar contexto de gestión eficiente de memoria
        with self._memory_efficient_context():
            try:
                # Verificar que el DataFrame no está vacío
                if df.empty:
                    logger.error("DataFrame vacío. No se pueden preparar secuencias.")
                    return np.array([]), np.array([])
                
                # Verificar que tenemos suficientes datos
                if len(df) < self.sequence_length + self.prediction_horizon:
                    logger.error(f"Insuficientes datos ({len(df)}) para secuencia de {self.sequence_length} + horizonte de {self.prediction_horizon}")
                    return np.array([]), np.array([])
                
                # Monitorear uso de memoria
                initial_memory = self._get_memory_usage()
                logger.debug(f"Memoria antes de preparar secuencias: {initial_memory:.2f} MB")
                
                # Preparar datos según modo (entrenamiento o predicción)
                if for_training:
                    # Para entrenamiento necesitamos características y target
                    if 'target' not in df.columns:
                        logger.error("Columna 'target' no encontrada en el DataFrame. Necesaria para entrenamiento.")
                        return np.array([]), np.array([])
                    
                    # Separar características y target
                    features = df.drop(columns=['target', 'timestamp'] if 'timestamp' in df.columns else ['target'])
                    target = df['target'].values
                else:
                    # Para predicción solo necesitamos características
                    features = df.drop(columns=['timestamp'] if 'timestamp' in df.columns else [])
                    target = None
                
                # Escalar características si el escalador está disponible
                if self.scaler is not None:
                    # Para entrenamiento, ajustamos y transformamos
                    if for_training:
                        features_scaled = self.scaler.fit_transform(features)
                        # Guardar escalador después de ajustar
                        self._save_scaler()
                    else:
                        # Para predicción, solo transformamos
                        try:
                            features_scaled = self.scaler.transform(features)
                        except Exception as e:
                            logger.error(f"Error al transformar características: {str(e)}")
                            # Si falla, intentamos usar fit_transform como fallback
                            features_scaled = self.scaler.fit_transform(features)
                            logger.warning("Se usó fit_transform como fallback para escalado")
                else:
                    logger.warning("Escalador no disponible. Usando características sin escalar.")
                    features_scaled = features.values
                
                # Crear secuencias con gestión eficiente de memoria
                X, y = [], []
                
                # Usar procesamiento por lotes para reducir uso de memoria
                batch_size = 1000  # Ajustar según disponibilidad de memoria
                total_sequences = len(features_scaled) - self.sequence_length - (self.prediction_horizon if for_training else 0) + 1
                
                for batch_start in range(0, total_sequences, batch_size):
                    batch_end = min(batch_start + batch_size, total_sequences)
                    
                    # Procesar lote actual
                    for i in range(batch_start, batch_end):
                        # Secuencia de entrada
                        seq = features_scaled[i:(i + self.sequence_length)]
                        X.append(seq)
                        
                        # Valor objetivo (futuro) solo para entrenamiento
                        if for_training and target is not None:
                            future_idx = i + self.sequence_length + self.prediction_horizon - 1
                            if future_idx < len(target):
                                y.append(target[future_idx])
                    
                    # Liberar memoria después de cada lote si estamos en entorno cloud
                    if self.is_cloud_env and batch_end < total_sequences:
                        gc.collect()
                
                # Convertir listas a arrays numpy
                X_array = np.array(X)
                
                if for_training and target is not None:
                    y_array = np.array(y).reshape(-1, 1)
                else:
                    y_array = np.array([])
                
                # Monitorear uso de memoria después del procesamiento
                final_memory = self._get_memory_usage()
                memory_increase = final_memory - initial_memory
                logger.debug(f"Memoria después de preparar secuencias: {final_memory:.2f} MB (incremento: {memory_increase:.2f} MB)")
                
                # Limpiar memoria si el incremento es significativo
                if memory_increase > 500:  # Más de 500 MB
                    self._cleanup_memory()
                
                return X_array, y_array
                
            except Exception as e:
                logger.error(f"Error al preparar secuencias: {str(e)}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                return np.array([]), np.array([])
    
    def predict_efficiently(self, model, data: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones de forma eficiente en memoria para entornos cloud.
        
        Args:
            model: Modelo de deep learning (TensorFlow/Keras)
            data: DataFrame con datos para predecir
            
        Returns:
            Array con predicciones
        """
        with self._memory_efficient_context():
            try:
                # Verificar que tenemos datos suficientes
                if len(data) < self.sequence_length:
                    logger.error(f"Insuficientes datos ({len(data)}) para secuencia de {self.sequence_length}")
                    return np.array([])
                
                # Preparar secuencias para predicción (sin target)
                X, _ = self.prepare_sequences(data, for_training=False)
                
                if len(X) == 0:
                    logger.error("No se pudieron preparar secuencias para predicción")
                    return np.array([])
                
                # Monitorear uso de memoria antes de predecir
                initial_memory = self._get_memory_usage()
                
                # Realizar predicciones por lotes para reducir uso de memoria
                batch_size = 32  # Ajustar según disponibilidad de memoria
                predictions = []
                
                # Usar TensorFlow en modo de memoria eficiente
                if self.is_cloud_env:
                    try:
                        import tensorflow as tf
                        # Configurar TensorFlow para limitar uso de memoria
                        gpus = tf.config.experimental.list_physical_devices('GPU')
                        if gpus:
                            for gpu in gpus:
                                tf.config.experimental.set_memory_growth(gpu, True)
                        
                        # Limitar memoria de GPU si está disponible
                        memory_limit_mb = int(os.environ.get('MEMORY_LIMIT_MB', '2048'))
                        if memory_limit_mb > 0:
                            tf.config.set_logical_device_configuration(
                                gpus[0],
                                [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)])
                    except Exception as e:
                        logger.warning(f"No se pudo configurar TensorFlow para memoria limitada: {str(e)}")
                
                # Predecir por lotes
                for i in range(0, len(X), batch_size):
                    batch_X = X[i:i + batch_size]
                    
                    try:
                        # Realizar predicción del lote
                        batch_pred = model.predict(batch_X, verbose=0)
                        predictions.append(batch_pred)
                        
                        # Liberar memoria después de cada lote si estamos en entorno cloud
                        if self.is_cloud_env and i + batch_size < len(X):
                            gc.collect()
                            
                    except Exception as batch_error:
                        logger.error(f"Error al predecir lote {i//batch_size}: {str(batch_error)}")
                        # Intentar con un lote más pequeño como fallback
                        try:
                            if len(batch_X) > 1:
                                logger.info("Intentando con un lote más pequeño...")
                                small_batch = batch_X[:1]  # Solo una muestra
                                small_pred = model.predict(small_batch, verbose=0)
                                predictions.append(small_pred)
                        except Exception as small_batch_error:
                            logger.error(f"Error al predecir lote pequeño: {str(small_batch_error)}")
                
                # Combinar predicciones de todos los lotes
                if predictions:
                    combined_predictions = np.vstack(predictions)
                    
                    # Monitorear uso de memoria después de predecir
                    final_memory = self._get_memory_usage()
                    memory_increase = final_memory - initial_memory
                    logger.debug(f"Memoria después de predecir: {final_memory:.2f} MB (incremento: {memory_increase:.2f} MB)")
                    
                    # Limpiar memoria si el incremento es significativo
                    if memory_increase > 200:  # Más de 200 MB
                        self._cleanup_memory()
                    
                    return combined_predictions
                else:
                    logger.error("No se generaron predicciones")
                    return np.array([])
                    
            except Exception as e:
                logger.error(f"Error al realizar predicciones eficientes: {str(e)}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                return np.array([])
    
    def save_data_to_cloud(self, data: pd.DataFrame, filename: str) -> bool:
        """
        Guarda datos en Google Cloud Storage de forma eficiente.
        
        Args:
            data: DataFrame a guardar
            filename: Nombre del archivo (sin ruta)
            
        Returns:
            True si se guardó correctamente, False en caso contrario
        """
        if not self.use_cloud_storage or not self.cloud_bucket:
            logger.warning("Almacenamiento en la nube no configurado. Guardando solo localmente.")
            return self.save_data_locally(data, filename)
            
        try:
            # Crear directorio para datos si no existe
            data_dir = os.path.join(self.base_dir, 'data')
            ensure_directory_exists(data_dir)
            
            # Ruta local completa
            local_path = os.path.join(data_dir, filename)
            
            # Guardar localmente primero
            local_save_success = False
            try:
                # Usar formato parquet para mejor eficiencia
                if filename.endswith('.parquet'):
                    data.to_parquet(local_path, index=True, compression='snappy')
                else:
                    # Fallback a CSV si no es parquet
                    data.to_csv(local_path, index=True)
                local_save_success = True
                logger.info(f"Datos guardados localmente en {local_path}")
            except Exception as local_error:
                logger.error(f"Error al guardar datos localmente: {str(local_error)}")
            
            # Guardar en cloud storage
            cloud_save_success = False
            try:
                from google.cloud import storage
                client = storage.Client()
                bucket = client.bucket(self.cloud_bucket)
                blob_name = f"data/{filename}"
                blob = bucket.blob(blob_name)
                
                # Subir archivo a Google Cloud Storage
                blob.upload_from_filename(local_path)
                cloud_save_success = True
                logger.info(f"Datos guardados en cloud storage: {blob_name}")
            except Exception as cloud_error:
                logger.error(f"Error al guardar datos en cloud storage: {str(cloud_error)}")
            
            return local_save_success or cloud_save_success
            
        except Exception as e:
            logger.error(f"Error al guardar datos: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return False
    
    def load_data_from_cloud(self, filename: str) -> Optional[pd.DataFrame]:
        """
        Carga datos desde Google Cloud Storage de forma eficiente.
        
        Args:
            filename: Nombre del archivo a cargar (sin ruta)
            
        Returns:
            DataFrame con los datos cargados o None si hay error
        """
        try:
            # Verificar si estamos usando cloud storage
            if self.use_cloud_storage and self.cloud_bucket:
                try:
                    from google.cloud import storage
                    client = storage.Client()
                    bucket = client.bucket(self.cloud_bucket)
                    blob_name = f"data/{filename}"
                    blob = bucket.blob(blob_name)
                    
                    # Verificar si existe en cloud storage
                    if blob.exists():
                        # Crear archivo temporal para descargar
                        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                            temp_path = temp_file.name
                        
                        # Descargar a archivo temporal
                        blob.download_to_filename(temp_path)
                        logger.info(f"Datos descargados desde cloud storage: {blob_name}")
                        
                        # Cargar desde archivo temporal
                        if filename.endswith('.parquet'):
                            df = pd.read_parquet(temp_path)
                        else:
                            # Fallback a CSV si no es parquet
                            df = pd.read_csv(temp_path, index_col=0)
                        
                        # Limpiar archivo temporal
                        try:
                            os.unlink(temp_path)
                        except Exception:
                            pass
                        
                        logger.info(f"Datos cargados desde cloud storage: {len(df)} filas")
                        return df
                    else:
                        logger.info(f"No se encontraron datos en cloud storage: {blob_name}")
                        # Continuamos con carga local como fallback
                except Exception as cloud_error:
                    logger.warning(f"Error al cargar desde cloud storage: {str(cloud_error)}. Intentando carga local.")
            
            # Cargar desde almacenamiento local
            data_dir = os.path.join(self.base_dir, 'data')
            local_path = os.path.join(data_dir, filename)
            
            if os.path.exists(local_path):
                try:
                    # Cargar según formato
                    if filename.endswith('.parquet'):
                        df = pd.read_parquet(local_path)
                    else:
                        # Fallback a CSV si no es parquet
                        df = pd.read_csv(local_path, index_col=0)
                    
                    logger.info(f"Datos cargados localmente desde {local_path}: {len(df)} filas")
                    return df
                except Exception as local_error:
                    logger.error(f"Error al cargar datos localmente: {str(local_error)}")
            else:
                logger.warning(f"Archivo local no encontrado: {local_path}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error al cargar datos: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
    
    def save_data_locally(self, data: pd.DataFrame, filename: str) -> bool:
        """
        Guarda datos localmente de forma eficiente.
        
        Args:
            data: DataFrame a guardar
            filename: Nombre del archivo (sin ruta)
            
        Returns:
            True si se guardó correctamente, False en caso contrario
        """
        try:
            # Crear directorio para datos si no existe
            data_dir = os.path.join(self.base_dir, 'data')
            ensure_directory_exists(data_dir)
            
            # Ruta local completa
            local_path = os.path.join(data_dir, filename)
            
            # Guardar según formato
            if filename.endswith('.parquet'):
                data.to_parquet(local_path, index=True, compression='snappy')
            else:
                # Fallback a CSV si no es parquet
                data.to_csv(local_path, index=True)
                
            logger.info(f"Datos guardados localmente en {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error al guardar datos localmente: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return False
    
    def prepare_multi_timeframe_data(self, 
                                     data_dict: Dict[str, pd.DataFrame], 
                                     base_timeframe: str = '15m') -> pd.DataFrame:
        """
        Prepara datos de múltiples timeframes para entrenamiento.
        
        Args:
            data_dict: Diccionario con DataFrames por timeframe
            base_timeframe: Timeframe base para sincronización
            
        Returns:
            DataFrame combinado con características de todos los timeframes
        """
        try:
            if base_timeframe not in data_dict:
                logger.error(f"Timeframe base '{base_timeframe}' no encontrado en los datos")
                return pd.DataFrame()
            
            # DataFrame base
            base_df = data_dict[base_timeframe].copy()
            
            # Para cada timeframe adicional, añadir características con sufijo
            for tf, df in data_dict.items():
                if tf == base_timeframe:
                    continue
                
                # Añadir sufijo a las columnas
                df_renamed = df.copy()
                df_renamed.columns = [f"{col}_{tf}" for col in df_renamed.columns]
                
                # Combinar con DataFrame base
                base_df = base_df.join(df_renamed, how='left')
            
            # Rellenar valores NaN con forward fill
            base_df.fillna(method='ffill', inplace=True)
            
            # Eliminar filas que aún tengan NaN
            base_df.dropna(inplace=True)
            
            logger.info(f"Datos multi-timeframe preparados: {len(base_df)} filas, {len(base_df.columns)} columnas")
            
            return base_df
            
        except Exception as e:
            logger.error(f"Error al preparar datos multi-timeframe: {str(e)}")
            return pd.DataFrame()
    
    def create_target(self, 
                      df: pd.DataFrame, 
                      target_type: str = 'classification',
                      threshold: float = 0.005) -> pd.DataFrame:
        """
        Crea la variable objetivo para el entrenamiento.
        
        Args:
            df: DataFrame con datos OHLCV
            target_type: Tipo de objetivo ('classification' o 'regression')
            threshold: Umbral para clasificación (% de cambio)
            
        Returns:
            DataFrame con columna objetivo añadida
        """
        try:
            result = df.copy()
            
            # Calcular cambio porcentual futuro
            future_pct_change = result['close'].pct_change(periods=self.prediction_horizon).shift(-self.prediction_horizon)
            
            if target_type == 'classification':
                # Clasificación: 1 (subida), 0 (neutral), -1 (bajada)
                target = np.zeros(len(future_pct_change))
                target[future_pct_change > threshold] = 1  # Subida significativa
                target[future_pct_change < -threshold] = -1  # Bajada significativa
                
                result['target'] = target
                logger.info(f"Target de clasificación creado: {np.sum(target == 1)} subidas, {np.sum(target == -1)} bajadas, {np.sum(target == 0)} neutral")
                
            elif target_type == 'regression':
                # Regresión: valor directo del cambio porcentual
                result['target'] = future_pct_change
                logger.info(f"Target de regresión creado: media={future_pct_change.mean():.4f}, std={future_pct_change.std():.4f}")
                
            else:
                logger.error(f"Tipo de target '{target_type}' no reconocido")
                return df
            
            # Eliminar filas con NaN en target
            result.dropna(subset=['target'], inplace=True)
            
            return result
            
        except Exception as e:
            logger.error(f"Error al crear target: {str(e)}")
            return df
    
    def get_feature_names(self) -> List[str]:
        """
        Obtiene los nombres de las características utilizadas.
        
        Returns:
            Lista de nombres de características
        """
        return self.scaler.feature_names_in_ if hasattr(self.scaler, 'feature_names_in_') else []
    
    def prepare_prediction_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepara datos para predicción (última secuencia disponible).
        
        Args:
            df: DataFrame con datos recientes
            
        Returns:
            Array numpy con la secuencia para predicción
        """
        try:
            # Verificar que hay suficientes datos
            if len(df) < self.sequence_length:
                logger.warning(f"Datos insuficientes para predicción. Se necesitan al menos {self.sequence_length} registros.")
                return np.array([])
            
            # Obtener últimas filas
            recent_data = df.iloc[-self.sequence_length:].copy()
            
            # Preparar características
            features = recent_data.drop(columns=['target'], errors='ignore')
            
            # Escalar características
            if hasattr(self.scaler, 'transform'):
                features_scaled = self.scaler.transform(features)
            else:
                logger.warning("Escalador no entrenado, usando fit_transform")
                features_scaled = self.scaler.fit_transform(features)
                self._save_scaler()
            
            # Crear secuencia (1 muestra con sequence_length pasos temporales)
            X = np.array([features_scaled])
            
            logger.info(f"Datos de predicción preparados con forma {X.shape}")
            
            return X
            
        except Exception as e:
            logger.error(f"Error al preparar datos para predicción: {str(e)}")
            return np.array([])

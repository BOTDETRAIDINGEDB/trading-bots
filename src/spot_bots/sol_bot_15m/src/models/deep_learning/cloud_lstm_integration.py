#!/usr/bin/env python3
"""
Módulo de integración optimizado para LSTM en Google Cloud.
Proporciona funcionalidades específicas para entornos cloud.
"""

import os
import logging
import time
import json
import traceback
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime

# Importar módulos locales
from models.deep_learning.lstm_model import DeepTimeSeriesModel
from models.deep_learning.data_processor import DeepLearningDataProcessor
from utils.tensorflow_cloud_config import configure_tensorflow_for_cloud, cleanup_tensorflow_memory

# Configurar logging
logger = logging.getLogger(__name__)

class CloudLSTMIntegration:
    """
    Integración optimizada de LSTM para entornos Google Cloud.
    Proporciona funcionalidades específicas para maximizar rendimiento y estabilidad.
    """
    
    def __init__(
        self,
        model_path: str,
        sequence_length: int = 60,
        prediction_horizon: int = 3,
        use_cloud_storage: bool = True,
        storage_bucket: Optional[str] = None,
        memory_limit_mb: Optional[int] = None
    ):
        """
        Inicializa la integración de LSTM para Google Cloud.
        
        Args:
            model_path: Ruta al modelo LSTM
            sequence_length: Longitud de secuencia para el modelo
            prediction_horizon: Horizonte de predicción
            use_cloud_storage: Si es True, utiliza Google Cloud Storage
            storage_bucket: Nombre del bucket de almacenamiento
            memory_limit_mb: Límite de memoria en MB
        """
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.use_cloud_storage = use_cloud_storage
        self.storage_bucket = storage_bucket or os.environ.get("STORAGE_BUCKET")
        self.memory_limit_mb = memory_limit_mb or int(os.environ.get("MEMORY_LIMIT_MB", "2048"))
        
        # Inicializar modelo y procesador
        self.model = None
        self.data_processor = None
        self.model_metadata = {}
        self.tf_config = {}
        
        # Configurar TensorFlow para entorno cloud
        self._configure_tensorflow()
        
        logger.info(f"Integración LSTM para Google Cloud inicializada. Modelo: {model_path}")
    
    def _configure_tensorflow(self) -> None:
        """Configura TensorFlow para rendimiento óptimo en Google Cloud."""
        try:
            # Determinar si estamos en entorno cloud
            in_cloud = os.environ.get("CLOUD_ENV", "false").lower() == "true"
            
            # Configurar TensorFlow
            self.tf_config = configure_tensorflow_for_cloud(
                memory_growth=True,
                memory_limit_mb=self.memory_limit_mb if in_cloud else None,
                deterministic=False,
                mixed_precision=in_cloud,
                log_device_placement=False
            )
            
            logger.info(f"TensorFlow configurado para entorno cloud: {self.tf_config}")
        except Exception as e:
            logger.error(f"Error al configurar TensorFlow: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
    
    def load_model_and_processor(self) -> bool:
        """
        Carga el modelo LSTM y el procesador de datos.
        
        Returns:
            True si la carga fue exitosa, False en caso contrario
        """
        try:
            # Cargar modelo
            self.model = DeepTimeSeriesModel(
                model_type="lstm",
                sequence_length=self.sequence_length,
                num_features=20,  # Se actualizará al cargar
                num_classes=3,
                model_path=self.model_path
            )
            
            # Intentar cargar el modelo
            start_time = time.time()
            load_success = self.model.load(custom_path=None)
            load_time = time.time() - start_time
            
            if not load_success:
                logger.error(f"Error al cargar el modelo LSTM desde {self.model_path}")
                return False
            
            logger.info(f"Modelo LSTM cargado correctamente en {load_time:.2f} segundos")
            
            # Obtener metadatos del modelo
            self.model_metadata = self.model.metadata
            
            # Inicializar procesador de datos con la configuración del modelo
            scaler_path = os.path.join(os.path.dirname(self.model_path), "dl_scaler.pkl")
            self.data_processor = DeepLearningDataProcessor(
                sequence_length=self.sequence_length,
                prediction_horizon=self.prediction_horizon,
                scaler_path=scaler_path,
                base_dir=os.path.dirname(self.model_path),
                use_cloud_storage=self.use_cloud_storage
            )
            
            logger.info(f"Procesador de datos inicializado. Scaler: {scaler_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error al cargar modelo y procesador: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return False
    
    def predict(self, input_data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Realiza una predicción con el modelo LSTM.
        
        Args:
            input_data: Datos de entrada para la predicción
            
        Returns:
            Tupla con (predicciones, metadatos)
        """
        if self.model is None:
            logger.error("El modelo no está cargado. Llame a load_model_and_processor primero.")
            return np.array([]), {"error": "Modelo no cargado"}
        
        try:
            # Medir uso de memoria antes de la predicción
            import psutil
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / (1024 * 1024)
            
            # Realizar predicción
            start_time = time.time()
            predictions = self.model.predict(input_data)
            prediction_time = time.time() - start_time
            
            # Medir uso de memoria después de la predicción
            mem_after = process.memory_info().rss / (1024 * 1024)
            mem_increase = mem_after - mem_before
            
            # Limpiar memoria si el incremento es significativo
            if mem_increase > 100:  # Más de 100 MB
                cleanup_tensorflow_memory()
                logger.info(f"Memoria liberada después de predicción. Incremento: {mem_increase:.2f} MB")
            
            # Preparar metadatos
            metadata = {
                "prediction_time_ms": prediction_time * 1000,
                "memory_before_mb": mem_before,
                "memory_after_mb": mem_after,
                "memory_increase_mb": mem_increase,
                "timestamp": datetime.now().isoformat(),
                "model_path": self.model_path,
                "model_metadata": self.model_metadata
            }
            
            logger.info(f"Predicción completada en {prediction_time:.4f} segundos")
            return predictions, metadata
        
        except Exception as e:
            logger.error(f"Error durante la predicción: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return np.array([]), {"error": str(e)}
    
    def process_and_predict(
        self, 
        data: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Procesa los datos y realiza una predicción.
        
        Args:
            data: Diccionario con datos para diferentes timeframes
            
        Returns:
            Tupla con (predicciones, metadatos)
        """
        if self.data_processor is None or self.model is None:
            logger.error("Procesador o modelo no inicializados. Llame a load_model_and_processor primero.")
            return np.array([]), {"error": "Componentes no inicializados"}
        
        try:
            # Procesar datos
            start_time = time.time()
            processed_data = self.data_processor.prepare_sequences(data)
            processing_time = time.time() - start_time
            
            # Realizar predicción
            predictions, pred_metadata = self.predict(processed_data)
            
            # Combinar metadatos
            metadata = {
                **pred_metadata,
                "processing_time_ms": processing_time * 1000,
                "total_time_ms": (processing_time + pred_metadata.get("prediction_time_ms", 0)/1000) * 1000
            }
            
            return predictions, metadata
        
        except Exception as e:
            logger.error(f"Error durante el procesamiento y predicción: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return np.array([]), {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Realiza una verificación de salud del sistema.
        
        Returns:
            Diccionario con estado de salud
        """
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "status": "unknown",
            "components": {},
            "memory_usage_mb": 0,
            "tensorflow_config": self.tf_config
        }
        
        try:
            # Verificar TensorFlow
            health_status["components"]["tensorflow"] = {
                "status": "ok" if self.tf_config.get("success", False) else "error",
                "version": tf.__version__,
                "gpu_available": self.tf_config.get("gpu_available", False)
            }
            
            # Verificar modelo
            if self.model is not None:
                health_status["components"]["model"] = {
                    "status": "loaded",
                    "path": self.model_path,
                    "metadata": self.model_metadata
                }
            else:
                health_status["components"]["model"] = {
                    "status": "not_loaded"
                }
            
            # Verificar procesador de datos
            if self.data_processor is not None:
                health_status["components"]["data_processor"] = {
                    "status": "initialized",
                    "scaler_loaded": hasattr(self.data_processor, "scaler") and self.data_processor.scaler is not None
                }
            else:
                health_status["components"]["data_processor"] = {
                    "status": "not_initialized"
                }
            
            # Verificar almacenamiento en la nube
            if self.use_cloud_storage and self.storage_bucket:
                try:
                    from google.cloud import storage
                    client = storage.Client()
                    bucket = client.bucket(self.storage_bucket)
                    health_status["components"]["cloud_storage"] = {
                        "status": "connected" if bucket.exists() else "not_found",
                        "bucket": self.storage_bucket
                    }
                except Exception as e:
                    health_status["components"]["cloud_storage"] = {
                        "status": "error",
                        "error": str(e)
                    }
            else:
                health_status["components"]["cloud_storage"] = {
                    "status": "disabled"
                }
            
            # Verificar uso de memoria
            try:
                import psutil
                process = psutil.Process(os.getpid())
                memory_usage = process.memory_info().rss / (1024 * 1024)
                health_status["memory_usage_mb"] = memory_usage
                health_status["components"]["memory"] = {
                    "status": "ok" if memory_usage < self.memory_limit_mb else "warning",
                    "usage_mb": memory_usage,
                    "limit_mb": self.memory_limit_mb,
                    "usage_percent": (memory_usage / self.memory_limit_mb) * 100
                }
            except Exception as e:
                health_status["components"]["memory"] = {
                    "status": "error",
                    "error": str(e)
                }
            
            # Determinar estado general
            component_statuses = [comp.get("status", "") for comp in health_status["components"].values()]
            if "error" in component_statuses:
                health_status["status"] = "error"
            elif "warning" in component_statuses:
                health_status["status"] = "warning"
            elif "not_loaded" in component_statuses or "not_initialized" in component_statuses:
                health_status["status"] = "not_ready"
            else:
                health_status["status"] = "healthy"
            
            return health_status
        
        except Exception as e:
            logger.error(f"Error durante la verificación de salud: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            health_status["status"] = "error"
            health_status["error"] = str(e)
            return health_status
    
    def cleanup(self) -> None:
        """Libera recursos y limpia la memoria."""
        try:
            # Limpiar memoria de TensorFlow
            cleanup_tensorflow_memory()
            
            # Liberar referencias
            self.model = None
            self.data_processor = None
            
            # Forzar recolección de basura
            import gc
            gc.collect()
            
            logger.info("Recursos liberados correctamente")
        except Exception as e:
            logger.error(f"Error al liberar recursos: {str(e)}")

# Función para crear una instancia de integración cloud
def create_cloud_lstm_integration(
    model_name: str = "lstm_model",
    base_dir: Optional[str] = None,
    **kwargs
) -> CloudLSTMIntegration:
    """
    Crea una instancia de integración LSTM para cloud.
    
    Args:
        model_name: Nombre del modelo
        base_dir: Directorio base (si es None, usa DATA_DIR o directorio temporal)
        **kwargs: Argumentos adicionales para CloudLSTMIntegration
        
    Returns:
        Instancia de CloudLSTMIntegration
    """
    # Determinar directorio base
    if base_dir is None:
        base_dir = os.environ.get("MODELS_DIR", "/tmp/models")
    
    # Construir ruta al modelo
    model_path = os.path.join(base_dir, model_name)
    
    # Crear y devolver instancia
    integration = CloudLSTMIntegration(model_path=model_path, **kwargs)
    return integration

# Si se ejecuta como script principal, realizar una verificación de salud
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Crear integración
    integration = create_cloud_lstm_integration()
    
    # Cargar modelo y procesador
    if integration.load_model_and_processor():
        # Realizar verificación de salud
        health = integration.health_check()
        print(json.dumps(health, indent=2))
    else:
        print("Error al cargar modelo y procesador")

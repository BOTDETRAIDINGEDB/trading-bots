#!/usr/bin/env python3
"""
Módulo de configuración de TensorFlow para entornos Google Cloud.
Optimiza el rendimiento y la estabilidad de los modelos en la nube.
"""

import os
import logging
import tensorflow as tf
import numpy as np
import traceback
from typing import Dict, Any, Optional

# Configurar logging
logger = logging.getLogger(__name__)

def configure_tensorflow_for_cloud(
    memory_growth: bool = True,
    memory_limit_mb: Optional[int] = None,
    deterministic: bool = False,
    mixed_precision: bool = True,
    log_device_placement: bool = False
) -> Dict[str, Any]:
    """
    Configura TensorFlow para un rendimiento óptimo en Google Cloud.
    
    Args:
        memory_growth: Si es True, permite el crecimiento gradual de memoria en GPU
        memory_limit_mb: Límite de memoria en MB para GPU (None = sin límite)
        deterministic: Si es True, hace que las operaciones sean deterministas
        mixed_precision: Si es True, habilita precisión mixta para acelerar entrenamiento
        log_device_placement: Si es True, registra la ubicación de las operaciones
        
    Returns:
        Diccionario con el estado de la configuración
    """
    status = {
        "success": True,
        "gpu_available": False,
        "memory_growth": False,
        "memory_limit": None,
        "deterministic": False,
        "mixed_precision": False,
        "errors": []
    }
    
    try:
        # Configurar logging de TensorFlow
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
        tf.get_logger().setLevel('WARNING')
        
        # Verificar versión de TensorFlow
        logger.info(f"TensorFlow versión: {tf.__version__}")
        status["tf_version"] = tf.__version__
        
        # Configurar dispositivos
        gpus = tf.config.list_physical_devices('GPU')
        status["gpu_available"] = len(gpus) > 0
        status["gpu_count"] = len(gpus)
        
        if gpus:
            logger.info(f"GPUs disponibles: {len(gpus)}")
            
            # Configurar memory growth
            if memory_growth:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    status["memory_growth"] = True
                    logger.info("Memory growth habilitado para todas las GPUs")
                except Exception as e:
                    status["errors"].append(f"Error al configurar memory growth: {str(e)}")
                    logger.warning(f"Error al configurar memory growth: {str(e)}")
            
            # Configurar límite de memoria
            if memory_limit_mb is not None:
                try:
                    tf.config.set_logical_device_configuration(
                        gpus[0],
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
                    )
                    status["memory_limit"] = memory_limit_mb
                    logger.info(f"Límite de memoria configurado: {memory_limit_mb} MB")
                except Exception as e:
                    status["errors"].append(f"Error al configurar límite de memoria: {str(e)}")
                    logger.warning(f"Error al configurar límite de memoria: {str(e)}")
        else:
            logger.info("No se detectaron GPUs. Usando CPU.")
        
        # Configurar determinismo
        if deterministic:
            try:
                os.environ['TF_DETERMINISTIC_OPS'] = '1'
                os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
                tf.random.set_seed(42)
                np.random.seed(42)
                
                # Limitar paralelismo para determinismo
                tf.config.threading.set_inter_op_parallelism_threads(1)
                tf.config.threading.set_intra_op_parallelism_threads(1)
                
                status["deterministic"] = True
                logger.info("Modo determinista habilitado")
            except Exception as e:
                status["errors"].append(f"Error al configurar modo determinista: {str(e)}")
                logger.warning(f"Error al configurar modo determinista: {str(e)}")
        
        # Configurar precisión mixta
        if mixed_precision and status["gpu_available"]:
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                status["mixed_precision"] = True
                logger.info("Precisión mixta habilitada")
            except Exception as e:
                status["errors"].append(f"Error al configurar precisión mixta: {str(e)}")
                logger.warning(f"Error al configurar precisión mixta: {str(e)}")
        
        # Configurar log de ubicación de dispositivos
        if log_device_placement:
            try:
                tf.debugging.set_log_device_placement(True)
                logger.info("Log de ubicación de dispositivos habilitado")
            except Exception as e:
                status["errors"].append(f"Error al configurar log de dispositivos: {str(e)}")
                logger.warning(f"Error al configurar log de dispositivos: {str(e)}")
        
        # Configurar optimizaciones XLA
        try:
            tf.config.optimizer.set_jit(True)  # Enable XLA (Accelerated Linear Algebra)
            logger.info("Optimizaciones XLA habilitadas")
        except Exception as e:
            status["errors"].append(f"Error al configurar XLA: {str(e)}")
            logger.warning(f"Error al configurar XLA: {str(e)}")
        
        # Verificar si hay errores
        if status["errors"]:
            status["success"] = False
        
        return status
        
    except Exception as e:
        logger.error(f"Error al configurar TensorFlow: {str(e)}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        status["success"] = False
        status["errors"].append(str(e))
        return status

def get_tensorflow_memory_usage() -> Dict[str, Any]:
    """
    Obtiene información sobre el uso de memoria de TensorFlow.
    
    Returns:
        Diccionario con información de uso de memoria
    """
    try:
        memory_info = {}
        
        # Obtener información de memoria de TensorFlow
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # En TF 2.x no hay una API directa para obtener el uso de memoria
            # Podemos usar la API experimental
            try:
                memory_info["gpu_memory_info"] = tf.config.experimental.get_memory_info('GPU:0')
            except Exception:
                memory_info["gpu_memory_info"] = "No disponible"
        
        # Obtener información de memoria del sistema
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info["process_memory_mb"] = process.memory_info().rss / (1024 * 1024)
            memory_info["system_memory_percent"] = psutil.virtual_memory().percent
        except ImportError:
            memory_info["process_memory"] = "psutil no disponible"
        
        return memory_info
    
    except Exception as e:
        logger.error(f"Error al obtener información de memoria: {str(e)}")
        return {"error": str(e)}

def cleanup_tensorflow_memory():
    """
    Limpia la memoria utilizada por TensorFlow.
    Útil después de entrenar o hacer predicciones con modelos grandes.
    """
    try:
        # Limpiar sesión de Keras
        tf.keras.backend.clear_session()
        
        # Forzar recolección de basura
        import gc
        gc.collect()
        
        logger.info("Memoria de TensorFlow liberada")
        return True
    except Exception as e:
        logger.error(f"Error al limpiar memoria de TensorFlow: {str(e)}")
        return False

def create_tensorflow_session_config(
    allow_growth: bool = True,
    per_process_gpu_memory_fraction: float = 0.8,
    log_device_placement: bool = False
) -> tf.compat.v1.ConfigProto:
    """
    Crea una configuración de sesión para TensorFlow 1.x (compatible con TF 2.x).
    
    Args:
        allow_growth: Si es True, permite el crecimiento gradual de memoria
        per_process_gpu_memory_fraction: Fracción de memoria GPU a utilizar
        log_device_placement: Si es True, registra la ubicación de las operaciones
        
    Returns:
        Configuración de sesión de TensorFlow
    """
    try:
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = allow_growth
        config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
        config.log_device_placement = log_device_placement
        return config
    except Exception as e:
        logger.error(f"Error al crear configuración de sesión: {str(e)}")
        return None

def configure_from_environment():
    """
    Configura TensorFlow basado en variables de entorno.
    
    Variables de entorno soportadas:
    - TF_MEMORY_GROWTH: "true" o "false"
    - TF_MEMORY_LIMIT_MB: límite de memoria en MB
    - TF_DETERMINISTIC: "true" o "false"
    - TF_MIXED_PRECISION: "true" o "false"
    - TF_LOG_DEVICE_PLACEMENT: "true" o "false"
    
    Returns:
        Estado de la configuración
    """
    # Obtener configuración de variables de entorno
    memory_growth = os.environ.get('TF_MEMORY_GROWTH', 'true').lower() == 'true'
    
    memory_limit_mb = None
    if 'TF_MEMORY_LIMIT_MB' in os.environ:
        try:
            memory_limit_mb = int(os.environ.get('TF_MEMORY_LIMIT_MB'))
        except ValueError:
            logger.warning("Valor inválido para TF_MEMORY_LIMIT_MB. Usando None.")
    
    deterministic = os.environ.get('TF_DETERMINISTIC', 'false').lower() == 'true'
    mixed_precision = os.environ.get('TF_MIXED_PRECISION', 'true').lower() == 'true'
    log_device_placement = os.environ.get('TF_LOG_DEVICE_PLACEMENT', 'false').lower() == 'true'
    
    # Configurar TensorFlow
    return configure_tensorflow_for_cloud(
        memory_growth=memory_growth,
        memory_limit_mb=memory_limit_mb,
        deterministic=deterministic,
        mixed_precision=mixed_precision,
        log_device_placement=log_device_placement
    )

# Si se ejecuta como script principal, configurar desde variables de entorno
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    status = configure_from_environment()
    print("Estado de configuración de TensorFlow:")
    for key, value in status.items():
        print(f"  {key}: {value}")

#!/usr/bin/env python3
"""
Módulo de utilidades para garantizar la compatibilidad con Google Cloud VM.
Proporciona funciones para manejar limitaciones de recursos y optimizar el rendimiento.
"""

import os
import sys
import gc
import json
import logging
import traceback
from datetime import datetime
import time

# Configurar logging
logger = logging.getLogger(__name__)

def is_cloud_environment():
    """
    Detecta si el código se está ejecutando en un entorno de Google Cloud VM.
    
    Returns:
        bool: True si se está ejecutando en Google Cloud VM, False en caso contrario
    """
    return os.environ.get('CLOUD_ENV', 'false').lower() == 'true'

def get_memory_limit():
    """
    Obtiene el límite de memoria configurado para el entorno.
    
    Returns:
        float: Límite de memoria en MB, 0 si no está configurado
    """
    try:
        return float(os.environ.get('MEMORY_LIMIT_MB', '0'))
    except (ValueError, TypeError):
        logger.warning("Valor inválido para MEMORY_LIMIT_MB, usando 0")
        return 0.0

def get_memory_usage():
    """
    Obtiene el uso actual de memoria en MB.
    
    Returns:
        float: Uso de memoria en MB, 0 si no se puede determinar
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)
    except ImportError:
        logger.warning("No se pudo importar psutil para monitorear memoria")
        return 0.0
    except Exception as e:
        logger.warning(f"Error al obtener uso de memoria: {str(e)}")
        return 0.0

def cleanup_memory():
    """
    Limpia la memoria y el backend de TensorFlow.
    """
    try:
        # Forzar recolección de basura
        gc.collect()
        
        # Limpiar sesión de TensorFlow si está disponible
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            
            # En entornos con GPU, intentar liberar memoria GPU
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except Exception as e:
                        logger.warning(f"Error al configurar memory growth para GPU: {str(e)}")
        except ImportError:
            logger.debug("TensorFlow no está disponible para limpiar sesión")
        except Exception as e:
            logger.warning(f"Error al limpiar sesión de TensorFlow: {str(e)}")
    except Exception as e:
        logger.warning(f"Error al limpiar memoria: {str(e)}")

def optimize_tensorflow_for_cloud():
    """
    Optimiza la configuración de TensorFlow para entornos de Google Cloud VM.
    """
    if not is_cloud_environment():
        return
    
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
            import numpy as np
            tf.random.set_seed(42)
            np.random.seed(42)
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(1)
        
        # Configurar logging de TensorFlow
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # INFO y WARNING solamente
        
        logger.info("TensorFlow optimizado para entorno cloud")
    except ImportError:
        logger.warning("No se pudo importar TensorFlow para optimización")
    except Exception as e:
        logger.warning(f"Error al optimizar TensorFlow: {str(e)}")

def ensure_directory_exists(directory_path):
    """
    Crea un directorio si no existe, con manejo de errores.
    
    Args:
        directory_path: Ruta del directorio a crear
        
    Returns:
        bool: True si el directorio existe o fue creado, False en caso contrario
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error al crear directorio {directory_path}: {str(e)}")
        return False

def safe_json_dump(data, file_path, indent=4):
    """
    Guarda datos en formato JSON con manejo de errores.
    
    Args:
        data: Datos a guardar
        file_path: Ruta del archivo
        indent: Indentación para el JSON
        
    Returns:
        bool: True si se guardó correctamente, False en caso contrario
    """
    try:
        directory = os.path.dirname(file_path)
        ensure_directory_exists(directory)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent, default=lambda o: o.isoformat() if isinstance(o, datetime) else None)
        return True
    except Exception as e:
        logger.error(f"Error al guardar JSON en {file_path}: {str(e)}")
        return False

def safe_json_load(file_path, default=None):
    """
    Carga datos en formato JSON con manejo de errores.
    
    Args:
        file_path: Ruta del archivo
        default: Valor por defecto si no se puede cargar
        
    Returns:
        Datos cargados o valor por defecto
    """
    try:
        if not os.path.exists(file_path):
            return default
        
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error al cargar JSON desde {file_path}: {str(e)}")
        return default

def retry_operation(operation_func, max_retries=3, retry_delay=1.0, 
                   error_types=(Exception,), cleanup_func=None):
    """
    Ejecuta una operación con reintentos automáticos.
    
    Args:
        operation_func: Función a ejecutar
        max_retries: Número máximo de reintentos
        retry_delay: Tiempo de espera entre reintentos (segundos)
        error_types: Tipos de errores que activarán un reintento
        cleanup_func: Función de limpieza a ejecutar antes de cada reintento
        
    Returns:
        Resultado de la operación o None si falló
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # Ejecutar función de limpieza si está definida
            if cleanup_func and attempt > 0:
                cleanup_func()
            
            # Ejecutar operación
            result = operation_func()
            return result
            
        except error_types as e:
            last_error = e
            logger.warning(f"Intento {attempt+1}/{max_retries} fallido: {str(e)}")
            
            # Esperar antes del siguiente intento
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    # Si llegamos aquí, todos los intentos fallaron
    logger.error(f"Operación fallida después de {max_retries} intentos. Último error: {str(last_error)}")
    return None

# Inicialización del módulo
logger.info("Módulo de compatibilidad con Google Cloud VM cargado")
if is_cloud_environment():
    logger.info(f"Detectado entorno de Google Cloud VM (Límite de memoria: {get_memory_limit()} MB)")
    optimize_tensorflow_for_cloud()

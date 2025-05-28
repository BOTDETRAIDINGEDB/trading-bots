#!/usr/bin/env python3
"""
Utilidades para compatibilidad con entornos cloud y mejora de robustez.
Proporciona funciones para manejo de errores, configuración y recursos.
"""

import os
import sys
import time
import json
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Any, Callable, Optional, Union
import numpy as np
import pandas as pd
import psutil
import functools
from retrying import retry

# Configurar logging básico
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Intentar configurar Google Cloud Logging si está disponible
try:
    import google.cloud.logging
    from google.cloud.logging.handlers import CloudLoggingHandler
    
    client = google.cloud.logging.Client()
    cloud_handler = CloudLoggingHandler(client, name="deep_learning_bot")
    
    # Configurar logger para Google Cloud
    cloud_logger = logging.getLogger()
    cloud_logger.setLevel(logging.INFO)
    cloud_logger.addHandler(cloud_handler)
    
    logger.info("Google Cloud Logging configurado correctamente")
    CLOUD_LOGGING_ENABLED = True
except (ImportError, Exception) as e:
    logger.info(f"Google Cloud Logging no disponible: {str(e)}")
    CLOUD_LOGGING_ENABLED = False

def ensure_directory_exists(directory_path: str) -> bool:
    """
    Asegura que un directorio exista, creándolo si es necesario.
    
    Args:
        directory_path: Ruta del directorio a verificar/crear
        
    Returns:
        True si el directorio existe o fue creado, False en caso de error
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        logger.debug(f"Directorio asegurado: {directory_path}")
        return True
    except Exception as e:
        logger.error(f"Error al crear directorio {directory_path}: {str(e)}")
        return False

def load_config_from_env(prefix: str = "BOT_") -> Dict[str, str]:
    """
    Carga configuración desde variables de entorno.
    
    Args:
        prefix: Prefijo para filtrar variables de entorno
        
    Returns:
        Diccionario con configuración
    """
    config = {}
    
    # Cargar todas las variables de entorno con el prefijo
    for key, value in os.environ.items():
        if key.startswith(prefix):
            config_key = key[len(prefix):].lower()
            config[config_key] = value
    
    # Valores por defecto para claves importantes
    defaults = {
        "models_dir": "models/deep_learning",
        "data_dir": "data",
        "log_level": "INFO",
        "max_retries": "3",
        "retry_delay": "5"
    }
    
    # Aplicar valores por defecto si no están en las variables de entorno
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
    
    logger.debug(f"Configuración cargada desde variables de entorno: {len(config)} valores")
    return config

def retry_operation(max_attempts: int = 3, delay: int = 5, 
                   retry_on_exceptions: tuple = (ConnectionError, TimeoutError)) -> Callable:
    """
    Decorador para reintentar operaciones propensas a fallos temporales.
    
    Args:
        max_attempts: Número máximo de intentos
        delay: Tiempo de espera entre intentos en segundos
        retry_on_exceptions: Tupla de excepciones que provocan reintento
        
    Returns:
        Decorador configurado
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except retry_on_exceptions as e:
                    attempts += 1
                    logger.warning(f"Error temporal en {func.__name__} (intento {attempts}/{max_attempts}): {str(e)}")
                    if attempts < max_attempts:
                        time.sleep(delay)
                    else:
                        logger.error(f"Operación {func.__name__} falló después de {max_attempts} intentos")
                        raise
                except Exception as e:
                    logger.error(f"Error no recuperable en {func.__name__}: {str(e)}")
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                    raise
            return None
        return wrapper
    return decorator

# Versión alternativa usando la biblioteca retrying
def retry_with_backoff(stop_max_attempt_number=3, wait_exponential_multiplier=1000, 
                      wait_exponential_max=10000):
    """
    Decorador para reintentar operaciones con backoff exponencial.
    
    Args:
        stop_max_attempt_number: Número máximo de intentos
        wait_exponential_multiplier: Multiplicador para espera exponencial (ms)
        wait_exponential_max: Tiempo máximo de espera (ms)
        
    Returns:
        Decorador configurado
    """
    def should_retry(exception):
        """Determina si se debe reintentar basado en la excepción."""
        retry_exceptions = (ConnectionError, TimeoutError, OSError)
        should_retry = isinstance(exception, retry_exceptions)
        if should_retry:
            logger.warning(f"Reintentando operación debido a: {str(exception)}")
        return should_retry
    
    return retry(
        retry_on_exception=should_retry,
        stop_max_attempt_number=stop_max_attempt_number,
        wait_exponential_multiplier=wait_exponential_multiplier,
        wait_exponential_max=wait_exponential_max
    )

def monitor_resources() -> Dict[str, float]:
    """
    Monitorea el uso de recursos del sistema.
    
    Returns:
        Diccionario con métricas de recursos
    """
    try:
        # Obtener información de CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Obtener información de memoria
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Obtener información de disco
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        # Obtener información de proceso actual
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        resources = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'disk_percent': disk_percent,
            'process_memory_mb': process_memory,
            'timestamp': datetime.now().isoformat()
        }
        
        # Registrar si los recursos están cerca de límites críticos
        if cpu_percent > 90 or memory_percent > 90 or disk_percent > 90:
            logger.warning(f"Uso crítico de recursos: CPU={cpu_percent}%, MEM={memory_percent}%, DISK={disk_percent}%")
        
        return resources
    
    except Exception as e:
        logger.error(f"Error al monitorear recursos: {str(e)}")
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def json_serializable(obj: Any) -> Any:
    """
    Convierte objetos a formato serializable para JSON.
    
    Args:
        obj: Objeto a convertir
        
    Returns:
        Versión serializable del objeto
        
    Raises:
        TypeError: Si el objeto no puede ser serializado
    """
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, set):
        return list(obj)
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    raise TypeError(f"Objeto de tipo {type(obj)} no es serializable")

def safe_json_dump(data: Any, file_path: str, indent: int = 4) -> bool:
    """
    Guarda datos en formato JSON de manera segura.
    
    Args:
        data: Datos a guardar
        file_path: Ruta del archivo
        indent: Indentación para el archivo JSON
        
    Returns:
        True si se guardó correctamente, False en caso contrario
    """
    try:
        # Asegurar que el directorio existe
        directory = os.path.dirname(file_path)
        ensure_directory_exists(directory)
        
        # Guardar con manejo de tipos especiales
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent, default=json_serializable)
        
        logger.debug(f"Datos guardados correctamente en {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error al guardar JSON en {file_path}: {str(e)}")
        return False

def safe_json_load(file_path: str, default: Any = None) -> Any:
    """
    Carga datos en formato JSON de manera segura.
    
    Args:
        file_path: Ruta del archivo
        default: Valor por defecto si hay error
        
    Returns:
        Datos cargados o valor por defecto
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"Archivo no encontrado: {file_path}")
            return default
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        logger.debug(f"Datos cargados correctamente desde {file_path}")
        return data
    
    except Exception as e:
        logger.error(f"Error al cargar JSON desde {file_path}: {str(e)}")
        return default

def get_absolute_path(relative_path: str) -> str:
    """
    Convierte una ruta relativa a absoluta basada en la ubicación del script.
    
    Args:
        relative_path: Ruta relativa
        
    Returns:
        Ruta absoluta
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, relative_path)

def setup_logging(log_file: str = None, log_level: str = "INFO") -> None:
    """
    Configura el sistema de logging.
    
    Args:
        log_file: Ruta del archivo de log (opcional)
        log_level: Nivel de logging
    """
    # Convertir string de nivel a constante de logging
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configuración básica
    logging_config = {
        'level': numeric_level,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'handlers': [logging.StreamHandler()]
    }
    
    # Añadir handler de archivo si se especifica
    if log_file:
        # Asegurar que el directorio existe
        log_dir = os.path.dirname(log_file)
        ensure_directory_exists(log_dir)
        
        # Añadir file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging_config['handlers'].append(file_handler)
    
    # Aplicar configuración
    logging.basicConfig(**logging_config)
    
    logger.info(f"Logging configurado con nivel {log_level}")

# Ejemplo de uso
if __name__ == "__main__":
    setup_logging(log_file="logs/cloud_utils.log", log_level="DEBUG")
    logger.info("Módulo de utilidades cloud cargado correctamente")
    
    # Probar monitoreo de recursos
    resources = monitor_resources()
    logger.info(f"Recursos del sistema: {resources}")
    
    # Probar carga de configuración
    config = load_config_from_env()
    logger.info(f"Configuración cargada: {config}")

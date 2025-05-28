#!/usr/bin/env python3
"""
Módulo para optimizar el rendimiento en Google Cloud VM.
Proporciona funciones para manejar limitaciones de recursos y mejorar la estabilidad.
"""

import os
import sys
import gc
import logging
import traceback
import time
from datetime import datetime

# Configurar logging
logger = logging.getLogger(__name__)

def is_cloud_environment():
    """Detecta si estamos en un entorno de Google Cloud VM."""
    return os.environ.get('CLOUD_ENV', 'false').lower() == 'true'

def get_memory_limit():
    """Obtiene el límite de memoria configurado."""
    try:
        return float(os.environ.get('MEMORY_LIMIT_MB', '0'))
    except (ValueError, TypeError):
        logger.warning("Valor inválido para MEMORY_LIMIT_MB, usando 0")
        return 0.0

def get_memory_usage():
    """Obtiene el uso actual de memoria en MB."""
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
    """Limpia la memoria y el backend de TensorFlow."""
    try:
        # Forzar recolección de basura
        gc.collect()
        
        # Limpiar sesión de TensorFlow si está disponible
        try:
            import tensorflow as tf
            if hasattr(tf, 'keras'):
                tf.keras.backend.clear_session()
            logger.info("Sesión de TensorFlow limpiada")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Error al limpiar sesión de TensorFlow: {str(e)}")
        
        return True
    except Exception as e:
        logger.error(f"Error al limpiar memoria: {str(e)}")
        return False

def optimize_for_cloud():
    """Optimiza la configuración para Google Cloud VM."""
    if not is_cloud_environment():
        logger.info("No estamos en un entorno cloud, no se aplicarán optimizaciones")
        return False
    
    logger.info("Aplicando optimizaciones para Google Cloud VM")
    
    # Configurar TensorFlow para usar menos memoria
    try:
        import tensorflow as tf
        
        # Limitar crecimiento de memoria GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"GPU configurada para crecimiento de memoria: {gpu}")
                except RuntimeError as e:
                    logger.warning(f"Error al configurar GPU {gpu}: {str(e)}")
        else:
            logger.info("No se detectaron GPUs, usando CPU para entrenamiento")
            # Configuraciones específicas para CPU
            os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Optimizaciones para CPU
        
        # Configurar para modo determinista si está habilitado
        if os.environ.get('TF_DETERMINISTIC', 'false').lower() == 'true':
            try:
                tf.config.experimental.enable_op_determinism()
                logger.info("Modo determinista de TensorFlow activado")
            except Exception as e:
                logger.warning(f"Error al activar modo determinista: {str(e)}")
                # Intentar método alternativo para versiones anteriores de TensorFlow
                try:
                    os.environ['TF_DETERMINISTIC_OPS'] = '1'
                    tf.random.set_seed(42)
                    import numpy as np
                    np.random.seed(42)
                    logger.info("Modo determinista configurado mediante variables de entorno")
                except Exception as e2:
                    logger.warning(f"Error al configurar modo determinista alternativo: {str(e2)}")
        
        # Desactivar multiprocesamiento si está configurado
        if os.environ.get('USE_MULTIPROCESSING', 'true').lower() == 'false':
            os.environ['TF_NUM_INTEROP_THREADS'] = '1'
            os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
            logger.info("Multiprocesamiento desactivado")
        else:
            # Configurar número de hilos para optimizar rendimiento
            import multiprocessing
            num_cores = multiprocessing.cpu_count()
            os.environ['TF_NUM_INTEROP_THREADS'] = str(min(num_cores, 4))  # Limitar a 4 hilos máximo
            os.environ['TF_NUM_INTRAOP_THREADS'] = str(min(num_cores, 4))  # Limitar a 4 hilos máximo
            logger.info(f"Multiprocesamiento configurado con {min(num_cores, 4)} hilos")
        
        # Configurar opciones adicionales para optimizar rendimiento en cloud
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'  # Habilitar precisión mixta para mejor rendimiento
        
        # Configurar manejo de memoria
        memory_limit = get_memory_limit()
        if memory_limit > 0:
            # Limitar uso de memoria de TensorFlow
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    tf.config.set_logical_device_configuration(
                        gpus[0],
                        [tf.config.LogicalDeviceConfiguration(memory_limit=int(memory_limit * 0.8))]  # 80% del límite
                    )
                    logger.info(f"Límite de memoria GPU configurado a {int(memory_limit * 0.8)} MB")
                except Exception as e:
                    logger.warning(f"Error al configurar límite de memoria GPU: {str(e)}")
        
        logger.info("TensorFlow configurado para entorno cloud")
    except ImportError:
        logger.warning("TensorFlow no está disponible, no se pueden aplicar optimizaciones")
    except Exception as e:
        logger.error(f"Error al configurar TensorFlow: {str(e)}")
    
    # Configurar sistema para optimizar rendimiento
    try:
        # Intentar liberar memoria no utilizada
        cleanup_memory()
        
        # Configurar recolector de basura para ser más agresivo
        gc.set_threshold(100, 5, 5)  # Valores más bajos hacen que el GC sea más agresivo
        
        logger.info("Sistema configurado para optimizar rendimiento en cloud")
    except Exception as e:
        logger.warning(f"Error al configurar sistema: {str(e)}")
    
    return True

# Inicializar el módulo
logger.info("Módulo de optimización para Google Cloud VM cargado")
if is_cloud_environment():
    optimize_for_cloud()

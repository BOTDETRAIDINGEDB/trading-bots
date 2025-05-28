#!/usr/bin/env python3
"""
Módulo de inicialización para el paquete de deep learning.
Este archivo asegura que los módulos divididos funcionen correctamente juntos.
"""

import sys
import os
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DEEP_LEARNING")

# Asegurar que los métodos de los archivos auxiliares estén disponibles
try:
    # Importar métodos de lstm_model_part2.py
    from .lstm_model_part2 import _build_bilstm_model, _build_attention_model
    
    # Importar métodos de lstm_model_part3.py
    from .lstm_model_part3 import train
    
    # Registrar importaciones exitosas
    logger.info("Métodos auxiliares importados correctamente")
    
    # Agregar métodos a DeepTimeSeriesModel
    from .lstm_model import DeepTimeSeriesModel
    
    # Asignar métodos de lstm_model_part2.py a la clase DeepTimeSeriesModel
    DeepTimeSeriesModel._build_bilstm_model = _build_bilstm_model
    DeepTimeSeriesModel._build_attention_model = _build_attention_model
    
    # Asignar métodos de lstm_model_part3.py a la clase DeepTimeSeriesModel
    DeepTimeSeriesModel.train = train
    
    logger.info("Métodos asignados correctamente a DeepTimeSeriesModel")
    
except ImportError as e:
    logger.error(f"Error al importar métodos auxiliares: {str(e)}")
    logger.error("El modelo LSTM puede no funcionar correctamente")

# Importar componentes principales para uso externo
from .lstm_model import DeepTimeSeriesModel

# Importar optimizaciones para Google Cloud VM
try:
    from .cloud_optimizer import optimize_for_cloud, cleanup_memory, is_cloud_environment
except ImportError:
    # Definir funciones dummy si no se puede importar el optimizador
    def optimize_for_cloud():
        return False
    
    def cleanup_memory():
        import gc
        gc.collect()
        return True
    
    def is_cloud_environment():
        return False

# Versión del módulo
__version__ = '1.0.0'

# Aplicar optimizaciones para Google Cloud VM si estamos en ese entorno
if os.environ.get('CLOUD_ENV', 'false').lower() == 'true':
    try:
        optimize_for_cloud()
        logger.info("Optimizaciones para Google Cloud VM aplicadas automáticamente")
    except Exception as e:
        logger.warning(f"Error al aplicar optimizaciones automáticas: {str(e)}")
        
logger.info(f"Módulo deep_learning v{__version__} inicializado correctamente")

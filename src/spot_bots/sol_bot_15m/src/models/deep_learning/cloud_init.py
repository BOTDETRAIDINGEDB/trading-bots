#!/usr/bin/env python3
"""
Script de inicialización para Google Cloud VM.
Ejecutar este script antes de usar el bot en Google Cloud VM.
"""

import os
import sys
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CLOUD_INIT")

def check_dependencies():
    """Verifica que todas las dependencias necesarias estén instaladas."""
    dependencies = {
        'tensorflow': 'TensorFlow',
        'numpy': 'NumPy',
        'psutil': 'psutil (opcional para monitoreo de memoria)',
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            logger.info(f"Dependencia encontrada: {name}")
        except ImportError:
            if module == 'psutil':
                logger.warning(f"Dependencia opcional no encontrada: {name}")
            else:
                missing.append(name)
                logger.error(f"Dependencia requerida no encontrada: {name}")
    
    if missing:
        logger.warning(f"Faltan dependencias requeridas: {', '.join(missing)}")
        return False
    return True

def setup_cloud_environment():
    """Configura el entorno para Google Cloud VM."""
    # Verificar dependencias
    dependencies_ok = check_dependencies()
    if not dependencies_ok:
        logger.warning("Algunas dependencias requeridas no están instaladas. El rendimiento puede verse afectado.")
    
    # Establecer variables de entorno
    os.environ['CLOUD_ENV'] = 'true'
    os.environ['MEMORY_LIMIT_MB'] = '2048'  # Ajustar según la VM
    os.environ['TF_DETERMINISTIC'] = 'true'
    os.environ['USE_MULTIPROCESSING'] = 'false'
    
    # Configurar variables adicionales para TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reducir verbosidad de TensorFlow
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Permitir crecimiento gradual de memoria GPU
    
    logger.info("Variables de entorno configuradas para Google Cloud VM")
    
    # Importar y ejecutar optimizador
    try:
        from cloud_optimizer import optimize_for_cloud
        optimize_for_cloud()
        logger.info("Optimizaciones aplicadas para Google Cloud VM")
    except ImportError:
        logger.warning("No se pudo importar el optimizador para Google Cloud VM")
    except Exception as e:
        logger.error(f"Error al aplicar optimizaciones: {str(e)}")
    
    # Verificar disponibilidad de memoria
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"Memoria total disponible: {memory.total / (1024**3):.2f} GB")
        logger.info(f"Memoria disponible: {memory.available / (1024**3):.2f} GB")
        
        # Advertir si hay poca memoria disponible
        if memory.available < 1 * (1024**3):  # Menos de 1 GB disponible
            logger.warning("Poca memoria disponible. Considere aumentar la memoria de la VM o reducir MEMORY_LIMIT_MB.")
    except ImportError:
        logger.warning("No se pudo verificar la memoria disponible (psutil no está instalado)")
    except Exception as e:
        logger.warning(f"Error al verificar memoria: {str(e)}")
    
    return True

if __name__ == "__main__":
    logger.info("Iniciando configuración para Google Cloud VM...")
    setup_cloud_environment()
    logger.info("Configuración completada para Google Cloud VM")
    logger.info("")
    logger.info("El bot está listo para ejecutarse en Google Cloud VM")
    logger.info("Recuerde que estas configuraciones se aplican solo a la sesión actual")
    logger.info("Para configuración permanente, añada las variables de entorno al sistema")

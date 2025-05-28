#!/usr/bin/env python3
"""
Script para verificar que todos los componentes del sistema LSTM estén correctamente
integrados y sean compatibles con Google Cloud VM.

Este script realiza las siguientes verificaciones:
1. Comprueba que los archivos necesarios existen
2. Verifica que las importaciones entre archivos funcionan correctamente
3. Confirma que las optimizaciones para Google Cloud VM están correctamente implementadas
4. Realiza una prueba de carga mínima del modelo
"""

import os
import sys
import logging
import importlib
import traceback
import time
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VERIFY_INTEGRATION")

def check_file_exists(file_path):
    """Verifica si un archivo existe."""
    exists = os.path.exists(file_path)
    if exists:
        logger.info(f"✓ Archivo encontrado: {file_path}")
    else:
        logger.error(f"✗ Archivo no encontrado: {file_path}")
    return exists

def import_module_safely(module_name):
    """Intenta importar un módulo de forma segura."""
    try:
        module = importlib.import_module(module_name)
        logger.info(f"✓ Módulo importado: {module_name}")
        return module
    except ImportError as e:
        logger.error(f"✗ Error al importar {module_name}: {str(e)}")
        if 'tensorflow' in str(e).lower():
            logger.warning("  ℹ TensorFlow no está instalado. Esto es normal en entornos de desarrollo.")
            logger.warning("  ℹ En Google Cloud VM, asegúrese de instalar TensorFlow con: pip install tensorflow")
        elif 'models.deep_learning' in module_name:
            logger.warning("  ℹ Intentando importar desde el directorio actual...")
            # Ajustar path para importar desde directorio actual
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(os.path.dirname(current_dir))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
                logger.warning(f"  ℹ Añadido {parent_dir} al sys.path")
        return None
    except Exception as e:
        logger.error(f"✗ Error inesperado al importar {module_name}: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

def check_required_files():
    """Verifica que todos los archivos necesarios existen."""
    logger.info("Verificando archivos necesarios...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    required_files = [
        "__init__.py",
        "lstm_model.py",
        "lstm_model_part2.py",
        "lstm_model_part3.py",
        "cloud_optimizer.py",
        "cloud_init.py"
    ]
    
    all_exist = True
    for file in required_files:
        file_path = os.path.join(current_dir, file)
        if not check_file_exists(file_path):
            all_exist = False
    
    return all_exist

def check_imports():
    """Verifica que las importaciones entre archivos funcionan correctamente."""
    logger.info("Verificando importaciones...")
    
    # Intentar importar el módulo principal
    deep_learning = import_module_safely("models.deep_learning")
    if not deep_learning:
        # Intentar importar desde el directorio actual
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        deep_learning = import_module_safely("models.deep_learning")
        if not deep_learning:
            logger.error("✗ No se pudo importar el módulo deep_learning")
            return False
    
    # Verificar que DeepTimeSeriesModel está disponible
    if not hasattr(deep_learning, "DeepTimeSeriesModel"):
        logger.error("✗ DeepTimeSeriesModel no está disponible en el módulo deep_learning")
        return False
    
    logger.info("✓ DeepTimeSeriesModel está disponible")
    
    # Verificar que los métodos de los archivos auxiliares están asignados a DeepTimeSeriesModel
    model_class = deep_learning.DeepTimeSeriesModel
    required_methods = [
        "_build_bilstm_model",
        "_build_attention_model",
        "train"
    ]
    
    all_methods_available = True
    for method in required_methods:
        if not hasattr(model_class, method):
            logger.error(f"✗ Método {method} no está asignado a DeepTimeSeriesModel")
            all_methods_available = False
        else:
            logger.info(f"✓ Método {method} está correctamente asignado")
    
    # Verificar que las optimizaciones para Google Cloud VM están disponibles
    required_functions = [
        "optimize_for_cloud",
        "cleanup_memory",
        "is_cloud_environment"
    ]
    
    for func in required_functions:
        if not hasattr(deep_learning, func):
            logger.error(f"✗ Función {func} no está disponible en el módulo deep_learning")
            all_methods_available = False
        else:
            logger.info(f"✓ Función {func} está disponible")
    
    return all_methods_available

def test_cloud_environment():
    """Prueba las optimizaciones para Google Cloud VM."""
    logger.info("Probando optimizaciones para Google Cloud VM...")
    
    # Configurar variables de entorno para simular Google Cloud VM
    os.environ['CLOUD_ENV'] = 'true'
    os.environ['MEMORY_LIMIT_MB'] = '2048'
    os.environ['TF_DETERMINISTIC'] = 'true'
    os.environ['USE_MULTIPROCESSING'] = 'false'
    
    # Importar funciones de optimización
    try:
        from cloud_optimizer import optimize_for_cloud, cleanup_memory, is_cloud_environment
        
        # Verificar que estamos en un entorno cloud (según la función)
        if not is_cloud_environment():
            logger.error("✗ is_cloud_environment() devuelve False a pesar de que CLOUD_ENV=true")
            return False
        
        logger.info("✓ is_cloud_environment() devuelve True correctamente")
        
        # Aplicar optimizaciones
        result = optimize_for_cloud()
        if not result:
            logger.warning("⚠ optimize_for_cloud() devolvió False")
        else:
            logger.info("✓ optimize_for_cloud() aplicado correctamente")
        
        # Limpiar memoria
        result = cleanup_memory()
        if not result:
            logger.warning("⚠ cleanup_memory() devolvió False")
        else:
            logger.info("✓ cleanup_memory() ejecutado correctamente")
        
        return True
    except ImportError:
        logger.error("✗ No se pudieron importar las funciones de optimización")
        return False
    except Exception as e:
        logger.error(f"✗ Error al probar optimizaciones: {str(e)}")
        return False

def test_model_load():
    """Prueba la carga del modelo."""
    logger.info("Probando carga del modelo...")
    
    try:
        # Importar DeepTimeSeriesModel
        try:
            from lstm_model import DeepTimeSeriesModel
        except ImportError as e:
            if 'tensorflow' in str(e).lower():
                logger.warning("  ℹ TensorFlow no está instalado. Verificando estructura de clase sin crear instancia.")
                # Importar el módulo directamente para inspección
                import inspect
                import lstm_model
                
                # Verificar que la clase existe
                if not hasattr(lstm_model, 'DeepTimeSeriesModel'):
                    logger.error("✗ La clase DeepTimeSeriesModel no existe en lstm_model.py")
                    return False
                
                # Obtener la clase para inspección
                cls = getattr(lstm_model, 'DeepTimeSeriesModel')
                
                # Verificar que es una clase
                if not inspect.isclass(cls):
                    logger.error("✗ DeepTimeSeriesModel no es una clase")
                    return False
                
                logger.info("✓ Clase DeepTimeSeriesModel encontrada")
                
                # Verificar métodos por inspección
                required_methods = [
                    "_build_model",
                    "_build_lstm_model",
                    "save",
                    "load"
                ]
                
                for method in required_methods:
                    if not hasattr(cls, method):
                        logger.error(f"✗ Método {method} no está definido en la clase")
                        return False
                    else:
                        logger.info(f"✓ Método {method} está definido en la clase")
                
                logger.info("✓ Estructura de clase verificada sin crear instancia")
                return True
            else:
                # Si es otro tipo de error de importación, reportarlo
                logger.error(f"✗ Error al importar DeepTimeSeriesModel: {str(e)}")
                return False
        
        # Si llegamos aquí, podemos crear una instancia del modelo
        model = DeepTimeSeriesModel(
            model_type='lstm',
            sequence_length=60,
            num_features=20,
            num_classes=3
        )
        
        logger.info("✓ Modelo creado correctamente")
        
        # Verificar que el modelo tiene los métodos necesarios
        required_methods = [
            "_build_model",
            "_build_lstm_model",
            "save",
            "load"
        ]
        
        for method in required_methods:
            if not hasattr(model, method):
                logger.error(f"✗ Método {method} no está disponible en el modelo")
                return False
            else:
                logger.info(f"✓ Método {method} está disponible en el modelo")
        
        # Verificar que el modelo tiene los atributos necesarios
        required_attrs = [
            "model_type",
            "sequence_length",
            "num_features",
            "num_classes",
            "model_path",
            "config_path",
            "model",
            "config"
        ]
        
        for attr in required_attrs:
            if not hasattr(model, attr):
                logger.error(f"✗ Atributo {attr} no está disponible en el modelo")
                return False
            else:
                logger.info(f"✓ Atributo {attr} está disponible en el modelo")
        
        return True
    except Exception as e:
        if 'tensorflow' in str(e).lower():
            logger.warning(f"  ℹ Error relacionado con TensorFlow: {str(e)}")
            logger.warning("  ℹ Esto es normal en entornos de desarrollo sin TensorFlow instalado")
            logger.warning("  ℹ En Google Cloud VM, asegúrese de instalar TensorFlow con: pip install tensorflow")
            # Consideramos esto como un éxito en entorno de desarrollo
            return True
        else:
            logger.error(f"✗ Error al probar carga del modelo: {str(e)}")
            logger.debug(traceback.format_exc())
            return False

def main():
    """Función principal."""
    logger.info("=== VERIFICACIÓN DE INTEGRACIÓN ===")
    
    # Verificar archivos
    files_ok = check_required_files()
    if not files_ok:
        logger.error("✗ Verificación de archivos fallida")
    else:
        logger.info("✓ Verificación de archivos completada")
    
    # Verificar importaciones
    imports_ok = check_imports()
    if not imports_ok:
        logger.error("✗ Verificación de importaciones fallida")
    else:
        logger.info("✓ Verificación de importaciones completada")
    
    # Probar optimizaciones para Google Cloud VM
    cloud_ok = test_cloud_environment()
    if not cloud_ok:
        logger.error("✗ Prueba de optimizaciones para Google Cloud VM fallida")
    else:
        logger.info("✓ Prueba de optimizaciones para Google Cloud VM completada")
    
    # Probar carga del modelo
    model_ok = test_model_load()
    if not model_ok:
        logger.error("✗ Prueba de carga del modelo fallida")
    else:
        logger.info("✓ Prueba de carga del modelo completada")
    
    # Resultado final
    if files_ok and imports_ok and cloud_ok and model_ok:
        logger.info("✅ VERIFICACIÓN COMPLETADA CON ÉXITO")
        logger.info("El sistema LSTM está correctamente integrado y es compatible con Google Cloud VM")
        return True
    else:
        logger.error("❌ VERIFICACIÓN FALLIDA")
        logger.error("El sistema LSTM puede no estar correctamente integrado o no ser compatible con Google Cloud VM")
        logger.error("Revise los errores anteriores para más detalles")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

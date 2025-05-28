#!/usr/bin/env python3
"""
Script de integración para los modelos LSTM.
Este script asegura que todos los componentes trabajen juntos correctamente
y que el bot funcione en la máquina virtual de Google Cloud.
"""

import os
import sys
import logging
import importlib
import traceback
import shutil

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LSTM_INTEGRATION")

def check_tensorflow():
    """Verifica si TensorFlow está instalado correctamente."""
    try:
        import tensorflow as tf
        logger.info(f"TensorFlow {tf.__version__} detectado")
        return True
    except ImportError:
        logger.warning("TensorFlow no está instalado")
        return False
    except Exception as e:
        logger.error(f"Error al verificar TensorFlow: {str(e)}")
        return False

def check_file_exists(file_path):
    """Verifica si un archivo existe."""
    exists = os.path.exists(file_path)
    if exists:
        logger.info(f"Archivo encontrado: {file_path}")
    else:
        logger.error(f"Archivo no encontrado: {file_path}")
    return exists

def import_module_safely(module_name):
    """Intenta importar un módulo de forma segura."""
    try:
        module = importlib.import_module(module_name)
        logger.info(f"Módulo importado: {module_name}")
        return module
    except ImportError as e:
        logger.error(f"Error al importar {module_name}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error inesperado al importar {module_name}: {str(e)}")
        return None

def fix_imports():
    """Corrige las importaciones en los archivos."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Archivos a corregir
    files_to_fix = {
        "lstm_model.py": [
            ("from .lstm_model_unified import *", "# Importaciones necesarias"),
            ("# Este archivo es un wrapper", "# Implementación principal")
        ],
        "lstm_model_part2.py": [
            ("def _build_bilstm_model", "def _build_bilstm_model_impl"),
            ("def _build_attention_model", "def _build_attention_model_impl")
        ],
        "lstm_model_part3.py": [
            ("def train", "def train_impl"),
            ("def _cleanup_memory", "def _cleanup_memory_impl")
        ]
    }
    
    for file_name, replacements in files_to_fix.items():
        file_path = os.path.join(current_dir, file_name)
        if not os.path.exists(file_path):
            logger.warning(f"Archivo no encontrado para corregir: {file_path}")
            continue
        
        # Crear copia de seguridad
        backup_path = f"{file_path}.integration_backup"
        if not os.path.exists(backup_path):
            shutil.copy2(file_path, backup_path)
            logger.info(f"Copia de seguridad creada: {backup_path}")
        
        # Leer contenido
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Aplicar reemplazos
        modified = False
        for old_text, new_text in replacements:
            if old_text in content:
                content = content.replace(old_text, new_text)
                modified = True
        
        # Guardar cambios si se modificó
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Archivo corregido: {file_path}")
        else:
            logger.info(f"No se requirieron cambios en: {file_path}")

def create_cloud_init():
    """Crea un archivo de inicialización para Google Cloud VM."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cloud_init_path = os.path.join(current_dir, "cloud_init.py")
    
    content = """#!/usr/bin/env python3
\"\"\"
Script de inicialización para Google Cloud VM.
Este script configura el entorno para el bot de trading.
\"\"\"

import os
import sys
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CLOUD_INIT")

def setup_cloud_environment():
    \"\"\"Configura el entorno para Google Cloud VM.\"\"\"
    # Establecer variables de entorno
    os.environ['CLOUD_ENV'] = 'true'
    os.environ['MEMORY_LIMIT_MB'] = '2048'  # Ajustar según la VM
    os.environ['TF_DETERMINISTIC'] = 'true'
    os.environ['USE_MULTIPROCESSING'] = 'false'
    
    logger.info("Variables de entorno configuradas para Google Cloud VM")
    
    # Configurar TensorFlow
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
        
        # Configurar para determinismo
        tf.random.set_seed(42)
        import numpy as np
        np.random.seed(42)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        
        logger.info("TensorFlow configurado para Google Cloud VM")
    except ImportError:
        logger.warning("TensorFlow no está disponible")
    except Exception as e:
        logger.error(f"Error al configurar TensorFlow: {str(e)}")
    
    return True

if __name__ == "__main__":
    logger.info("Iniciando configuración para Google Cloud VM...")
    setup_cloud_environment()
    logger.info("Configuración completada para Google Cloud VM")
"""
    
    with open(cloud_init_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"Archivo de inicialización para Google Cloud VM creado: {cloud_init_path}")

def create_integration_test():
    """Crea un script de prueba para verificar la integración."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(current_dir, "test_lstm_integration.py")
    
    content = """#!/usr/bin/env python3
\"\"\"
Script de prueba para verificar la integración de los modelos LSTM.
\"\"\"

import os
import sys
import logging
import traceback

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TEST_INTEGRATION")

def run_integration_test():
    \"\"\"Ejecuta pruebas de integración para los modelos LSTM.\"\"\"
    try:
        # Importar módulos
        logger.info("Importando módulos...")
        from lstm_model import DeepTimeSeriesModel
        
        # Crear instancia del modelo
        logger.info("Creando instancia del modelo...")
        model = DeepTimeSeriesModel(
            model_type='lstm',
            sequence_length=60,
            num_features=20,
            num_classes=3
        )
        
        # Verificar atributos
        logger.info("Verificando atributos del modelo...")
        required_attrs = [
            'model_type', 'sequence_length', 'num_features', 'num_classes',
            'model_path', 'config_path', 'model', 'config'
        ]
        
        for attr in required_attrs:
            if not hasattr(model, attr):
                logger.error(f"Atributo faltante: {attr}")
                return False
        
        logger.info("Todos los atributos requeridos están presentes")
        
        # Verificar métodos
        logger.info("Verificando métodos del modelo...")
        required_methods = [
            '_load_config', '_build_model', '_build_lstm_model', 'save', 'load',
            'train', 'predict', 'evaluate'
        ]
        
        for method in required_methods:
            if not hasattr(model, method) or not callable(getattr(model, method)):
                logger.error(f"Método faltante: {method}")
                return False
        
        logger.info("Todos los métodos requeridos están presentes")
        
        # Simulación de Google Cloud VM
        logger.info("Simulando entorno de Google Cloud VM...")
        os.environ['CLOUD_ENV'] = 'true'
        os.environ['MEMORY_LIMIT_MB'] = '2048'
        
        # Verificar carga de configuración
        logger.info("Verificando carga de configuración...")
        config = model._load_config()
        if not config:
            logger.error("Error al cargar configuración")
            return False
        
        logger.info("Configuración cargada correctamente")
        
        # Prueba exitosa
        logger.info("Prueba de integración completada con éxito")
        return True
        
    except Exception as e:
        logger.error(f"Error en prueba de integración: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("Iniciando prueba de integración...")
    success = run_integration_test()
    sys.exit(0 if success else 1)
"""
    
    with open(test_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"Script de prueba de integración creado: {test_path}")

def create_readme():
    """Crea un archivo README con instrucciones."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    readme_path = os.path.join(current_dir, "README.md")
    
    content = """# Módulo LSTM para Trading Bot

Este módulo implementa modelos LSTM/GRU para predicción de mercados financieros.

## Estructura de Archivos

- `lstm_model.py`: Clase principal DeepTimeSeriesModel
- `lstm_model_part2.py`: Implementación de arquitecturas de modelos
- `lstm_model_part3.py`: Métodos de entrenamiento y evaluación
- `cloud_compatibility.py`: Utilidades para compatibilidad con Google Cloud VM
- `cloud_init.py`: Script de inicialización para Google Cloud VM

## Configuración para Google Cloud VM

Para ejecutar el bot en una máquina virtual de Google Cloud:

1. Ejecute el script de inicialización:
   ```
   python cloud_init.py
   ```

2. Configure las siguientes variables de entorno:
   - `CLOUD_ENV=true`: Activa optimizaciones para entorno cloud
   - `MEMORY_LIMIT_MB=2048`: Límite de memoria en MB (ajustar según la VM)
   - `TF_DETERMINISTIC=true`: Activa modo determinista para reproducibilidad
   - `USE_MULTIPROCESSING=false`: Desactiva multiprocesamiento en entornos con recursos limitados

3. Verifique la integración:
   ```
   python test_lstm_integration.py
   ```

## Dependencias

- TensorFlow >= 2.4.0
- NumPy >= 1.19.0
- psutil >= 5.8.0 (opcional, para monitoreo de memoria)

## Optimizaciones para Google Cloud VM

Este módulo incluye las siguientes optimizaciones para entornos cloud:

- Manejo eficiente de memoria para evitar errores OOM
- Reintentos automáticos para operaciones críticas
- Monitoreo de recursos
- Configuración optimizada de TensorFlow
- Manejo robusto de errores

## Solución de Problemas

Si experimenta errores en Google Cloud VM:

1. Verifique que las variables de entorno estén configuradas correctamente
2. Aumente el valor de `MEMORY_LIMIT_MB` si hay errores de memoria
3. Ejecute `python cloud_init.py` para reiniciar la configuración
4. Verifique los logs para identificar errores específicos
"""
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"Archivo README creado: {readme_path}")

def main():
    """Función principal."""
    logger.info("Iniciando integración de modelos LSTM...")
    
    # Verificar TensorFlow
    check_tensorflow()
    
    # Verificar archivos críticos
    current_dir = os.path.dirname(os.path.abspath(__file__))
    critical_files = [
        "lstm_model.py",
        "lstm_model_part2.py",
        "lstm_model_part3.py",
        "__init__.py"
    ]
    
    all_files_exist = all(check_file_exists(os.path.join(current_dir, file)) for file in critical_files)
    
    if not all_files_exist:
        logger.error("Faltan archivos críticos. Abortando integración.")
        return False
    
    # Corregir importaciones
    fix_imports()
    
    # Crear archivos de soporte
    create_cloud_init()
    create_integration_test()
    create_readme()
    
    logger.info("")
    logger.info("=== INTEGRACIÓN COMPLETADA ===")
    logger.info("El bot ahora debería funcionar correctamente en la máquina virtual de Google Cloud.")
    logger.info("")
    logger.info("Para verificar la integración:")
    logger.info("1. Ejecute: python cloud_init.py")
    logger.info("2. Ejecute: python test_lstm_integration.py")
    logger.info("")
    logger.info("Consulte README.md para más información.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

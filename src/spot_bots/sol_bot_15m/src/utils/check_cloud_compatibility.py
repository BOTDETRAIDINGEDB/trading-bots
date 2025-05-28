#!/usr/bin/env python3
"""
Script para verificar la compatibilidad del proyecto con Google Cloud VM.

Este script realiza las siguientes verificaciones:
1. Comprueba que todos los archivos necesarios existen
2. Verifica que las importaciones entre archivos son correctas
3. Confirma que las configuraciones para Google Cloud VM están implementadas
4. Identifica posibles problemas de compatibilidad
5. Genera recomendaciones para optimizar el despliegue en la nube

Uso:
    python check_cloud_compatibility.py [--fix] [--report-file REPORT_FILE]

Opciones:
    --fix: Intenta corregir automáticamente los problemas encontrados
    --report-file: Ruta del archivo donde guardar el informe (formato JSON)
"""

import os
import sys
import json
import logging
import argparse
import importlib
import subprocess
import platform
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CLOUD_COMPATIBILITY")

# Archivos críticos que deben existir para el correcto funcionamiento en Google Cloud VM
CRITICAL_FILES = [
    "src/models/deep_learning/lstm_model.py",
    "src/models/deep_learning/lstm_model_part2.py",
    "src/models/deep_learning/lstm_model_part3.py",
    "src/models/deep_learning/__init__.py",
    "src/models/deep_learning/cloud_optimizer.py",
    "src/models/deep_learning/cloud_init.py",
    "credentials_template.json",
    "requirements.txt",
    "main.py",
    "start.sh",
    "stop.sh"
]

# Dependencias necesarias para Google Cloud VM
CLOUD_DEPENDENCIES = [
    "tensorflow>=2.4.0",
    "numpy>=1.19.0",
    "pandas>=1.1.0",
    "scikit-learn>=0.24.0",
    "matplotlib>=3.3.0",
    "psutil>=5.8.0",
    "requests>=2.25.0",
    "python-binance>=1.0.0",
    "python-telegram-bot>=13.0",
    "google-cloud-storage>=1.40.0",
    "google-cloud-secret-manager>=2.0.0"
]

# Variables de entorno necesarias para Google Cloud VM
CLOUD_ENV_VARS = [
    "CLOUD_ENV=true",
    "MEMORY_LIMIT_MB=2048",
    "TF_DETERMINISTIC=true",
    "USE_MULTIPROCESSING=false",
    "TF_CPP_MIN_LOG_LEVEL=2",
    "TF_FORCE_GPU_ALLOW_GROWTH=true"
]

def check_file_exists(file_path: str, project_root: str) -> bool:
    """Verifica si un archivo existe."""
    full_path = os.path.join(project_root, file_path)
    exists = os.path.exists(full_path)
    if exists:
        logger.info(f"✓ Archivo encontrado: {file_path}")
    else:
        logger.error(f"✗ Archivo no encontrado: {file_path}")
    return exists

def check_critical_files(project_root: str) -> Dict[str, bool]:
    """Verifica que todos los archivos críticos existen."""
    logger.info("Verificando archivos críticos...")
    results = {}
    
    for file_path in CRITICAL_FILES:
        results[file_path] = check_file_exists(file_path, project_root)
    
    return results

def check_dependencies(project_root: str) -> Dict[str, Dict[str, Any]]:
    """Verifica que todas las dependencias necesarias están en requirements.txt."""
    logger.info("Verificando dependencias...")
    results = {}
    
    requirements_path = os.path.join(project_root, "requirements.txt")
    if not os.path.exists(requirements_path):
        logger.error("✗ No se encontró el archivo requirements.txt")
        return {dep: {"found": False, "version": None} for dep in CLOUD_DEPENDENCIES}
    
    # Leer requirements.txt
    with open(requirements_path, 'r') as f:
        requirements = f.read().splitlines()
    
    # Normalizar requirements (eliminar comentarios y líneas vacías)
    requirements = [r.strip() for r in requirements if r.strip() and not r.strip().startswith('#')]
    
    # Verificar cada dependencia
    for dep in CLOUD_DEPENDENCIES:
        dep_name = dep.split('>=')[0].split('==')[0].strip()
        dep_version = dep.split('>=')[1] if '>=' in dep else None
        
        # Buscar la dependencia en requirements.txt
        found = False
        actual_version = None
        
        for req in requirements:
            req_name = req.split('>=')[0].split('==')[0].strip()
            if req_name.lower() == dep_name.lower():
                found = True
                if '>=' in req:
                    actual_version = req.split('>=')[1]
                elif '==' in req:
                    actual_version = req.split('==')[1]
                break
        
        if found:
            logger.info(f"✓ Dependencia encontrada: {dep_name} (versión: {actual_version})")
        else:
            logger.error(f"✗ Dependencia no encontrada: {dep_name}")
        
        results[dep_name] = {
            "found": found,
            "required_version": dep_version,
            "actual_version": actual_version
        }
    
    return results

def check_environment_variables(project_root: str) -> Dict[str, bool]:
    """Verifica que las variables de entorno necesarias están configuradas."""
    logger.info("Verificando variables de entorno...")
    results = {}
    
    # Buscar archivos que podrían contener configuraciones de variables de entorno
    env_files = [
        os.path.join(project_root, "src/models/deep_learning/cloud_init.py"),
        os.path.join(project_root, "src/models/deep_learning/cloud_optimizer.py"),
        os.path.join(project_root, "deploy_to_cloud.py"),
        os.path.join(project_root, "start.sh")
    ]
    
    # Leer contenido de los archivos
    env_content = ""
    for file_path in env_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                env_content += f.read()
    
    # Verificar cada variable de entorno
    for env_var in CLOUD_ENV_VARS:
        var_name = env_var.split('=')[0]
        found = var_name in env_content
        
        if found:
            logger.info(f"✓ Variable de entorno encontrada: {var_name}")
        else:
            logger.error(f"✗ Variable de entorno no encontrada: {var_name}")
        
        results[var_name] = found
    
    return results

def check_imports(project_root: str) -> Dict[str, bool]:
    """Verifica que las importaciones entre archivos son correctas."""
    logger.info("Verificando importaciones...")
    results = {}
    
    # Archivos a verificar
    files_to_check = [
        "src/models/deep_learning/__init__.py",
        "src/models/deep_learning/lstm_model.py",
        "src/models/deep_learning/lstm_model_part2.py",
        "src/models/deep_learning/lstm_model_part3.py"
    ]
    
    # Importaciones críticas a verificar
    critical_imports = {
        "src/models/deep_learning/__init__.py": [
            "from .lstm_model import DeepTimeSeriesModel",
            "from .cloud_optimizer import optimize_for_cloud"
        ],
        "src/models/deep_learning/lstm_model.py": [
            "import numpy as np",
            "import os",
            "import json"
        ],
        "src/models/deep_learning/lstm_model_part2.py": [
            "import numpy as np",
            "from tensorflow.keras.layers"
        ],
        "src/models/deep_learning/lstm_model_part3.py": [
            "import numpy as np",
            "from tensorflow.keras.callbacks"
        ]
    }
    
    # Verificar cada archivo
    for file_path in files_to_check:
        full_path = os.path.join(project_root, file_path)
        if not os.path.exists(full_path):
            logger.error(f"✗ No se puede verificar importaciones: {file_path} no existe")
            results[file_path] = False
            continue
        
        # Leer contenido del archivo
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Verificar importaciones críticas
        if file_path in critical_imports:
            all_imports_found = True
            for imp in critical_imports[file_path]:
                if imp not in content:
                    logger.error(f"✗ Importación no encontrada en {file_path}: {imp}")
                    all_imports_found = False
            
            if all_imports_found:
                logger.info(f"✓ Todas las importaciones críticas encontradas en {file_path}")
            
            results[file_path] = all_imports_found
    
    return results

def check_tensorflow_compatibility() -> Dict[str, Any]:
    """Verifica la compatibilidad con TensorFlow."""
    logger.info("Verificando compatibilidad con TensorFlow...")
    results = {
        "tensorflow_available": False,
        "gpu_available": False,
        "python_version_compatible": False,
        "system_compatible": False
    }
    
    # Verificar versión de Python
    python_version = platform.python_version()
    python_major, python_minor, _ = map(int, python_version.split('.'))
    python_compatible = (python_major == 3 and python_minor >= 7 and python_minor <= 10)
    
    results["python_version"] = python_version
    results["python_version_compatible"] = python_compatible
    
    if python_compatible:
        logger.info(f"✓ Versión de Python compatible: {python_version}")
    else:
        logger.error(f"✗ Versión de Python incompatible: {python_version} (se recomienda Python 3.7-3.10)")
    
    # Verificar sistema operativo
    system = platform.system()
    results["system"] = system
    results["system_compatible"] = system in ["Linux", "Windows"]
    
    if results["system_compatible"]:
        logger.info(f"✓ Sistema operativo compatible: {system}")
    else:
        logger.error(f"✗ Sistema operativo incompatible: {system}")
    
    # Intentar importar TensorFlow
    try:
        import tensorflow as tf
        results["tensorflow_available"] = True
        results["tensorflow_version"] = tf.__version__
        logger.info(f"✓ TensorFlow disponible: versión {tf.__version__}")
        
        # Verificar disponibilidad de GPU
        gpus = tf.config.list_physical_devices('GPU')
        results["gpu_available"] = len(gpus) > 0
        results["gpu_count"] = len(gpus)
        
        if results["gpu_available"]:
            logger.info(f"✓ GPU disponible: {len(gpus)} dispositivo(s)")
        else:
            logger.warning("⚠ No se detectaron GPUs. Esto es normal en entorno de desarrollo, pero podría afectar el rendimiento en producción.")
    
    except ImportError:
        logger.warning("⚠ TensorFlow no está instalado. Esto es normal en entorno de desarrollo.")
    except Exception as e:
        logger.error(f"✗ Error al verificar TensorFlow: {str(e)}")
    
    return results

def check_memory_management(project_root: str) -> Dict[str, bool]:
    """Verifica que el manejo de memoria está implementado correctamente."""
    logger.info("Verificando manejo de memoria...")
    results = {
        "cleanup_memory_found": False,
        "memory_limit_found": False,
        "gc_collect_found": False,
        "keras_clear_session_found": False
    }
    
    # Archivos a verificar
    files_to_check = [
        "src/models/deep_learning/cloud_optimizer.py",
        "src/models/deep_learning/cloud_init.py",
        "src/models/deep_learning/lstm_model_part3.py"
    ]
    
    # Patrones a buscar
    patterns = {
        "cleanup_memory_found": "cleanup_memory",
        "memory_limit_found": "MEMORY_LIMIT_MB",
        "gc_collect_found": "gc.collect",
        "keras_clear_session_found": "keras.backend.clear_session"
    }
    
    # Verificar cada archivo
    for file_path in files_to_check:
        full_path = os.path.join(project_root, file_path)
        if not os.path.exists(full_path):
            continue
        
        # Leer contenido del archivo
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Verificar patrones
        for key, pattern in patterns.items():
            if pattern in content and not results[key]:
                results[key] = True
                logger.info(f"✓ Patrón encontrado: {pattern}")
    
    # Registrar patrones no encontrados
    for key, found in results.items():
        if not found:
            pattern = patterns[key]
            logger.error(f"✗ Patrón no encontrado: {pattern}")
    
    return results

def generate_recommendations(checks: Dict[str, Any]) -> List[str]:
    """Genera recomendaciones basadas en los resultados de las verificaciones."""
    recommendations = []
    
    # Recomendaciones para archivos críticos
    missing_files = [file for file, exists in checks["critical_files"].items() if not exists]
    if missing_files:
        recommendations.append(f"Crear los siguientes archivos críticos: {', '.join(missing_files)}")
    
    # Recomendaciones para dependencias
    missing_deps = [dep for dep, info in checks["dependencies"].items() if not info["found"]]
    if missing_deps:
        recommendations.append(f"Agregar las siguientes dependencias a requirements.txt: {', '.join(missing_deps)}")
    
    # Recomendaciones para variables de entorno
    missing_vars = [var for var, found in checks["environment_variables"].items() if not found]
    if missing_vars:
        recommendations.append(f"Configurar las siguientes variables de entorno: {', '.join(missing_vars)}")
    
    # Recomendaciones para importaciones
    problematic_imports = [file for file, ok in checks["imports"].items() if not ok]
    if problematic_imports:
        recommendations.append(f"Corregir las importaciones en los siguientes archivos: {', '.join(problematic_imports)}")
    
    # Recomendaciones para TensorFlow
    tf_checks = checks["tensorflow_compatibility"]
    if not tf_checks.get("python_version_compatible", True):
        recommendations.append(f"Usar Python 3.7-3.10 en lugar de {tf_checks.get('python_version', 'desconocido')}")
    
    # Recomendaciones para manejo de memoria
    memory_checks = checks["memory_management"]
    missing_memory_patterns = [key.replace("_found", "") for key, found in memory_checks.items() if not found]
    if missing_memory_patterns:
        recommendations.append(f"Implementar los siguientes patrones de manejo de memoria: {', '.join(missing_memory_patterns)}")
    
    return recommendations

def fix_issues(project_root: str, checks: Dict[str, Any]) -> Dict[str, Any]:
    """Intenta corregir automáticamente los problemas encontrados."""
    logger.info("Intentando corregir problemas...")
    fixes = {
        "fixed_dependencies": [],
        "fixed_environment_variables": [],
        "fixed_imports": [],
        "fixed_memory_management": []
    }
    
    # Corregir dependencias
    requirements_path = os.path.join(project_root, "requirements.txt")
    if os.path.exists(requirements_path):
        missing_deps = [dep for dep, info in checks["dependencies"].items() if not info["found"]]
        if missing_deps:
            try:
                with open(requirements_path, 'a') as f:
                    f.write("\n# Dependencias agregadas automáticamente para compatibilidad con Google Cloud VM\n")
                    for dep in missing_deps:
                        for full_dep in CLOUD_DEPENDENCIES:
                            if full_dep.startswith(dep):
                                f.write(f"{full_dep}\n")
                                fixes["fixed_dependencies"].append(dep)
                
                logger.info(f"✓ Agregadas {len(fixes['fixed_dependencies'])} dependencias a requirements.txt")
            except Exception as e:
                logger.error(f"✗ Error al corregir dependencias: {str(e)}")
    
    # Corregir variables de entorno en cloud_init.py
    cloud_init_path = os.path.join(project_root, "src/models/deep_learning/cloud_init.py")
    if os.path.exists(cloud_init_path):
        missing_vars = [var for var, found in checks["environment_variables"].items() if not found]
        if missing_vars:
            try:
                with open(cloud_init_path, 'r') as f:
                    content = f.read()
                
                # Buscar función setup_cloud_environment
                if "def setup_cloud_environment" in content and "return True" in content:
                    # Insertar variables de entorno antes de return True
                    new_content = content.replace(
                        "return True",
                        "    # Variables de entorno agregadas automáticamente\n" + 
                        "\n".join([f"    os.environ['{var}'] = '{CLOUD_ENV_VARS[i].split('=')[1]}'" for i, var in enumerate(missing_vars) if var in [env.split('=')[0] for env in CLOUD_ENV_VARS]]) + 
                        "\n\n    return True"
                    )
                    
                    with open(cloud_init_path, 'w') as f:
                        f.write(new_content)
                    
                    fixes["fixed_environment_variables"] = missing_vars
                    logger.info(f"✓ Agregadas {len(missing_vars)} variables de entorno a cloud_init.py")
            except Exception as e:
                logger.error(f"✗ Error al corregir variables de entorno: {str(e)}")
    
    # Corregir manejo de memoria en cloud_optimizer.py
    cloud_optimizer_path = os.path.join(project_root, "src/models/deep_learning/cloud_optimizer.py")
    if os.path.exists(cloud_optimizer_path):
        missing_memory_patterns = {key.replace("_found", ""): not found for key, found in checks["memory_management"].items()}
        if any(missing_memory_patterns.values()):
            try:
                with open(cloud_optimizer_path, 'r') as f:
                    content = f.read()
                
                # Verificar si necesitamos agregar gc.collect
                if missing_memory_patterns.get("gc_collect", False) and "def cleanup_memory" in content:
                    if "import gc" not in content:
                        content = "import gc\n" + content
                    
                    # Agregar gc.collect a la función cleanup_memory
                    if "def cleanup_memory" in content:
                        content = content.replace(
                            "def cleanup_memory():",
                            "def cleanup_memory():\n    \"\"\"Libera memoria no utilizada.\"\"\"\n    # Forzar recolección de basura\n    gc.collect()"
                        )
                        fixes["fixed_memory_management"].append("gc_collect")
                
                # Verificar si necesitamos agregar keras.backend.clear_session
                if missing_memory_patterns.get("keras_clear_session", False):
                    if "def cleanup_memory" in content:
                        content = content.replace(
                            "def cleanup_memory():",
                            "def cleanup_memory():\n    \"\"\"Libera memoria no utilizada.\"\"\"\n    try:\n        # Limpiar sesión de Keras\n        from tensorflow import keras\n        keras.backend.clear_session()\n    except ImportError:\n        pass"
                        )
                        fixes["fixed_memory_management"].append("keras_clear_session")
                
                with open(cloud_optimizer_path, 'w') as f:
                    f.write(content)
                
                logger.info(f"✓ Mejorado manejo de memoria en cloud_optimizer.py")
            except Exception as e:
                logger.error(f"✗ Error al corregir manejo de memoria: {str(e)}")
    
    return fixes

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Verifica la compatibilidad del proyecto con Google Cloud VM")
    parser.add_argument("--fix", action="store_true", help="Intenta corregir automáticamente los problemas encontrados")
    parser.add_argument("--report-file", help="Ruta del archivo donde guardar el informe (formato JSON)")
    parser.add_argument("--project-root", help="Ruta raíz del proyecto")
    
    args = parser.parse_args()
    
    # Determinar la ruta raíz del proyecto
    project_root = args.project_root
    if not project_root:
        # Subir tres niveles desde la ubicación de este script
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Realizar verificaciones
    checks = {
        "timestamp": datetime.now().isoformat(),
        "project_root": project_root,
        "critical_files": check_critical_files(project_root),
        "dependencies": check_dependencies(project_root),
        "environment_variables": check_environment_variables(project_root),
        "imports": check_imports(project_root),
        "tensorflow_compatibility": check_tensorflow_compatibility(),
        "memory_management": check_memory_management(project_root)
    }
    
    # Generar recomendaciones
    recommendations = generate_recommendations(checks)
    checks["recommendations"] = recommendations
    
    # Intentar corregir problemas si se solicitó
    if args.fix:
        fixes = fix_issues(project_root, checks)
        checks["fixes"] = fixes
    
    # Imprimir informe
    print("\n" + "="*80)
    print("INFORME DE COMPATIBILIDAD CON GOOGLE CLOUD VM")
    print(f"Fecha: {checks['timestamp']}")
    print(f"Proyecto: {checks['project_root']}")
    print("="*80 + "\n")
    
    # Resumen de verificaciones
    print("RESUMEN DE VERIFICACIONES:")
    print("-"*80)
    
    # Archivos críticos
    critical_files_ok = all(checks["critical_files"].values())
    print(f"Archivos críticos: {'✓' if critical_files_ok else '✗'}")
    
    # Dependencias
    dependencies_ok = all(info["found"] for info in checks["dependencies"].values())
    print(f"Dependencias: {'✓' if dependencies_ok else '✗'}")
    
    # Variables de entorno
    env_vars_ok = all(checks["environment_variables"].values())
    print(f"Variables de entorno: {'✓' if env_vars_ok else '✗'}")
    
    # Importaciones
    imports_ok = all(checks["imports"].values())
    print(f"Importaciones: {'✓' if imports_ok else '✗'}")
    
    # Compatibilidad con TensorFlow
    tf_compat = checks["tensorflow_compatibility"]
    tf_ok = tf_compat.get("python_version_compatible", False)
    print(f"Compatibilidad con TensorFlow: {'✓' if tf_ok else '✗'}")
    
    # Manejo de memoria
    memory_ok = all(checks["memory_management"].values())
    print(f"Manejo de memoria: {'✓' if memory_ok else '✗'}")
    
    # Resultado general
    all_ok = critical_files_ok and dependencies_ok and env_vars_ok and imports_ok and tf_ok and memory_ok
    print("\nRESULTADO GENERAL:")
    if all_ok:
        print("✅ El proyecto es compatible con Google Cloud VM")
    else:
        print("❌ El proyecto tiene problemas de compatibilidad con Google Cloud VM")
    
    # Recomendaciones
    if recommendations:
        print("\nRECOMENDACIONES:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    
    # Correcciones realizadas
    if args.fix and "fixes" in checks:
        fixes = checks["fixes"]
        print("\nCORRECCIONES REALIZADAS:")
        
        if fixes["fixed_dependencies"]:
            print(f"- Dependencias agregadas: {', '.join(fixes['fixed_dependencies'])}")
        
        if fixes["fixed_environment_variables"]:
            print(f"- Variables de entorno configuradas: {', '.join(fixes['fixed_environment_variables'])}")
        
        if fixes["fixed_memory_management"]:
            print(f"- Mejoras en manejo de memoria: {', '.join(fixes['fixed_memory_management'])}")
        
        if not any(fixes.values()):
            print("- No se realizaron correcciones automáticas")
    
    print("\n" + "="*80)
    
    # Guardar informe si se solicitó
    if args.report_file:
        try:
            with open(args.report_file, 'w', encoding='utf-8') as f:
                json.dump(checks, f, indent=2, ensure_ascii=False)
            logger.info(f"Informe guardado en {args.report_file}")
        except Exception as e:
            logger.error(f"Error al guardar el informe: {str(e)}")
    
    # Devolver código de salida
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())

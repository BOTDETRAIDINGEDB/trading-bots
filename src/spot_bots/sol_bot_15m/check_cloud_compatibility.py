#!/usr/bin/env python3
"""
Script para verificar la compatibilidad del bot SOL con Google Cloud.
Realiza pruebas de todos los componentes críticos y genera un informe.
"""

import os
import sys
import logging
import traceback
import importlib
import json
import platform
import tempfile
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cloud_compatibility_check.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("cloud_compatibility")

class CloudCompatibilityChecker:
    """
    Verifica la compatibilidad de todos los componentes con Google Cloud.
    """
    
    def __init__(self):
        """Inicializa el verificador de compatibilidad."""
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "platform": platform.platform(),
            "python_version": sys.version,
            "checks": {},
            "overall_status": "pending"
        }
        
        # Simular entorno cloud para pruebas
        os.environ["CLOUD_ENV"] = "true"
        os.environ["MODELS_DIR"] = os.path.join(tempfile.gettempdir(), "sol_bot_test", "models")
        os.environ["DATA_DIR"] = os.path.join(tempfile.gettempdir(), "sol_bot_test", "data")
        os.environ["MEMORY_LIMIT_MB"] = "2048"
        
        # Crear directorios temporales
        os.makedirs(os.environ["MODELS_DIR"], exist_ok=True)
        os.makedirs(os.environ["DATA_DIR"], exist_ok=True)
        
        logger.info(f"Verificador de compatibilidad inicializado. Directorios temporales creados en {tempfile.gettempdir()}/sol_bot_test")
    
    def run_all_checks(self) -> Dict[str, Any]:
        """
        Ejecuta todas las verificaciones de compatibilidad.
        
        Returns:
            Diccionario con resultados de las verificaciones
        """
        try:
            # Verificar dependencias
            self.check_dependencies()
            
            # Verificar componentes principales
            self.check_data_loader()
            self.check_data_processor()
            self.check_lstm_model()
            self.check_deep_learning_integration()
            
            # Verificar compatibilidad con Google Cloud
            self.check_cloud_storage()
            self.check_cloud_logging()
            self.check_cloud_secrets()
            
            # Verificar configuración
            self.check_config_files()
            
            # Determinar estado general
            failed_checks = [name for name, result in self.results["checks"].items() 
                            if result.get("status") == "failed"]
            
            if failed_checks:
                self.results["overall_status"] = "failed"
                self.results["failed_checks"] = failed_checks
            else:
                self.results["overall_status"] = "passed"
            
            # Guardar resultados
            with open("cloud_compatibility_results.json", "w") as f:
                json.dump(self.results, f, indent=2)
            
            logger.info(f"Verificación completada. Estado general: {self.results['overall_status']}")
            
            return self.results
        
        except Exception as e:
            logger.error(f"Error durante la verificación: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            self.results["overall_status"] = "error"
            self.results["error"] = str(e)
            return self.results
    
    def check_dependencies(self) -> None:
        """Verifica que todas las dependencias estén instaladas y sean compatibles."""
        logger.info("Verificando dependencias...")
        
        required_packages = [
            "tensorflow", "numpy", "pandas", "scikit-learn", "matplotlib",
            "ccxt", "pytz", "joblib", "psutil", "google-cloud-storage",
            "google-cloud-logging", "google-cloud-secret-manager"
        ]
        
        results = {
            "status": "pending",
            "details": {},
            "missing_packages": []
        }
        
        for package in required_packages:
            try:
                module = importlib.import_module(package)
                version = getattr(module, "__version__", "unknown")
                results["details"][package] = {
                    "installed": True,
                    "version": version
                }
                logger.info(f"✓ {package} {version} instalado correctamente")
            except ImportError:
                results["details"][package] = {
                    "installed": False,
                    "error": "No instalado"
                }
                results["missing_packages"].append(package)
                logger.warning(f"✗ {package} no está instalado")
            except Exception as e:
                results["details"][package] = {
                    "installed": False,
                    "error": str(e)
                }
                results["missing_packages"].append(package)
                logger.warning(f"✗ Error al verificar {package}: {str(e)}")
        
        if results["missing_packages"]:
            results["status"] = "failed"
            logger.error(f"Faltan dependencias: {', '.join(results['missing_packages'])}")
        else:
            results["status"] = "passed"
            logger.info("Todas las dependencias están instaladas correctamente")
        
        self.results["checks"]["dependencies"] = results
    
    def check_data_loader(self) -> None:
        """Verifica la compatibilidad del DataLoader con Google Cloud."""
        logger.info("Verificando MultiTimeframeDataLoader...")
        
        results = {
            "status": "pending",
            "details": {}
        }
        
        try:
            # Importar el módulo
            from models.deep_learning.data_loader import MultiTimeframeDataLoader
            
            # Verificar inicialización
            data_loader = MultiTimeframeDataLoader(
                symbol="BTC/USDT",
                timeframes=["5m", "15m"],
                data_dir=os.environ["DATA_DIR"]
            )
            
            results["details"]["initialization"] = "success"
            
            # Verificar métodos críticos
            methods_to_check = [
                "_fetch_historical_data",
                "load_all_timeframes",
                "get_latest_data"
            ]
            
            results["details"]["methods"] = {}
            
            for method in methods_to_check:
                if hasattr(data_loader, method) and callable(getattr(data_loader, method)):
                    results["details"]["methods"][method] = "available"
                else:
                    results["details"]["methods"][method] = "missing"
                    logger.warning(f"Método {method} no encontrado en MultiTimeframeDataLoader")
            
            # Verificar si todos los métodos están disponibles
            if all(status == "available" for status in results["details"]["methods"].values()):
                results["status"] = "passed"
                logger.info("MultiTimeframeDataLoader es compatible con Google Cloud")
            else:
                results["status"] = "failed"
                logger.error("MultiTimeframeDataLoader tiene métodos faltantes")
        
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            logger.error(f"Error al verificar MultiTimeframeDataLoader: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
        
        self.results["checks"]["data_loader"] = results
    
    def check_data_processor(self) -> None:
        """Verifica la compatibilidad del DataProcessor con Google Cloud."""
        logger.info("Verificando DeepLearningDataProcessor...")
        
        results = {
            "status": "pending",
            "details": {}
        }
        
        try:
            # Importar el módulo
            from models.deep_learning.data_processor import DeepLearningDataProcessor
            
            # Verificar inicialización
            data_processor = DeepLearningDataProcessor(
                sequence_length=60,
                prediction_horizon=3,
                scaler_path=os.path.join(os.environ["MODELS_DIR"], "dl_scaler.pkl"),
                base_dir=os.environ["MODELS_DIR"],
                use_cloud_storage=True
            )
            
            results["details"]["initialization"] = "success"
            
            # Verificar métodos críticos
            methods_to_check = [
                "_load_feature_config",
                "_load_scaler",
                "_save_scaler",
                "prepare_sequences"
            ]
            
            results["details"]["methods"] = {}
            
            for method in methods_to_check:
                if hasattr(data_processor, method) and callable(getattr(data_processor, method)):
                    results["details"]["methods"][method] = "available"
                else:
                    results["details"]["methods"][method] = "missing"
                    logger.warning(f"Método {method} no encontrado en DeepLearningDataProcessor")
            
            # Verificar si todos los métodos están disponibles
            if all(status == "available" for status in results["details"]["methods"].values()):
                results["status"] = "passed"
                logger.info("DeepLearningDataProcessor es compatible con Google Cloud")
            else:
                results["status"] = "failed"
                logger.error("DeepLearningDataProcessor tiene métodos faltantes")
        
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            logger.error(f"Error al verificar DeepLearningDataProcessor: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
        
        self.results["checks"]["data_processor"] = results
    
    def check_lstm_model(self) -> None:
        """Verifica la compatibilidad del modelo LSTM con Google Cloud."""
        logger.info("Verificando DeepTimeSeriesModel...")
        
        results = {
            "status": "pending",
            "details": {}
        }
        
        try:
            # Importar el módulo
            from models.deep_learning.lstm_model import DeepTimeSeriesModel
            
            # Verificar inicialización
            model = DeepTimeSeriesModel(
                model_type="lstm",
                sequence_length=60,
                num_features=20,
                num_classes=3,
                model_path=os.path.join(os.environ["MODELS_DIR"], "lstm_model")
            )
            
            results["details"]["initialization"] = "success"
            
            # Verificar métodos críticos
            methods_to_check = [
                "_build_model",
                "save",
                "load",
                "predict"
            ]
            
            results["details"]["methods"] = {}
            
            for method in methods_to_check:
                if hasattr(model, method) and callable(getattr(model, method)):
                    results["details"]["methods"][method] = "available"
                else:
                    results["details"]["methods"][method] = "missing"
                    logger.warning(f"Método {method} no encontrado en DeepTimeSeriesModel")
            
            # Verificar si todos los métodos están disponibles
            if all(status == "available" for status in results["details"]["methods"].values()):
                results["status"] = "passed"
                logger.info("DeepTimeSeriesModel es compatible con Google Cloud")
            else:
                results["status"] = "failed"
                logger.error("DeepTimeSeriesModel tiene métodos faltantes")
        
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            logger.error(f"Error al verificar DeepTimeSeriesModel: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
        
        self.results["checks"]["lstm_model"] = results
    
    def check_deep_learning_integration(self) -> None:
        """Verifica la compatibilidad de la integración de aprendizaje profundo con Google Cloud."""
        logger.info("Verificando DeepLearningIntegration...")
        
        results = {
            "status": "pending",
            "details": {}
        }
        
        try:
            # Importar el módulo
            from models.deep_learning.deep_learning_integration import DeepLearningIntegration
            
            # Verificar inicialización
            integration = DeepLearningIntegration(
                symbol="BTC/USDT",
                timeframes=["5m", "15m", "1h"],
                base_timeframe="15m",
                model_type="lstm",
                models_dir=os.environ["MODELS_DIR"],
                use_cloud_storage=True
            )
            
            results["details"]["initialization"] = "success"
            
            # Verificar métodos críticos
            methods_to_check = [
                "load_model",
                "get_prediction",
                "_save_config"
            ]
            
            results["details"]["methods"] = {}
            
            for method in methods_to_check:
                if hasattr(integration, method) and callable(getattr(integration, method)):
                    results["details"]["methods"][method] = "available"
                else:
                    results["details"]["methods"][method] = "missing"
                    logger.warning(f"Método {method} no encontrado en DeepLearningIntegration")
            
            # Verificar si todos los métodos están disponibles
            if all(status == "available" for status in results["details"]["methods"].values()):
                results["status"] = "passed"
                logger.info("DeepLearningIntegration es compatible con Google Cloud")
            else:
                results["status"] = "failed"
                logger.error("DeepLearningIntegration tiene métodos faltantes")
        
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            logger.error(f"Error al verificar DeepLearningIntegration: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
        
        self.results["checks"]["deep_learning_integration"] = results
    
    def check_cloud_storage(self) -> None:
        """Verifica la compatibilidad con Google Cloud Storage."""
        logger.info("Verificando compatibilidad con Google Cloud Storage...")
        
        results = {
            "status": "pending",
            "details": {}
        }
        
        try:
            # Verificar si google-cloud-storage está instalado
            try:
                from google.cloud import storage
                results["details"]["library"] = "installed"
            except ImportError:
                results["details"]["library"] = "missing"
                results["status"] = "failed"
                logger.warning("Biblioteca google-cloud-storage no está instalada")
                self.results["checks"]["cloud_storage"] = results
                return
            
            # Verificar si se puede crear un cliente (no necesita credenciales para esto)
            try:
                client = storage.Client.create_anonymous_client()
                results["details"]["client_creation"] = "success"
            except Exception as e:
                results["details"]["client_creation"] = "failed"
                results["details"]["client_error"] = str(e)
                results["status"] = "failed"
                logger.warning(f"Error al crear cliente de Storage: {str(e)}")
                self.results["checks"]["cloud_storage"] = results
                return
            
            # Verificar si gsutil está disponible como alternativa
            try:
                import subprocess
                result = subprocess.run(["gsutil", "--version"], 
                                       capture_output=True, text=True)
                if result.returncode == 0:
                    results["details"]["gsutil"] = "available"
                    results["details"]["gsutil_version"] = result.stdout.strip()
                else:
                    results["details"]["gsutil"] = "error"
                    results["details"]["gsutil_error"] = result.stderr
            except Exception:
                results["details"]["gsutil"] = "not_found"
            
            # Si llegamos aquí, la verificación pasó
            results["status"] = "passed"
            logger.info("Compatibilidad con Google Cloud Storage verificada correctamente")
        
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            logger.error(f"Error al verificar compatibilidad con Google Cloud Storage: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
        
        self.results["checks"]["cloud_storage"] = results
    
    def check_cloud_logging(self) -> None:
        """Verifica la compatibilidad con Google Cloud Logging."""
        logger.info("Verificando compatibilidad con Google Cloud Logging...")
        
        results = {
            "status": "pending",
            "details": {}
        }
        
        try:
            # Verificar si google-cloud-logging está instalado
            try:
                from google.cloud import logging as cloud_logging
                results["details"]["library"] = "installed"
            except ImportError:
                results["details"]["library"] = "missing"
                results["status"] = "failed"
                logger.warning("Biblioteca google-cloud-logging no está instalada")
                self.results["checks"]["cloud_logging"] = results
                return
            
            # Verificar si se puede crear un cliente (no necesita credenciales para esto)
            try:
                client = cloud_logging.Client()
                results["details"]["client_creation"] = "success"
            except Exception as e:
                results["details"]["client_creation"] = "failed"
                results["details"]["client_error"] = str(e)
                results["status"] = "failed"
                logger.warning(f"Error al crear cliente de Logging: {str(e)}")
                self.results["checks"]["cloud_logging"] = results
                return
            
            # Si llegamos aquí, la verificación pasó
            results["status"] = "passed"
            logger.info("Compatibilidad con Google Cloud Logging verificada correctamente")
        
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            logger.error(f"Error al verificar compatibilidad con Google Cloud Logging: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
        
        self.results["checks"]["cloud_logging"] = results
    
    def check_cloud_secrets(self) -> None:
        """Verifica la compatibilidad con Google Cloud Secret Manager."""
        logger.info("Verificando compatibilidad con Google Cloud Secret Manager...")
        
        results = {
            "status": "pending",
            "details": {}
        }
        
        try:
            # Verificar si google-cloud-secret-manager está instalado
            try:
                from google.cloud import secretmanager
                results["details"]["library"] = "installed"
            except ImportError:
                results["details"]["library"] = "missing"
                results["status"] = "failed"
                logger.warning("Biblioteca google-cloud-secret-manager no está instalada")
                self.results["checks"]["cloud_secrets"] = results
                return
            
            # Verificar si se puede crear un cliente (no necesita credenciales para esto)
            try:
                client = secretmanager.SecretManagerServiceClient()
                results["details"]["client_creation"] = "success"
            except Exception as e:
                results["details"]["client_creation"] = "failed"
                results["details"]["client_error"] = str(e)
                results["status"] = "failed"
                logger.warning(f"Error al crear cliente de Secret Manager: {str(e)}")
                self.results["checks"]["cloud_secrets"] = results
                return
            
            # Si llegamos aquí, la verificación pasó
            results["status"] = "passed"
            logger.info("Compatibilidad con Google Cloud Secret Manager verificada correctamente")
        
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            logger.error(f"Error al verificar compatibilidad con Google Cloud Secret Manager: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
        
        self.results["checks"]["cloud_secrets"] = results
    
    def check_config_files(self) -> None:
        """Verifica que los archivos de configuración sean compatibles con Google Cloud."""
        logger.info("Verificando archivos de configuración...")
        
        results = {
            "status": "pending",
            "details": {}
        }
        
        try:
            # Verificar cloud_config.yaml
            cloud_config_path = "cloud_config.yaml"
            if os.path.exists(cloud_config_path):
                try:
                    import yaml
                    with open(cloud_config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # Verificar campos críticos
                    required_fields = ["runtime", "service", "env_variables", "resources"]
                    missing_fields = [field for field in required_fields if field not in config]
                    
                    if missing_fields:
                        results["details"]["cloud_config"] = {
                            "status": "invalid",
                            "missing_fields": missing_fields
                        }
                        logger.warning(f"Archivo cloud_config.yaml incompleto. Faltan campos: {missing_fields}")
                    else:
                        results["details"]["cloud_config"] = {
                            "status": "valid"
                        }
                        logger.info("Archivo cloud_config.yaml válido")
                except Exception as e:
                    results["details"]["cloud_config"] = {
                        "status": "error",
                        "error": str(e)
                    }
                    logger.warning(f"Error al analizar cloud_config.yaml: {str(e)}")
            else:
                results["details"]["cloud_config"] = {
                    "status": "missing"
                }
                logger.warning("Archivo cloud_config.yaml no encontrado")
            
            # Verificar deploy_to_cloud.py
            deploy_script_path = "deploy_to_cloud.py"
            if os.path.exists(deploy_script_path):
                results["details"]["deploy_script"] = {
                    "status": "present"
                }
                logger.info("Script deploy_to_cloud.py encontrado")
            else:
                results["details"]["deploy_script"] = {
                    "status": "missing"
                }
                logger.warning("Script deploy_to_cloud.py no encontrado")
            
            # Verificar CLOUD_DEPLOYMENT.md
            deployment_doc_path = "CLOUD_DEPLOYMENT.md"
            if os.path.exists(deployment_doc_path):
                results["details"]["deployment_doc"] = {
                    "status": "present"
                }
                logger.info("Documentación CLOUD_DEPLOYMENT.md encontrada")
            else:
                results["details"]["deployment_doc"] = {
                    "status": "missing"
                }
                logger.warning("Documentación CLOUD_DEPLOYMENT.md no encontrada")
            
            # Determinar estado general
            if (results["details"].get("cloud_config", {}).get("status") in ["valid", "present"] and
                results["details"].get("deploy_script", {}).get("status") == "present"):
                results["status"] = "passed"
                logger.info("Archivos de configuración compatibles con Google Cloud")
            else:
                results["status"] = "failed"
                logger.error("Faltan archivos de configuración necesarios para Google Cloud")
        
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            logger.error(f"Error al verificar archivos de configuración: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
        
        self.results["checks"]["config_files"] = results
    
    def cleanup(self) -> None:
        """Limpia los archivos temporales creados durante las verificaciones."""
        try:
            import shutil
            temp_dir = os.path.join(tempfile.gettempdir(), "sol_bot_test")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            logger.info(f"Directorio temporal {temp_dir} eliminado")
        except Exception as e:
            logger.warning(f"Error al limpiar archivos temporales: {str(e)}")

def print_report(results: Dict[str, Any]) -> None:
    """
    Imprime un informe legible de los resultados de la verificación.
    
    Args:
        results: Diccionario con resultados de las verificaciones
    """
    print("\n" + "="*80)
    print(f"INFORME DE COMPATIBILIDAD CON GOOGLE CLOUD - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    print(f"\nEstado general: {results['overall_status'].upper()}")
    print(f"Plataforma: {results['platform']}")
    print(f"Versión de Python: {results['python_version'].split()[0]}")
    
    print("\nRESULTADOS DE VERIFICACIONES:")
    print("-"*80)
    
    for name, check in results["checks"].items():
        status = check.get("status", "unknown")
        status_symbol = "✓" if status == "passed" else "✗"
        print(f"{status_symbol} {name.replace('_', ' ').title()}: {status.upper()}")
    
    if results["overall_status"] == "failed":
        print("\nVERIFICACIONES FALLIDAS:")
        for name in results.get("failed_checks", []):
            print(f"- {name.replace('_', ' ').title()}")
            if "error" in results["checks"][name]:
                print(f"  Error: {results['checks'][name]['error']}")
    
    print("\nRECOMENDACIONES:")
    if results["overall_status"] == "passed":
        print("✓ El sistema es compatible con Google Cloud. Puede proceder con el despliegue.")
        print("  Asegúrese de seguir las instrucciones en CLOUD_DEPLOYMENT.md")
    else:
        print("✗ Se encontraron problemas de compatibilidad que deben resolverse:")
        
        if "dependencies" in results.get("failed_checks", []):
            missing = results["checks"]["dependencies"].get("missing_packages", [])
            if missing:
                print(f"  - Instale las dependencias faltantes: {', '.join(missing)}")
                print("    pip install " + " ".join(missing))
        
        if "cloud_storage" in results.get("failed_checks", []):
            print("  - Instale y configure Google Cloud Storage:")
            print("    pip install google-cloud-storage")
            print("    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json")
        
        if "cloud_logging" in results.get("failed_checks", []):
            print("  - Instale y configure Google Cloud Logging:")
            print("    pip install google-cloud-logging")
        
        if "cloud_secrets" in results.get("failed_checks", []):
            print("  - Instale y configure Google Cloud Secret Manager:")
            print("    pip install google-cloud-secret-manager")
        
        if "config_files" in results.get("failed_checks", []):
            details = results["checks"]["config_files"]["details"]
            if details.get("cloud_config", {}).get("status") in ["missing", "invalid"]:
                print("  - Cree o corrija el archivo cloud_config.yaml")
            if details.get("deploy_script", {}).get("status") == "missing":
                print("  - Cree el script deploy_to_cloud.py")
    
    print("\nPara más detalles, consulte el archivo cloud_compatibility_results.json")
    print("="*80 + "\n")

def main():
    """Función principal."""
    print("Iniciando verificación de compatibilidad con Google Cloud...")
    
    checker = CloudCompatibilityChecker()
    results = checker.run_all_checks()
    print_report(results)
    
    # Limpiar archivos temporales
    checker.cleanup()
    
    # Devolver código de salida según resultado
    if results["overall_status"] == "passed":
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())

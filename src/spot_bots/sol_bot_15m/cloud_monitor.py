#!/usr/bin/env python3
"""
Script para monitorear el bot SOL en Google Cloud.
Proporciona métricas de rendimiento y estado del sistema.
"""

import os
import sys
import argparse
import time
import json
import logging
from datetime import datetime, timedelta
import subprocess
from typing import Dict, List, Any, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cloud_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("cloud_monitor")

class CloudMonitor:
    """Monitorea el estado y rendimiento del bot en Google Cloud."""
    
    def __init__(self, project_id: str, service_name: str = "sol-bot-deep-learning"):
        """
        Inicializa el monitor de Google Cloud.
        
        Args:
            project_id: ID del proyecto en Google Cloud
            service_name: Nombre del servicio en App Engine
        """
        self.project_id = project_id
        self.service_name = service_name
        self.metrics = {}
        
        logger.info(f"Monitor inicializado para proyecto {project_id}, servicio {service_name}")
    
    def check_service_status(self) -> Dict[str, Any]:
        """
        Verifica el estado del servicio en App Engine.
        
        Returns:
            Diccionario con información del estado del servicio
        """
        try:
            result = subprocess.run(
                ['gcloud', 'app', 'services', 'describe', self.service_name, 
                 '--project', self.project_id, '--format', 'json'],
                capture_output=True, text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Error al verificar estado del servicio: {result.stderr}")
                return {"status": "error", "message": result.stderr}
            
            service_info = json.loads(result.stdout)
            
            status = {
                "name": service_info.get("name", ""),
                "status": service_info.get("servingStatus", "UNKNOWN"),
                "url": service_info.get("id", ""),
                "last_deployed": service_info.get("lastDeployTime", ""),
                "version": service_info.get("split", {}).get("allocations", {}).keys()
            }
            
            logger.info(f"Estado del servicio: {status['status']}")
            return status
        
        except Exception as e:
            logger.error(f"Error al verificar estado del servicio: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_logs(self, hours: int = 1, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Obtiene los logs recientes del servicio.
        
        Args:
            hours: Número de horas hacia atrás para obtener logs
            limit: Número máximo de entradas de log a obtener
            
        Returns:
            Lista de entradas de log
        """
        try:
            time_filter = f"time>=\"{(datetime.now() - timedelta(hours=hours)).isoformat()}Z\""
            
            result = subprocess.run(
                ['gcloud', 'logging', 'read', 
                 f"resource.type=gae_app AND resource.labels.module_id={self.service_name} AND {time_filter}",
                 f"--project={self.project_id}",
                 f"--limit={limit}",
                 '--format=json'],
                capture_output=True, text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Error al obtener logs: {result.stderr}")
                return []
            
            logs = json.loads(result.stdout)
            
            # Procesar logs para formato más legible
            processed_logs = []
            for log in logs:
                entry = {
                    "timestamp": log.get("timestamp", ""),
                    "severity": log.get("severity", ""),
                    "message": log.get("textPayload", log.get("jsonPayload", {}))
                }
                processed_logs.append(entry)
            
            logger.info(f"Obtenidos {len(processed_logs)} registros de log")
            return processed_logs
        
        except Exception as e:
            logger.error(f"Error al obtener logs: {str(e)}")
            return []
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtiene métricas de rendimiento del servicio.
        
        Returns:
            Diccionario con métricas de rendimiento
        """
        try:
            # Obtener métricas de CPU
            cpu_result = subprocess.run(
                ['gcloud', 'monitoring', 'metrics', 'list',
                 'appengine.googleapis.com/instance/cpu/utilization',
                 f'--project={self.project_id}',
                 '--service=app',
                 f'--service-name={self.service_name}',
                 '--minutes=60',
                 '--format=json'],
                capture_output=True, text=True
            )
            
            # Obtener métricas de memoria
            mem_result = subprocess.run(
                ['gcloud', 'monitoring', 'metrics', 'list',
                 'appengine.googleapis.com/instance/memory/usage',
                 f'--project={self.project_id}',
                 '--service=app',
                 f'--service-name={self.service_name}',
                 '--minutes=60',
                 '--format=json'],
                capture_output=True, text=True
            )
            
            metrics = {
                "cpu": json.loads(cpu_result.stdout) if cpu_result.returncode == 0 else {"error": cpu_result.stderr},
                "memory": json.loads(mem_result.stdout) if mem_result.returncode == 0 else {"error": mem_result.stderr}
            }
            
            # Procesar métricas para formato más legible
            processed_metrics = {
                "timestamp": datetime.now().isoformat(),
                "cpu_utilization": self._extract_metric_value(metrics["cpu"]),
                "memory_usage": self._extract_metric_value(metrics["memory"])
            }
            
            logger.info(f"Métricas obtenidas: CPU {processed_metrics['cpu_utilization']}, Memoria {processed_metrics['memory_usage']}")
            return processed_metrics
        
        except Exception as e:
            logger.error(f"Error al obtener métricas: {str(e)}")
            return {"error": str(e)}
    
    def _extract_metric_value(self, metric_data: Dict[str, Any]) -> float:
        """Extrae el valor de una métrica del formato de respuesta de Google Cloud."""
        try:
            if "error" in metric_data:
                return -1
            
            # Extraer el valor más reciente de la métrica
            points = metric_data.get("timeSeries", [{}])[0].get("points", [{}])
            if points:
                return points[0].get("value", {}).get("doubleValue", -1)
            return -1
        except Exception:
            return -1
    
    def check_storage(self, bucket_name: str) -> Dict[str, Any]:
        """
        Verifica el estado del almacenamiento en Cloud Storage.
        
        Args:
            bucket_name: Nombre del bucket a verificar
            
        Returns:
            Diccionario con información del almacenamiento
        """
        try:
            # Verificar si el bucket existe
            result = subprocess.run(
                ['gsutil', 'ls', '-L', '-b', f"gs://{bucket_name}"],
                capture_output=True, text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Error al verificar bucket {bucket_name}: {result.stderr}")
                return {"status": "error", "message": result.stderr}
            
            # Obtener tamaño del bucket
            size_result = subprocess.run(
                ['gsutil', 'du', '-s', f"gs://{bucket_name}"],
                capture_output=True, text=True
            )
            
            # Listar directorios principales
            ls_result = subprocess.run(
                ['gsutil', 'ls', f"gs://{bucket_name}/"],
                capture_output=True, text=True
            )
            
            storage_info = {
                "bucket": bucket_name,
                "status": "available",
                "size_bytes": size_result.stdout.split()[0] if size_result.returncode == 0 else "unknown",
                "directories": [d.strip() for d in ls_result.stdout.splitlines()] if ls_result.returncode == 0 else []
            }
            
            logger.info(f"Bucket {bucket_name} verificado: {storage_info['size_bytes']} bytes")
            return storage_info
        
        except Exception as e:
            logger.error(f"Error al verificar almacenamiento: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def check_model_files(self, bucket_name: str) -> Dict[str, Any]:
        """
        Verifica los archivos de modelo en Cloud Storage.
        
        Args:
            bucket_name: Nombre del bucket donde están los modelos
            
        Returns:
            Diccionario con información de los archivos de modelo
        """
        try:
            result = subprocess.run(
                ['gsutil', 'ls', '-l', f"gs://{bucket_name}/models/"],
                capture_output=True, text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Error al verificar archivos de modelo: {result.stderr}")
                return {"status": "error", "message": result.stderr}
            
            # Procesar salida para obtener lista de archivos
            files = []
            for line in result.stdout.splitlines():
                if line.strip() and not line.startswith("TOTAL"):
                    parts = line.split()
                    if len(parts) >= 2:
                        size = parts[0]
                        date = " ".join(parts[1:3])
                        path = parts[-1]
                        files.append({
                            "path": path,
                            "size": size,
                            "date": date
                        })
            
            model_info = {
                "status": "available" if files else "empty",
                "count": len(files),
                "files": files
            }
            
            logger.info(f"Archivos de modelo verificados: {model_info['count']} archivos")
            return model_info
        
        except Exception as e:
            logger.error(f"Error al verificar archivos de modelo: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def generate_health_report(self, bucket_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Genera un informe completo del estado del sistema.
        
        Args:
            bucket_name: Nombre del bucket para verificar almacenamiento
            
        Returns:
            Diccionario con informe completo de estado
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "project_id": self.project_id,
            "service_name": self.service_name,
            "service_status": self.check_service_status(),
            "metrics": self.get_metrics(),
            "recent_errors": self._get_recent_errors(hours=24)
        }
        
        if bucket_name:
            report["storage"] = self.check_storage(bucket_name)
            report["models"] = self.check_model_files(bucket_name)
        
        # Determinar estado general
        if report["service_status"].get("status") == "error":
            report["overall_status"] = "critical"
        elif len(report["recent_errors"]) > 10:
            report["overall_status"] = "warning"
        else:
            report["overall_status"] = "healthy"
        
        # Guardar informe
        with open(f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Informe de salud generado. Estado general: {report['overall_status']}")
        return report
    
    def _get_recent_errors(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Obtiene errores recientes de los logs."""
        logs = self.get_logs(hours=hours, limit=1000)
        errors = [log for log in logs if log.get("severity", "") in ["ERROR", "CRITICAL"]]
        return errors[:100]  # Limitar a 100 errores para evitar informes demasiado grandes

def parse_arguments():
    """Parsea los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description='Monitorear bot SOL en Google Cloud')
    
    parser.add_argument('--project-id', type=str, required=True, help='ID del proyecto en Google Cloud')
    parser.add_argument('--service-name', type=str, default='sol-bot-deep-learning', help='Nombre del servicio')
    parser.add_argument('--bucket-name', type=str, help='Nombre del bucket para almacenamiento')
    
    parser.add_argument('--check-status', action='store_true', help='Verificar estado del servicio')
    parser.add_argument('--get-logs', action='store_true', help='Obtener logs recientes')
    parser.add_argument('--get-metrics', action='store_true', help='Obtener métricas de rendimiento')
    parser.add_argument('--check-storage', action='store_true', help='Verificar almacenamiento')
    parser.add_argument('--full-report', action='store_true', help='Generar informe completo')
    
    parser.add_argument('--log-hours', type=int, default=1, help='Horas de logs a obtener')
    parser.add_argument('--log-limit', type=int, default=100, help='Límite de entradas de log')
    
    return parser.parse_args()

def print_report(report: Dict[str, Any]):
    """Imprime un informe legible."""
    print("\n" + "="*80)
    print(f"INFORME DE SALUD DEL BOT SOL - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    print(f"\nEstado general: {report['overall_status'].upper()}")
    print(f"Proyecto: {report['project_id']}")
    print(f"Servicio: {report['service_name']}")
    
    print("\nESTADO DEL SERVICIO:")
    print("-"*80)
    status = report["service_status"]
    if "error" in status:
        print(f"Error: {status.get('message', 'Desconocido')}")
    else:
        print(f"Estado: {status.get('status', 'Desconocido')}")
        print(f"URL: {status.get('url', 'N/A')}")
        print(f"Última implementación: {status.get('last_deployed', 'Desconocido')}")
    
    print("\nMÉTRICAS DE RENDIMIENTO:")
    print("-"*80)
    metrics = report["metrics"]
    if "error" in metrics:
        print(f"Error: {metrics['error']}")
    else:
        print(f"CPU: {metrics.get('cpu_utilization', 'N/A')}")
        print(f"Memoria: {metrics.get('memory_usage', 'N/A')}")
    
    if "storage" in report:
        print("\nALMACENAMIENTO:")
        print("-"*80)
        storage = report["storage"]
        print(f"Bucket: {storage.get('bucket', 'N/A')}")
        print(f"Estado: {storage.get('status', 'Desconocido')}")
        print(f"Tamaño: {storage.get('size_bytes', 'Desconocido')} bytes")
        
        if "directories" in storage and storage["directories"]:
            print("Directorios:")
            for directory in storage["directories"][:5]:  # Mostrar solo los primeros 5
                print(f"  - {directory}")
            if len(storage["directories"]) > 5:
                print(f"  ... y {len(storage['directories']) - 5} más")
    
    if "models" in report:
        print("\nARCHIVOS DE MODELO:")
        print("-"*80)
        models = report["models"]
        print(f"Estado: {models.get('status', 'Desconocido')}")
        print(f"Cantidad: {models.get('count', 0)} archivos")
        
        if "files" in models and models["files"]:
            print("Archivos recientes:")
            for file in models["files"][:5]:  # Mostrar solo los primeros 5
                print(f"  - {file['path']} ({file['size']} bytes, {file['date']})")
            if len(models["files"]) > 5:
                print(f"  ... y {len(models['files']) - 5} más")
    
    print("\nERRORES RECIENTES:")
    print("-"*80)
    errors = report.get("recent_errors", [])
    if not errors:
        print("No se encontraron errores recientes.")
    else:
        print(f"Se encontraron {len(errors)} errores recientes:")
        for i, error in enumerate(errors[:5]):  # Mostrar solo los primeros 5
            print(f"  {i+1}. [{error.get('timestamp', 'N/A')}] {error.get('message', 'N/A')}")
        if len(errors) > 5:
            print(f"  ... y {len(errors) - 5} más")
    
    print("\nRECOMENDACIONES:")
    print("-"*80)
    if report["overall_status"] == "healthy":
        print("✓ El sistema está funcionando correctamente.")
    elif report["overall_status"] == "warning":
        print("⚠ El sistema está funcionando pero con advertencias:")
        print("  - Revise los errores recientes en los logs")
        print("  - Verifique el uso de recursos (CPU/memoria)")
    else:
        print("✗ El sistema tiene problemas críticos:")
        print("  - Verifique el estado del servicio en Google Cloud Console")
        print("  - Revise los logs detallados con 'gcloud app logs tail'")
    
    print("\nPara más detalles, consulte el archivo JSON del informe completo.")
    print("="*80 + "\n")

def main():
    """Función principal."""
    args = parse_arguments()
    
    monitor = CloudMonitor(args.project_id, args.service_name)
    
    if args.check_status:
        status = monitor.check_service_status()
        print(json.dumps(status, indent=2))
    
    if args.get_logs:
        logs = monitor.get_logs(hours=args.log_hours, limit=args.log_limit)
        print(json.dumps(logs, indent=2))
    
    if args.get_metrics:
        metrics = monitor.get_metrics()
        print(json.dumps(metrics, indent=2))
    
    if args.check_storage and args.bucket_name:
        storage = monitor.check_storage(args.bucket_name)
        print(json.dumps(storage, indent=2))
    
    if args.full_report:
        report = monitor.generate_health_report(args.bucket_name)
        print_report(report)
    
    # Si no se especifica ninguna acción, generar informe completo
    if not any([args.check_status, args.get_logs, args.get_metrics, 
                args.check_storage, args.full_report]):
        report = monitor.generate_health_report(args.bucket_name)
        print_report(report)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script para desplegar el bot SOL con aprendizaje profundo en Google Cloud.
Automatiza la configuración y despliegue del sistema en GCP.
"""

import os
import sys
import argparse
import subprocess
import json
import yaml
import time
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deploy_cloud.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parsea los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description='Desplegar bot SOL con aprendizaje profundo en Google Cloud')
    
    # Argumentos generales
    parser.add_argument('--project-id', type=str, required=True, help='ID del proyecto en Google Cloud')
    parser.add_argument('--region', type=str, default='us-central1', help='Región de Google Cloud')
    parser.add_argument('--service-name', type=str, default='sol-bot-deep-learning', help='Nombre del servicio')
    
    # Argumentos de configuración
    parser.add_argument('--config-file', type=str, default='cloud_config.yaml', help='Archivo de configuración')
    parser.add_argument('--bucket-name', type=str, help='Nombre del bucket para almacenamiento (opcional)')
    
    # Argumentos de credenciales
    parser.add_argument('--create-secrets', action='store_true', help='Crear secretos en Secret Manager')
    parser.add_argument('--exchange-key-file', type=str, help='Archivo con API key del exchange')
    parser.add_argument('--exchange-secret-file', type=str, help='Archivo con API secret del exchange')
    parser.add_argument('--telegram-token-file', type=str, help='Archivo con token de Telegram')
    parser.add_argument('--telegram-chat-id-file', type=str, help='Archivo con chat ID de Telegram')
    
    # Modos de ejecución
    parser.add_argument('--prepare', action='store_true', help='Preparar entorno sin desplegar')
    parser.add_argument('--deploy', action='store_true', help='Desplegar aplicación')
    parser.add_argument('--update-config', action='store_true', help='Actualizar configuración')
    
    return parser.parse_args()

def check_gcloud_installed():
    """Verifica si gcloud está instalado y configurado."""
    try:
        result = subprocess.run(['gcloud', 'version'], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("gcloud no está instalado o accesible en el PATH")
            return False
        
        # Verificar autenticación
        result = subprocess.run(['gcloud', 'auth', 'list'], capture_output=True, text=True)
        if "No credentialed accounts." in result.stdout:
            logger.error("No hay cuentas autenticadas en gcloud. Ejecute 'gcloud auth login'")
            return False
        
        logger.info("gcloud está instalado y configurado correctamente")
        return True
    
    except Exception as e:
        logger.error(f"Error al verificar gcloud: {str(e)}")
        return False

def update_config_file(config_file, project_id, region, bucket_name=None):
    """Actualiza el archivo de configuración con los valores proporcionados."""
    try:
        # Leer archivo de configuración
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Actualizar valores
        if 'vpc_access_connector' in config:
            config['vpc_access_connector']['name'] = f"projects/{project_id}/locations/{region}/connectors/sol-bot-connector"
        
        if 'logging' in config and 'options' in config['logging']:
            config['logging']['options']['gcp-project'] = project_id
        
        # Actualizar bucket si se proporciona
        if bucket_name:
            config['env_variables']['STORAGE_BUCKET'] = bucket_name
            
            # Actualizar comandos de lifecycle
            if 'lifecycle' in config:
                if 'postStart' in config['lifecycle'] and 'exec' in config['lifecycle']['postStart']:
                    config['lifecycle']['postStart']['exec']['command'] = [
                        "bash", "-c", f"mkdir -p /tmp/data /tmp/models && gsutil -m cp -r gs://{bucket_name}/* /tmp/ || true"
                    ]
                
                if 'preStop' in config['lifecycle'] and 'exec' in config['lifecycle']['preStop']:
                    config['lifecycle']['preStop']['exec']['command'] = [
                        "bash", "-c", f"gsutil -m cp -r /tmp/models/* gs://{bucket_name}/models/ && gsutil -m cp -r /tmp/data/* gs://{bucket_name}/data/"
                    ]
        
        # Guardar archivo actualizado
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Archivo de configuración {config_file} actualizado correctamente")
        return True
    
    except Exception as e:
        logger.error(f"Error al actualizar archivo de configuración: {str(e)}")
        return False

def create_bucket(project_id, bucket_name, region):
    """Crea un bucket en Google Cloud Storage."""
    try:
        # Verificar si el bucket ya existe
        result = subprocess.run(
            ['gsutil', 'ls', '-p', project_id, f"gs://{bucket_name}"],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            logger.info(f"El bucket {bucket_name} ya existe")
            return True
        
        # Crear bucket
        result = subprocess.run(
            ['gsutil', 'mb', '-p', project_id, '-l', region, f"gs://{bucket_name}"],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Error al crear bucket: {result.stderr}")
            return False
        
        logger.info(f"Bucket {bucket_name} creado correctamente")
        return True
    
    except Exception as e:
        logger.error(f"Error al crear bucket: {str(e)}")
        return False

def create_secret(project_id, secret_name, secret_file):
    """Crea un secreto en Secret Manager."""
    try:
        # Verificar si el secreto ya existe
        result = subprocess.run(
            ['gcloud', 'secrets', 'describe', secret_name, '--project', project_id],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            logger.info(f"El secreto {secret_name} ya existe, actualizando valor")
        else:
            # Crear secreto
            result = subprocess.run(
                ['gcloud', 'secrets', 'create', secret_name, '--project', project_id],
                capture_output=True, text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Error al crear secreto {secret_name}: {result.stderr}")
                return False
        
        # Leer valor del secreto
        with open(secret_file, 'r') as f:
            secret_value = f.read().strip()
        
        # Guardar valor en archivo temporal
        temp_file = f"temp_secret_{int(time.time())}.txt"
        with open(temp_file, 'w') as f:
            f.write(secret_value)
        
        # Añadir versión al secreto
        result = subprocess.run(
            ['gcloud', 'secrets', 'versions', 'add', secret_name, '--data-file', temp_file, '--project', project_id],
            capture_output=True, text=True
        )
        
        # Eliminar archivo temporal
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        if result.returncode != 0:
            logger.error(f"Error al añadir versión al secreto {secret_name}: {result.stderr}")
            return False
        
        logger.info(f"Secreto {secret_name} creado/actualizado correctamente")
        return True
    
    except Exception as e:
        logger.error(f"Error al crear/actualizar secreto {secret_name}: {str(e)}")
        return False

def deploy_to_cloud(project_id, region, service_name, config_file):
    """Despliega la aplicación en Google Cloud Run."""
    try:
        # Verificar que el archivo de configuración existe
        if not os.path.exists(config_file):
            logger.error(f"Archivo de configuración {config_file} no encontrado")
            return False
        
        # Desplegar aplicación
        logger.info(f"Desplegando aplicación {service_name} en Google Cloud...")
        
        result = subprocess.run(
            ['gcloud', 'app', 'deploy', config_file, '--project', project_id, '--quiet'],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Error al desplegar aplicación: {result.stderr}")
            return False
        
        logger.info(f"Aplicación {service_name} desplegada correctamente")
        
        # Mostrar URL de la aplicación
        result = subprocess.run(
            ['gcloud', 'app', 'describe', '--project', project_id],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            logger.info(f"Aplicación desplegada en: https://{service_name}-dot-{project_id}.appspot.com")
        
        return True
    
    except Exception as e:
        logger.error(f"Error al desplegar aplicación: {str(e)}")
        return False

def prepare_environment(args):
    """Prepara el entorno para el despliegue."""
    try:
        # Verificar gcloud
        if not check_gcloud_installed():
            return False
        
        # Configurar proyecto
        result = subprocess.run(
            ['gcloud', 'config', 'set', 'project', args.project_id],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Error al configurar proyecto: {result.stderr}")
            return False
        
        # Habilitar APIs necesarias
        apis = [
            'appengine.googleapis.com',
            'cloudbuild.googleapis.com',
            'cloudresourcemanager.googleapis.com',
            'storage-api.googleapis.com',
            'secretmanager.googleapis.com',
            'logging.googleapis.com',
            'monitoring.googleapis.com'
        ]
        
        for api in apis:
            logger.info(f"Habilitando API: {api}")
            result = subprocess.run(
                ['gcloud', 'services', 'enable', api, '--project', args.project_id],
                capture_output=True, text=True
            )
            
            if result.returncode != 0:
                logger.warning(f"Error al habilitar API {api}: {result.stderr}")
        
        # Crear bucket si se proporciona nombre
        bucket_name = args.bucket_name or f"{args.project_id}-sol-bot-data"
        if not create_bucket(args.project_id, bucket_name, args.region):
            logger.warning("No se pudo crear el bucket, continuando...")
        
        # Actualizar archivo de configuración
        if not update_config_file(args.config_file, args.project_id, args.region, bucket_name):
            logger.warning("No se pudo actualizar el archivo de configuración, continuando...")
        
        # Crear secretos si se solicita
        if args.create_secrets:
            if args.exchange_key_file:
                create_secret(args.project_id, 'exchange-api-key', args.exchange_key_file)
            
            if args.exchange_secret_file:
                create_secret(args.project_id, 'exchange-api-secret', args.exchange_secret_file)
            
            if args.telegram_token_file:
                create_secret(args.project_id, 'telegram-token', args.telegram_token_file)
            
            if args.telegram_chat_id_file:
                create_secret(args.project_id, 'telegram-chat-id', args.telegram_chat_id_file)
        
        logger.info("Entorno preparado correctamente")
        return True
    
    except Exception as e:
        logger.error(f"Error al preparar entorno: {str(e)}")
        return False

def main():
    """Función principal."""
    # Parsear argumentos
    args = parse_arguments()
    
    # Ejecutar según modo
    if args.update_config:
        bucket_name = args.bucket_name or f"{args.project_id}-sol-bot-data"
        update_config_file(args.config_file, args.project_id, args.region, bucket_name)
    
    if args.prepare:
        prepare_environment(args)
    
    if args.deploy:
        # Preparar entorno primero
        if prepare_environment(args):
            # Desplegar aplicación
            deploy_to_cloud(args.project_id, args.region, args.service_name, args.config_file)
    
    # Si no se especificó ningún modo, mostrar ayuda
    if not (args.prepare or args.deploy or args.update_config):
        logger.info("No se especificó ningún modo de ejecución. Use --prepare, --deploy o --update-config.")
        logger.info("Ejecute con --help para ver todas las opciones.")

if __name__ == "__main__":
    main()

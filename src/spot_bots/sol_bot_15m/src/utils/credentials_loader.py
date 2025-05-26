#!/usr/bin/env python3
"""
Cargador de credenciales para el bot SOL
Lee las credenciales desde credentials.json en lugar de .env
"""

import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_credentials():
    """
    Carga las credenciales desde credentials.json y las hace disponibles como variables de entorno.
    
    Returns:
        dict: Diccionario con las credenciales cargadas
    """
    # Buscar el archivo credentials.json en varias ubicaciones posibles
    possible_paths = [
        # Ruta en el servidor
        Path("/home/edisonbautistaruiz2025/trading-bots-api/credentials.json"),
        # Ruta en desarrollo local
        Path(os.path.expanduser("~")) / "Documents" / "GitHub" / "trading-bots-api" / "credentials.json",
        # Ruta relativa desde el directorio actual
        Path(__file__).parent.parent.parent.parent.parent / "trading-bots-api" / "credentials.json"
    ]
    
    credentials_file = None
    for path in possible_paths:
        if path.exists():
            credentials_file = path
            logger.info(f"Archivo credentials.json encontrado en: {path}")
            break
    
    if not credentials_file:
        logger.error("No se encontró el archivo credentials.json")
        return {}
    
    try:
        with open(credentials_file, 'r', encoding='utf-8') as f:
            credentials = json.load(f)
            
        # Extraer las variables de entorno
        env_vars = credentials.get('env', {})
        
        # Establecer las variables de entorno para que sean accesibles por os.getenv
        for key, value in env_vars.items():
            os.environ[key] = str(value)
            
        logger.info(f"Credenciales cargadas correctamente: {', '.join(env_vars.keys())}")
        return env_vars
        
    except Exception as e:
        logger.error(f"Error al cargar las credenciales: {str(e)}")
        return {}

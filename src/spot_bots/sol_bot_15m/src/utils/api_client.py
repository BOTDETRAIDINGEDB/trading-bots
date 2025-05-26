# api_client.py
import os
import logging
import json
import requests
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class APIClient:
    """Cliente para interactuar con la API de trading-bots."""
    
    def __init__(self, api_url=None, bot_id=None, secret_key=None):
        """
        Inicializa el cliente de la API.
        
        Args:
            api_url (str, optional): URL de la API. Si no se proporciona, se busca en variables de entorno.
            bot_id (str, optional): ID del bot. Si no se proporciona, se busca en variables de entorno.
            secret_key (str, optional): Clave secreta para autenticación. Si no se proporciona, se busca en variables de entorno.
        """
        self.api_url = api_url or os.getenv('API_URL', 'http://localhost:5000')
        self.bot_id = bot_id or os.getenv('BOT_ID', 'sol_bot_15m')
        self.secret_key = secret_key or os.getenv('API_SECRET_KEY')
        
        if not self.secret_key:
            logger.warning("Clave secreta de API no proporcionada. La autenticación puede fallar.")
        
        self.enabled = True
        logger.info(f"Cliente de API inicializado. URL: {self.api_url}, Bot ID: {self.bot_id}")
    
    def verify_connection(self):
        """
        Verifica la conexión con la API.
        
        Returns:
            bool: True si la conexión es exitosa, False en caso contrario.
        """
        if not self.enabled:
            logger.warning("Cliente de API no está habilitado.")
            return False
        
        try:
            response = requests.get(f"{self.api_url}/health", timeout=10)
            if response.status_code == 200:
                logger.info("Conexión con API verificada exitosamente")
                return True
            else:
                logger.error(f"Error al verificar conexión con API: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error al verificar conexión con API: {str(e)}")
            return False
    
    def register_bot(self, bot_name, bot_type, symbol, interval):
        """
        Registra el bot en la API.
        
        Args:
            bot_name (str): Nombre del bot.
            bot_type (str): Tipo de bot (ej. 'spot', 'futures').
            symbol (str): Par de trading.
            interval (str): Intervalo de tiempo.
            
        Returns:
            bool: True si el registro fue exitoso, False en caso contrario.
        """
        if not self.enabled:
            logger.warning("Cliente de API no está habilitado. Bot no registrado.")
            return False
        
        try:
            url = f"{self.api_url}/bots/register"
            headers = {'Content-Type': 'application/json'}
            if self.secret_key:
                headers['X-API-KEY'] = self.secret_key
            
            data = {
                'bot_id': self.bot_id,
                'name': bot_name,
                'type': bot_type,
                'symbol': symbol,
                'interval': interval,
                'status': 'active',
                'created_at': datetime.now().isoformat()
            }
            
            response = requests.post(url, json=data, headers=headers, timeout=10)
            
            if response.status_code in [200, 201]:
                logger.info(f"Bot registrado exitosamente en la API: {self.bot_id}")
                return True
            else:
                logger.error(f"Error al registrar bot en API: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error al registrar bot en API: {str(e)}")
            return False
    
    def update_bot_status(self, status, metrics=None):
        """
        Actualiza el estado del bot en la API.
        
        Args:
            status (str): Estado del bot ('active', 'paused', 'stopped', 'error').
            metrics (dict, optional): Métricas de rendimiento del bot.
            
        Returns:
            bool: True si la actualización fue exitosa, False en caso contrario.
        """
        if not self.enabled:
            logger.warning("Cliente de API no está habilitado. Estado no actualizado.")
            return False
        
        try:
            url = f"{self.api_url}/bots/{self.bot_id}/status"
            headers = {'Content-Type': 'application/json'}
            if self.secret_key:
                headers['X-API-KEY'] = self.secret_key
            
            data = {
                'status': status,
                'updated_at': datetime.now().isoformat()
            }
            
            if metrics:
                data['metrics'] = metrics
            
            response = requests.put(url, json=data, headers=headers, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"Estado del bot actualizado exitosamente: {status}")
                return True
            else:
                logger.error(f"Error al actualizar estado del bot: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error al actualizar estado del bot: {str(e)}")
            return False
    
    def report_trade(self, trade_data):
        """
        Reporta una operación de trading a la API.
        
        Args:
            trade_data (dict): Datos de la operación.
            
        Returns:
            bool: True si el reporte fue exitoso, False en caso contrario.
        """
        if not self.enabled:
            logger.warning("Cliente de API no está habilitado. Operación no reportada.")
            return False
        
        try:
            url = f"{self.api_url}/bots/{self.bot_id}/trades"
            headers = {'Content-Type': 'application/json'}
            if self.secret_key:
                headers['X-API-KEY'] = self.secret_key
            
            # Asegurarse de que el trade_data incluya un timestamp
            if 'timestamp' not in trade_data:
                trade_data['timestamp'] = datetime.now().isoformat()
            
            response = requests.post(url, json=trade_data, headers=headers, timeout=10)
            
            if response.status_code in [200, 201]:
                logger.info(f"Operación reportada exitosamente: {trade_data.get('type', 'unknown')} a precio {trade_data.get('price', 'unknown')}")
                return True
            else:
                logger.error(f"Error al reportar operación: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error al reportar operación: {str(e)}")
            return False
    
    def get_bot_commands(self):
        """
        Obtiene comandos pendientes para el bot desde la API.
        
        Returns:
            list: Lista de comandos pendientes o None si hay un error.
        """
        if not self.enabled:
            logger.warning("Cliente de API no está habilitado. No se pueden obtener comandos.")
            return None
        
        try:
            url = f"{self.api_url}/bots/{self.bot_id}/commands"
            headers = {}
            if self.secret_key:
                headers['X-API-KEY'] = self.secret_key
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                commands = response.json()
                if commands:
                    logger.info(f"Obtenidos {len(commands)} comandos pendientes")
                return commands
            else:
                logger.error(f"Error al obtener comandos: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error al obtener comandos: {str(e)}")
            return None
    
    def acknowledge_command(self, command_id, status, message=None):
        """
        Confirma la ejecución de un comando.
        
        Args:
            command_id (str): ID del comando.
            status (str): Estado de la ejecución ('success', 'failed').
            message (str, optional): Mensaje adicional.
            
        Returns:
            bool: True si la confirmación fue exitosa, False en caso contrario.
        """
        if not self.enabled:
            logger.warning("Cliente de API no está habilitado. No se puede confirmar comando.")
            return False
        
        try:
            url = f"{self.api_url}/bots/{self.bot_id}/commands/{command_id}"
            headers = {'Content-Type': 'application/json'}
            if self.secret_key:
                headers['X-API-KEY'] = self.secret_key
            
            data = {
                'status': status,
                'processed_at': datetime.now().isoformat()
            }
            
            if message:
                data['message'] = message
            
            response = requests.put(url, json=data, headers=headers, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"Comando {command_id} confirmado con estado: {status}")
                return True
            else:
                logger.error(f"Error al confirmar comando: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error al confirmar comando: {str(e)}")
            return False

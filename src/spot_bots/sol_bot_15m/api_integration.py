#!/usr/bin/env python3
"""
Integración con la API para el bot SOL
Permite un monitoreo más detallado y control remoto del bot
"""

import os
import sys
import json
import logging
import argparse
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Ajustar el PYTHONPATH para encontrar los módulos correctamente
bot_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(bot_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class APIIntegration:
    """Integración con la API para el bot SOL."""
    
    def __init__(self, state_file, api_url=None, api_key=None, jwt_token=None):
        """
        Inicializa la integración con la API.
        
        Args:
            state_file (str): Ruta al archivo de estado del bot.
            api_url (str, optional): URL base de la API.
            api_key (str, optional): Clave de la API.
            jwt_token (str, optional): Token JWT para autenticación.
        """
        self.state_file = state_file
        
        # Cargar variables de entorno si no se proporcionan
        if not api_url:
            api_url = os.getenv('API_URL', 'https://tradebotscentral.com/api')
        if not api_key:
            api_key = os.getenv('API_KEY')
        if not jwt_token:
            # Intentar cargar desde auth_config.json
            auth_config_file = os.path.join(os.path.dirname(bot_dir), 'trading-bots-api', 'auth_config.json')
            if os.path.exists(auth_config_file):
                try:
                    with open(auth_config_file, 'r') as f:
                        auth_config = json.load(f)
                        jwt_token = auth_config.get('jwt_token')
                except Exception as e:
                    logger.error(f"Error al cargar auth_config.json: {str(e)}")
        
        self.api_url = api_url
        self.api_key = api_key
        self.jwt_token = jwt_token
        self.bot_id = 'sol_bot_15m'
        
        logger.info(f"Integración con API inicializada. URL: {api_url}")
    
    def load_state(self):
        """
        Carga el estado del bot desde el archivo de estado.
        
        Returns:
            dict: Estado del bot, o None si no se pudo cargar.
        """
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Archivo de estado no encontrado: {self.state_file}")
                return None
        except Exception as e:
            logger.error(f"Error al cargar el estado: {str(e)}")
            return None
    
    def get_headers(self):
        """
        Obtiene los headers para las peticiones a la API.
        
        Returns:
            dict: Headers para las peticiones.
        """
        headers = {
            'Content-Type': 'application/json'
        }
        
        if self.api_key:
            headers['X-API-Key'] = self.api_key
        
        if self.jwt_token:
            headers['Authorization'] = f'Bearer {self.jwt_token}'
        
        return headers
    
    def update_bot_status(self, status=None, metrics=None):
        """
        Actualiza el estado del bot en la API.
        
        Args:
            status (str, optional): Estado del bot ('active', 'inactive', 'error').
            metrics (dict, optional): Métricas de rendimiento.
            
        Returns:
            bool: True si se actualizó correctamente, False en caso contrario.
        """
        if not self.jwt_token:
            logger.error("No se puede actualizar el estado sin token JWT")
            return False
        
        # Cargar estado actual
        state = self.load_state()
        if not state:
            logger.error("No se pudo cargar el estado del bot")
            return False
        
        # Preparar datos para la API
        data = {
            'status': status or 'active',
            'metrics': {}
        }
        
        # Añadir métricas básicas
        if 'performance_metrics' in state:
            metrics = state['performance_metrics']
            data['metrics'] = {
                'win_rate': metrics.get('win_rate', 0),
                'total_trades': metrics.get('total_trades', 0),
                'winning_trades': metrics.get('winning_trades', 0),
                'losing_trades': metrics.get('losing_trades', 0),
                'profit_factor': metrics.get('profit_factor', 0),
                'total_profit': metrics.get('total_profit', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'current_balance': state.get('current_balance', 0),
                'last_update': datetime.now().isoformat()
            }
        
        # Añadir métricas adicionales si se proporcionan
        if metrics:
            data['metrics'].update(metrics)
        
        # Enviar actualización a la API
        try:
            url = f"{self.api_url}/bots/{self.bot_id}/status"
            response = requests.post(
                url,
                headers=self.get_headers(),
                json=data
            )
            
            if response.status_code == 200:
                logger.info(f"Estado del bot actualizado correctamente en la API")
                return True
            else:
                logger.error(f"Error al actualizar estado del bot: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error al comunicarse con la API: {str(e)}")
            return False
    
    def register_trade(self, trade):
        """
        Registra una operación en la API.
        
        Args:
            trade (dict): Detalles de la operación.
            
        Returns:
            bool: True si se registró correctamente, False en caso contrario.
        """
        if not self.jwt_token:
            logger.error("No se puede registrar operación sin token JWT")
            return False
        
        # Preparar datos para la API
        data = {
            'bot_id': self.bot_id,
            'trade_id': trade.get('id'),
            'type': trade.get('type', 'long'),
            'symbol': 'SOLUSDT',
            'entry_price': trade.get('entry_price', 0),
            'entry_time': trade.get('entry_time', datetime.now().isoformat()),
            'position_size': trade.get('position_size', 0),
            'stop_loss': trade.get('stop_loss', 0),
            'take_profit': trade.get('take_profit', 0),
            'exit_price': trade.get('exit_price', 0),
            'exit_time': trade.get('exit_time'),
            'profit_loss': trade.get('profit_loss', 0),
            'profit_loss_pct': trade.get('profit_loss_pct', 0),
            'status': trade.get('status', 'open')
        }
        
        # Enviar operación a la API
        try:
            url = f"{self.api_url}/trades"
            response = requests.post(
                url,
                headers=self.get_headers(),
                json=data
            )
            
            if response.status_code == 200:
                logger.info(f"Operación registrada correctamente en la API: ID {trade.get('id')}")
                return True
            else:
                logger.error(f"Error al registrar operación: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error al comunicarse con la API: {str(e)}")
            return False
    
    def check_commands(self):
        """
        Verifica si hay comandos pendientes en la API.
        
        Returns:
            dict: Comandos pendientes, o None si no hay comandos o hubo un error.
        """
        if not self.jwt_token:
            logger.error("No se pueden verificar comandos sin token JWT")
            return None
        
        try:
            url = f"{self.api_url}/bots/{self.bot_id}/commands"
            response = requests.get(
                url,
                headers=self.get_headers()
            )
            
            if response.status_code == 200:
                commands = response.json().get('data', [])
                if commands:
                    logger.info(f"Comandos pendientes encontrados: {len(commands)}")
                    return commands
                else:
                    logger.debug("No hay comandos pendientes")
                    return []
            else:
                logger.error(f"Error al verificar comandos: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error al comunicarse con la API: {str(e)}")
            return None
    
    def acknowledge_command(self, command_id, status, message=None):
        """
        Confirma la ejecución de un comando.
        
        Args:
            command_id (str): ID del comando.
            status (str): Estado de la ejecución ('success', 'error').
            message (str, optional): Mensaje adicional.
            
        Returns:
            bool: True si se confirmó correctamente, False en caso contrario.
        """
        if not self.jwt_token:
            logger.error("No se puede confirmar comando sin token JWT")
            return False
        
        try:
            url = f"{self.api_url}/bots/{self.bot_id}/commands/{command_id}/ack"
            data = {
                'status': status,
                'message': message or f"Comando ejecutado con estado: {status}"
            }
            
            response = requests.post(
                url,
                headers=self.get_headers(),
                json=data
            )
            
            if response.status_code == 200:
                logger.info(f"Comando {command_id} confirmado con estado: {status}")
                return True
            else:
                logger.error(f"Error al confirmar comando: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error al comunicarse con la API: {str(e)}")
            return False
    
    def sync_trades(self):
        """
        Sincroniza las operaciones del bot con la API.
        
        Returns:
            bool: True si se sincronizó correctamente, False en caso contrario.
        """
        # Cargar estado actual
        state = self.load_state()
        if not state:
            logger.error("No se pudo cargar el estado del bot")
            return False
        
        # Obtener operaciones
        trades = state.get('trades', [])
        if not trades:
            logger.info("No hay operaciones para sincronizar")
            return True
        
        # Registrar cada operación en la API
        success = True
        for trade in trades:
            if not self.register_trade(trade):
                success = False
        
        return success
    
    def run(self, sync_trades=True, update_status=True, check_commands=True):
        """
        Ejecuta la integración con la API.
        
        Args:
            sync_trades (bool): Si es True, sincroniza las operaciones.
            update_status (bool): Si es True, actualiza el estado del bot.
            check_commands (bool): Si es True, verifica comandos pendientes.
            
        Returns:
            bool: True si se ejecutó correctamente, False en caso contrario.
        """
        success = True
        
        # Sincronizar operaciones
        if sync_trades:
            if not self.sync_trades():
                success = False
        
        # Actualizar estado
        if update_status:
            if not self.update_bot_status():
                success = False
        
        # Verificar comandos
        if check_commands:
            commands = self.check_commands()
            if commands is None:
                success = False
            elif commands:
                logger.info(f"Hay {len(commands)} comandos pendientes para procesar")
                # Aquí se podrían procesar los comandos, pero eso requeriría
                # modificar el bot principal para que pueda recibir comandos
        
        return success

def parse_arguments():
    """Parsea los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description='Integración con API para el bot SOL')
    parser.add_argument('--state-file', type=str, default='sol_bot_15min_state.json', help='Archivo de estado del bot')
    parser.add_argument('--api-url', type=str, help='URL base de la API')
    parser.add_argument('--api-key', type=str, help='Clave de la API')
    parser.add_argument('--jwt-token', type=str, help='Token JWT para autenticación')
    parser.add_argument('--no-sync', action='store_true', help='No sincronizar operaciones')
    parser.add_argument('--no-status', action='store_true', help='No actualizar estado')
    parser.add_argument('--no-commands', action='store_true', help='No verificar comandos')
    return parser.parse_args()

def main():
    """Función principal."""
    # Cargar variables de entorno
    load_dotenv()
    
    # Parsear argumentos
    args = parse_arguments()
    
    # Inicializar integración
    api = APIIntegration(
        args.state_file,
        api_url=args.api_url,
        api_key=args.api_key,
        jwt_token=args.jwt_token
    )
    
    # Ejecutar integración
    success = api.run(
        sync_trades=not args.no_sync,
        update_status=not args.no_status,
        check_commands=not args.no_commands
    )
    
    if success:
        logger.info("Integración con API ejecutada correctamente")
    else:
        logger.error("Error al ejecutar integración con API")

if __name__ == "__main__":
    main()

# telegram_notifier.py
import os
import logging
import requests
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class TelegramNotifier:
    """Envía notificaciones a través de Telegram."""
    
    def __init__(self, token=None, chat_id=None):
        """
        Inicializa el notificador de Telegram.
        
        Args:
            token (str, optional): Token del bot de Telegram. Si no se proporciona, se busca en variables de entorno.
            chat_id (str, optional): ID del chat donde enviar mensajes. Si no se proporciona, se busca en variables de entorno.
        """
        self.token = token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        
        if not self.token or not self.chat_id:
            logger.warning("Token de Telegram o Chat ID no proporcionados. Las notificaciones no serán enviadas.")
            self.enabled = False
        else:
            self.enabled = True
            self.base_url = f"https://api.telegram.org/bot{self.token}"
            logger.info("Notificador de Telegram inicializado.")
    
    def verify_connection(self):
        """
        Verifica la conexión con la API de Telegram.
        
        Returns:
            bool: True si la conexión es exitosa, False en caso contrario.
        """
        if not self.enabled:
            logger.warning("Notificador de Telegram no está habilitado.")
            return False
        
        try:
            response = requests.get(f"{self.base_url}/getMe", timeout=10)
            if response.status_code == 200:
                logger.info("Conexión con Telegram verificada exitosamente")
                return True
            else:
                logger.error(f"Error al verificar conexión con Telegram: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error al verificar conexión con Telegram: {str(e)}")
            return False
    
    def send_message(self, message, parse_mode="Markdown"):
        """
        Envía un mensaje a través de Telegram.
        
        Args:
            message (str): Mensaje a enviar.
            parse_mode (str, optional): Modo de parseo del mensaje ('Markdown' o 'HTML').
            
        Returns:
            bool: True si el mensaje se envió correctamente, False en caso contrario.
        """
        if not self.enabled:
            logger.warning(f"Mensaje no enviado (Telegram deshabilitado): {message}")
            return False
        
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"Mensaje enviado exitosamente: {message[:50]}...")
                return True
            else:
                logger.error(f"Error al enviar mensaje: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error al enviar mensaje: {str(e)}")
            return False
    
    def send_trade_notification(self, trade_type, symbol, price, size, profit_loss=None):
        """
        Envía una notificación sobre una operación de trading.
        
        Args:
            trade_type (str): Tipo de operación ('entry', 'exit', 'stop_loss', 'take_profit').
            symbol (str): Par de trading.
            price (float): Precio de la operación.
            size (float): Tamaño de la posición.
            profit_loss (float, optional): Ganancia o pérdida de la operación (solo para salidas).
            
        Returns:
            bool: True si la notificación se envió correctamente, False en caso contrario.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if trade_type == 'entry':
            message = f"🚀 *ENTRADA EN OPERACIÓN* 🚀\n\n" \
                      f"*Par:* {symbol}\n" \
                      f"*Precio:* {price:.4f}\n" \
                      f"*Tamaño:* {size:.4f}\n" \
                      f"*Fecha:* {timestamp}"
        
        elif trade_type in ['exit', 'stop_loss', 'take_profit']:
            emoji = "🔴" if profit_loss and profit_loss < 0 else "🟢"
            pl_text = f"{profit_loss:.4f} ({(profit_loss / (price * size) * 100):.2f}%)" if profit_loss is not None else "N/A"
            
            if trade_type == 'stop_loss':
                title = "STOP LOSS ACTIVADO"
            elif trade_type == 'take_profit':
                title = "TAKE PROFIT ALCANZADO"
            else:
                title = "SALIDA DE OPERACIÓN"
            
            message = f"{emoji} *{title}* {emoji}\n\n" \
                      f"*Par:* {symbol}\n" \
                      f"*Precio:* {price:.4f}\n" \
                      f"*Tamaño:* {size:.4f}\n" \
                      f"*P/L:* {pl_text}\n" \
                      f"*Fecha:* {timestamp}"
        
        else:
            logger.warning(f"Tipo de operación desconocido: {trade_type}")
            return False
        
        return self.send_message(message)
    
    def send_error_notification(self, error_message):
        """
        Envía una notificación de error.
        
        Args:
            error_message (str): Mensaje de error.
            
        Returns:
            bool: True si la notificación se envió correctamente, False en caso contrario.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"⚠️ *ERROR* ⚠️\n\n" \
                  f"{error_message}\n\n" \
                  f"*Fecha:* {timestamp}"
        
        return self.send_message(message)
    
    def send_status_update(self, balance, performance_metrics, symbol):
        """
        Envía una actualización de estado del bot.
        
        Args:
            balance (float): Balance actual.
            performance_metrics (dict): Métricas de rendimiento.
            symbol (str): Par de trading.
            
        Returns:
            bool: True si la notificación se envió correctamente, False en caso contrario.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"📊 *ACTUALIZACIÓN DE ESTADO* 📊\n\n" \
                  f"*Par:* {symbol}\n" \
                  f"*Balance:* {balance:.2f} USDT\n" \
                  f"*Operaciones:* {performance_metrics['total_trades']}\n" \
                  f"*Win Rate:* {performance_metrics['win_rate']:.2f}%\n" \
                  f"*P/L Total:* {performance_metrics['total_profit']:.2f} USDT\n" \
                  f"*Drawdown:* {performance_metrics['current_drawdown']:.2f}%\n" \
                  f"*Fecha:* {timestamp}"
        
        return self.send_message(message)

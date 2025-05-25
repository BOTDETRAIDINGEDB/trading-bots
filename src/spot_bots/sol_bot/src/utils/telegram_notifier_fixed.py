import logging
import requests
import datetime
from typing import Dict, Optional, Union, Any
from requests.exceptions import RequestException
from tenacity import retry, stop_after_attempt, wait_exponential

# Configuración de logging
logger = logging.getLogger(__name__)

# Constantes
MAX_RETRIES = 3
BASE_WAIT = 2
MAX_WAIT = 10
API_TIMEOUT = 10

class TelegramNotifier:
    """Clase para manejar notificaciones de Telegram de manera robusta y profesional.
    
    Attributes:
        token (str): Token del bot de Telegram
        chat_id (str): ID del chat donde se enviarán los mensajes
        base_url (str): URL base de la API de Telegram
        last_buy_price (float): Último precio de compra registrado
        last_sol_price (float): Último precio de SOL registrado
    """
    
    def __init__(self, token: str, chat_id: str):
        if not token or not chat_id:
            raise ValueError("Token y chat_id son requeridos")
            
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.last_buy_price: Optional[float] = None
        self.last_sol_price: Optional[float] = None
        
        # Verificar conexión al inicializar
        try:
            self._verify_connection()
        except Exception as e:
            logger.warning(f"No se pudo verificar la conexión con Telegram: {e}")
        
    def _verify_connection(self) -> bool:
        """Verifica la conexión con Telegram."""
        try:
            response = requests.get(f"{self.base_url}/getMe", timeout=API_TIMEOUT)
            response.raise_for_status()
            logger.info("Conexión con Telegram verificada exitosamente")
            return True
        except Exception as e:
            logger.error(f"Error al verificar conexión con Telegram: {e}")
            return False
            
    @retry(stop=stop_after_attempt(MAX_RETRIES),
           wait=wait_exponential(multiplier=BASE_WAIT, max=MAX_WAIT))
    def send_message(self, text: str) -> Dict:
        """Envía un mensaje a través de Telegram.
        
        Args:
            text (str): Texto del mensaje a enviar
            
        Returns:
            Dict: Respuesta de la API de Telegram
            
        Raises:
            RequestException: Si hay un error al enviar el mensaje
        """
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, json=payload, timeout=API_TIMEOUT)
            response.raise_for_status()
            
            logger.debug(f"Mensaje enviado: {text[:50]}...")
            return response.json()
            
        except RequestException as e:
            logger.error(f"Error al enviar mensaje: {e}")
            raise
            
    def send_balance_update(self, balance: Dict, pnl: float = 0) -> Dict:
        """Envía una actualización del balance.
        
        Args:
            balance (Dict): Balance actual
            pnl (float): Beneficio/pérdida total
            
        Returns:
            Dict: Respuesta de la API de Telegram
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Formatear balance
        balance_text = ""
        for asset, amount in balance.items():
            balance_text += f"• {asset}: `{amount:.4f}`\n"
            
        # Determinar emoji según PnL
        pnl_emoji = "📈" if pnl >= 0 else "📉"
        
        # Añadir identificación clara del bot
        message = f"*💰 Balance Actual - BOT SOL*\n"
        message += f"{balance_text}\n"
        message += f"{pnl_emoji} PnL Total: `{pnl:.2f} USDT`\n"
        
        # Añadir precio actual de SOL si está disponible
        if hasattr(self, 'last_sol_price') and self.last_sol_price is not None:
            message += f"📊 Precio SOL: `{self.last_sol_price} USDT`\n"
            
        message += f"⏰ Fecha: `{timestamp}`"
        
        return self.send_message(message)
        
    def send_trade_notification(self, symbol: str, side: str, price: float, quantity: float, total_value: float) -> Dict:
        """Envía una notificación de operación de trading.
        
        Args:
            symbol (str): Símbolo de la operación (ej: SOLUSDT)
            side (str): Lado de la operación (BUY/SELL)
            price (float): Precio de la operación
            quantity (float): Cantidad de la operación
            total_value (float): Valor total de la operación
            
        Returns:
            Dict: Respuesta de la API de Telegram
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Extraer base y quote asset del símbolo
        if len(symbol) >= 7 and symbol.endswith("USDT"):
            base_asset = symbol[:-4]
            quote_asset = "USDT"
        else:
            # Intentar dividir el símbolo en base y quote
            for i in range(3, len(symbol)):
                if symbol[i:] in ["BTC", "ETH", "USDT", "BNB", "BUSD"]:
                    base_asset = symbol[:i]
                    quote_asset = symbol[i:]
                    break
            else:
                base_asset = symbol
                quote_asset = "USDT"
        
        # Determinar emoji según el lado
        if side == "BUY":
            emoji = "🟢"
        elif side == "SELL":
            emoji = "🔴"
        else:
            emoji = "⚪"
        
        # Añadir identificación clara del bot
        message = f"*{emoji}: {symbol} - BOT SOL*\n"
        message += f"📊 Precio: `{price} {quote_asset}`\n"
        message += f"📈 Cantidad: `{quantity} {base_asset}`\n"
        message += f"💰 Valor Total: `{total_value} {quote_asset}`\n"
        
        # Añadir información sobre beneficio/pérdida si es una venta y tenemos el precio de compra
        if side == "SELL" and hasattr(self, 'last_buy_price') and self.last_buy_price is not None:
            profit = (price - self.last_buy_price) * quantity
            profit_percentage = ((price / self.last_buy_price) - 1) * 100
            
            profit_emoji = "📈" if profit >= 0 else "📉"
            message += f"{profit_emoji} Beneficio: `{profit:.2f} {quote_asset} ({profit_percentage:.2f}%)`\n"
            message += f"🔄 Precio de entrada: `{self.last_buy_price} {quote_asset}`\n"
        
        # Si es una compra, guardamos el precio para futuros cálculos
        if side == "BUY":
            self.last_buy_price = price
            setattr(self, f'last_{base_asset.lower()}_price', price)
        
        message += f"⏰ Fecha: `{timestamp}`"
        
        return self.send_message(message)
    
    def send_prediction_notification(self, symbol, signal, probability, threshold):
        """Envía una notificación de predicción"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if signal == "BUY":
            emoji = "🟢"
            action = "COMPRAR"
        elif signal == "SELL":
            emoji = "🔴"
            action = "VENDER"
        else:
            emoji = "⚪"
            action = "MANTENER"
        
        confidence = probability * 100
        threshold_percent = threshold * 100
        
        # Añadir identificación clara del bot
        message = f"*{emoji} Predicción para {symbol} - BOT SOL*\n"
        message += f"📊 Señal: `{action}`\n"
        message += f"🎯 Confianza: `{confidence:.2f}%`\n"
        message += f"📏 Umbral: `{threshold_percent:.2f}%`\n"
        message += f"⏰ Fecha: `{timestamp}`"
        
        return self.send_message(message)
    
    def send_error_notification(self, error_message: str, error_type: str = "Error") -> Dict:
        """Envía una notificación de error.
        
        Args:
            error_message (str): Mensaje de error
            error_type (str): Tipo de error
            
        Returns:
            Dict: Respuesta de la API de Telegram
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"*⚠️ {error_type} - BOT SOL*\n"
        message += f"```\n{error_message}\n```\n"
        message += f"⏰ Fecha: `{timestamp}`"
        
        return self.send_message(message)
    
    def send_daily_summary(self, balance: Dict, metrics: Dict) -> Dict:
        """Envía un resumen diario.
        
        Args:
            balance (Dict): Balance actual
            metrics (Dict): Métricas de rendimiento
            
        Returns:
            Dict: Respuesta de la API de Telegram
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Formatear balance
        balance_text = ""
        for asset, amount in balance.items():
            balance_text += f"• {asset}: `{amount:.4f}`\n"
            
        # Formatear métricas
        total_trades = metrics.get("total_trades", 0)
        win_rate = metrics.get("win_rate", 0) * 100
        total_pnl = metrics.get("total_pnl", 0)
        
        # Determinar emoji según PnL
        pnl_emoji = "📈" if total_pnl >= 0 else "📉"
        
        # Añadir identificación clara del bot
        message = f"*📅 Resumen Diario {timestamp} - BOT SOL*\n\n"
        message += f"*💰 Balance:*\n{balance_text}\n"
        message += f"*📊 Métricas:*\n"
        message += f"• Operaciones: `{total_trades}`\n"
        message += f"• Tasa de éxito: `{win_rate:.2f}%`\n"
        message += f"• {pnl_emoji} PnL Total: `{total_pnl:.2f} USDT`\n"
        
        # Añadir precio actual de SOL si está disponible
        if hasattr(self, 'last_sol_price') and self.last_sol_price is not None:
            message += f"\n📊 Precio SOL: `{self.last_sol_price} USDT`\n"
            
        return self.send_message(message)

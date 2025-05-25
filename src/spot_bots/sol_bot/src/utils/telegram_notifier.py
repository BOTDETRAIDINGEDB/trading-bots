import logging
import requests
import time
from datetime import datetime
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
        self._verify_connection()
        
    @retry(stop=stop_after_attempt(MAX_RETRIES),
           wait=wait_exponential(multiplier=BASE_WAIT, max=MAX_WAIT))
    def send_message(self, message: str) -> bool:
        """Envía un mensaje a Telegram con reintentos automáticos.
        
        Args:
            message (str): Mensaje a enviar, puede contener formato Markdown
            
        Returns:
            bool: True si el mensaje se envió correctamente
            
        Raises:
            RequestException: Si hay un error de red
            ValueError: Si el mensaje está vacío
        """
        """Envía un mensaje a Telegram"""
        if not message:
            raise ValueError("El mensaje no puede estar vacío")
            
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, data=data, timeout=API_TIMEOUT)
            response.raise_for_status()
            
            logger.debug(f"Mensaje enviado exitosamente: {message[:50]}...")
            return True
            
        except RequestException as e:
            logger.error(f"Error de red al enviar mensaje: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado al enviar mensaje: {str(e)}")
            raise
    
    def send_trade_notification(self, symbol, side, price, quantity, total_value):
        """Envía una notificación de operación de trading con información enriquecida"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Extraer la moneda base y la cotización del símbolo
        base_asset = symbol[:-4] if symbol.endswith("USDT") else symbol.split('/')[0]
        quote_asset = "USDT"
        
        emoji = "🟢 COMPRA" if side == "BUY" else "🔴 VENTA"
        
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
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
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
    
    def send_balance_update(self, balance, pnl=None):
        """Envía una actualización del balance con información enriquecida"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calcular el valor total en USD
        total_usd_value = 0
        for asset, amount in balance.items():
            if asset == "USDT":
                total_usd_value += amount
            else:
                # Usar el último precio conocido si está disponible
                asset_price = getattr(self, f'last_{asset.lower()}_price', None)
                if asset_price:
                    total_usd_value += amount * asset_price
        
        # Añadir identificación clara del bot
        message = f"*💼 Balance Actualizado - BOT SOL*\n"
        
        # Mostrar cada activo con su valor en USD si es posible
        for asset, amount in balance.items():
            if asset == "USDT":
                message += f"💰 {asset}: `{amount}`\n"
            else:
                asset_price = getattr(self, f'last_{asset.lower()}_price', None)
                if asset_price:
                    usd_value = amount * asset_price
                    message += f"💰 {asset}: `{amount}` ≈ `{usd_value:.2f} USDT`\n"
                else:
                    # Si no tenemos el precio, intentamos obtenerlo del último precio conocido
                    if asset == "SOL" and self.last_buy_price:
                        usd_value = amount * self.last_buy_price
                        message += f"💰 {asset}: `{amount}` ≈ `{usd_value:.2f} USDT` (último precio conocido)\n"
                    else:
                        message += f"💰 {asset}: `{amount}`\n"
        
        # Mostrar valor total del portafolio
        message += f"🏦 Valor Total: `{total_usd_value:.2f} USDT`\n"
        
        if pnl is not None:
            emoji = "📈" if pnl >= 0 else "📉"
            pnl_percentage = (pnl / (total_usd_value - pnl)) * 100 if total_usd_value > pnl else 0
            message += f"{emoji} PnL: `{pnl:.2f} USDT ({pnl_percentage:.2f}%)`\n"
        
        message += f"⏰ Fecha: `{timestamp}`"
        
        return self.send_message(message)
    
    def send_daily_summary(self, total_trades, win_rate, avg_profit, avg_loss, profit_factor, total_pnl, balance):
        """Envía un resumen diario del rendimiento"""
        timestamp = datetime.now().strftime("%Y-%m-%d")
        
        # Añadir identificación clara del bot
        message = f"*📊 Resumen Diario - BOT SOL ({timestamp})*\n\n"
        message += f"*Operaciones:*\n"
        message += f"📈 Total: `{total_trades}`\n"
        message += f"✅ Tasa de éxito: `{win_rate*100:.2f}%`\n\n"
        
        message += f"*Rendimiento:*\n"
        message += f"💹 Beneficio promedio: `{avg_profit:.2f} USDT`\n"
        message += f"📉 Pérdida promedio: `{avg_loss:.2f} USDT`\n"
        message += f"⚖️ Factor de beneficio: `{profit_factor:.2f}`\n"
        message += f"💰 PnL total: `{total_pnl:.2f} USDT`\n\n"
        
        message += f"*Balance:*\n"
        
        # Calcular el valor total en USD
        total_usd_value = 0
        for asset, amount in balance.items():
            if asset == "USDT":
                total_usd_value += amount
                message += f"💰 {asset}: `{amount}`\n"
            else:
                # Usar el último precio conocido si está disponible
                asset_price = getattr(self, f'last_{asset.lower()}_price', None)
                if asset_price:
                    usd_value = amount * asset_price
                    total_usd_value += usd_value
                    message += f"💰 {asset}: `{amount}` ≈ `{usd_value:.2f} USDT`\n"
                else:
                    message += f"💰 {asset}: `{amount}`\n"
        
        message += f"🏦 Valor Total: `{total_usd_value:.2f} USDT`\n"
        
        return self.send_message(message)

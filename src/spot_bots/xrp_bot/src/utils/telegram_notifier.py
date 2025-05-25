import logging
import requests
from datetime import datetime
import traceback

class TelegramNotifier:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.logger = logging.getLogger(__name__)
        self.last_buy_price = None
        self.last_sol_price = None
        self.last_xrp_price = None
        
    def send_message(self, message):
        """Envía un mensaje a Telegram"""
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            response = requests.post(url, data=data)
            if response.status_code != 200:
                self.logger.error(f"Error al enviar mensaje a Telegram: {response.text}")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Excepción al enviar mensaje a Telegram: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def send_trade_notification(self, symbol, side, price, quantity, total_value):
        """Envía una notificación de operación de trading con información enriquecida"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Extraer la moneda base y la cotización del símbolo
        base_asset = symbol[:-4] if symbol.endswith("USDT") else symbol.split('/')[0]
        quote_asset = "USDT"
        
        emoji = "🟢 COMPRA" if side == "BUY" else "🔴 VENTA"
        
        # Añadir identificación clara del bot
        message = f"*{emoji}: {symbol} - BOT XRP*\n"
        message += f"📊 Precio: `{price} {quote_asset}`\n"
        message += f"📈 Cantidad: `{quantity} {base_asset}`\n"
        message += f"💰 Valor Total: `{total_value} {quote_asset}`\n"
        
        # Añadir información sobre beneficio/pérdida si es una venta y tenemos el precio de compra específico para este activo
        asset_key = f'last_{base_asset.lower()}_price'
        if side == "SELL" and hasattr(self, asset_key) and getattr(self, asset_key) is not None:
            last_price = getattr(self, asset_key)
            profit = (price - last_price) * quantity
            profit_percentage = ((price / last_price) - 1) * 100
            
            profit_emoji = "📈" if profit >= 0 else "📉"
            message += f"{profit_emoji} Beneficio: `{profit:.2f} {quote_asset} ({profit_percentage:.2f}%)`\n"
            message += f"🔄 Precio de entrada: `{last_price} {quote_asset}`\n"
        
        # Si es una compra, guardamos el precio para futuros cálculos
        if side == "BUY":
            setattr(self, asset_key, price)
        
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
        message = f"*{emoji} Predicción para {symbol} - BOT XRP*\n"
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
        message = f"*💼 Balance Actualizado - BOT XRP*\n"
        
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
        message = f"*📊 Resumen Diario - BOT XRP ({timestamp})*\n\n"
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

#!/usr/bin/env python3
"""
Notificador de Telegram mejorado para el bot SOL
Envía notificaciones con formato mejorado e información detallada del mercado
"""

import os
import logging
import requests
from datetime import datetime
import time
import json

logger = logging.getLogger(__name__)

class EnhancedTelegramNotifier:
    """Notificador de Telegram mejorado con formato e iconos."""
    
    def __init__(self, token=None, chat_id=None):
        """
        Inicializa el notificador de Telegram.
        
        Args:
            token (str, optional): Token del bot de Telegram.
            chat_id (str, optional): ID del chat de Telegram.
        """
        self.token = token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        
        if not self.token or not self.chat_id:
            logger.warning("Token o chat_id de Telegram no configurados")
        else:
            logger.info("Notificador de Telegram inicializado")
    
    def send_message(self, message, retry=3):
        """
        Envía un mensaje a Telegram.
        
        Args:
            message (str): Mensaje a enviar.
            retry (int): Número de reintentos en caso de error.
            
        Returns:
            bool: True si se envió correctamente, False en caso contrario.
        """
        if not self.token or not self.chat_id:
            logger.warning("No se puede enviar mensaje: Token o chat_id no configurados")
            return False
        
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        data = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }
        
        for attempt in range(retry):
            try:
                response = requests.post(url, data=data)
                if response.status_code == 200:
                    logger.info("Mensaje enviado a Telegram")
                    return True
                else:
                    logger.error(f"Error al enviar mensaje a Telegram: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"Error al enviar mensaje a Telegram: {str(e)}")
            
            if attempt < retry - 1:
                time.sleep(2)  # Esperar antes de reintentar
        
        return False
    
    def format_price(self, price, decimals=2):
        """
        Formatea un precio para mostrar solo los decimales relevantes.
        
        Args:
            price (float): Precio a formatear.
            decimals (int): Número máximo de decimales a mostrar.
            
        Returns:
            str: Precio formateado.
        """
        # Si el precio es menor que 1, mostrar más decimales
        if price < 1:
            return f"{price:.4f}"
        elif price < 10:
            return f"{price:.3f}"
        else:
            return f"{price:.2f}"
    
    def format_sol_value(self, sol_amount, sol_price):
        """
        Formatea el valor de SOL en USDT de manera amigable.
        
        Args:
            sol_amount (float): Cantidad de SOL.
            sol_price (float): Precio de SOL en USDT.
            
        Returns:
            str: Valor formateado.
        """
        usdt_value = sol_amount * sol_price
        return f"{sol_amount} SOL = {self.format_price(usdt_value)} USDT"
    
    def get_market_emoji(self, market_conditions):
        """
        Obtiene emojis basados en las condiciones del mercado.
        
        Args:
            market_conditions (dict): Condiciones del mercado.
            
        Returns:
            dict: Emojis para diferentes métricas.
        """
        emojis = {}
        
        # Volatilidad
        volatility = market_conditions.get('volatility', 0.5)
        if volatility > 0.7:
            emojis['volatility'] = "🌋"  # Alta volatilidad
        elif volatility > 0.4:
            emojis['volatility'] = "📊"  # Volatilidad media
        else:
            emojis['volatility'] = "🏞️"  # Baja volatilidad
        
        # Tendencia
        trend = market_conditions.get('trend_strength', 0)
        if trend > 0.5:
            emojis['trend'] = "🚀"  # Tendencia alcista fuerte
        elif trend > 0.1:
            emojis['trend'] = "📈"  # Tendencia alcista moderada
        elif trend < -0.5:
            emojis['trend'] = "🧸"  # Tendencia bajista fuerte
        elif trend < -0.1:
            emojis['trend'] = "📉"  # Tendencia bajista moderada
        else:
            emojis['trend'] = "↔️"  # Tendencia lateral
        
        # RSI
        rsi = market_conditions.get('rsi', 50)
        if rsi > 70:
            emojis['rsi'] = "🔥"  # Sobrecomprado
        elif rsi < 30:
            emojis['rsi'] = "❄️"  # Sobrevendido
        else:
            emojis['rsi'] = "⚖️"  # Neutral
        
        # Volumen
        volume_change = market_conditions.get('volume_change', 0)
        if volume_change > 0.5:
            emojis['volume'] = "💪"  # Alto volumen
        elif volume_change > 0.1:
            emojis['volume'] = "📊"  # Volumen medio
        else:
            emojis['volume'] = "🔇"  # Bajo volumen
        
        return emojis
    
    def notify_bot_start(self, config):
        """
        Notifica el inicio del bot.
        
        Args:
            config (dict): Configuración del bot.
            
        Returns:
            bool: True si se envió correctamente, False en caso contrario.
        """
        message = f"""🤖 *BOT SOL ADAPTATIVO INICIADO* 🤖

*⚙️ Configuración:*
• 📊 Par: `{config['symbol']}`
• ⏱️ Intervalo: `{config['interval']}`
• 🛑 Stop Loss: `{config['stop_loss']*100}%` (fijo)
• 🎯 Take Profit: Adaptativo
• ⚠️ Riesgo inicial: `{config['risk']*100}%`
• 🧪 Simulación: {'✅' if config['simulation'] else '❌'}
• 🧠 ML activado: {'✅' if config['use_ml'] else '❌'}

💰 *Balance inicial:* `{config['balance']} USDT`

📈 *Precio actual SOL:* `{self.format_price(config.get('current_price', 0))} USDT`

🔍 *Modo aprendizaje:* Activo (operaciones al 50% hasta alcanzar 55% win rate)

⏰ *Iniciado:* {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

_Bot esperando señales de entrada..._
"""
        return self.send_message(message)
    
    def notify_market_update(self, market_conditions, current_price):
        """
        Notifica actualización de condiciones del mercado.
        
        Args:
            market_conditions (dict): Condiciones del mercado.
            current_price (float): Precio actual de SOL.
            
        Returns:
            bool: True si se envió correctamente, False en caso contrario.
        """
        # Obtener emojis
        emojis = self.get_market_emoji(market_conditions)
        
        # Formatear tendencia
        trend = market_conditions.get('trend_strength', 0)
        if trend > 0.5:
            trend_text = "Fuertemente alcista"
        elif trend > 0.1:
            trend_text = "Alcista"
        elif trend < -0.5:
            trend_text = "Fuertemente bajista"
        elif trend < -0.1:
            trend_text = "Bajista"
        else:
            trend_text = "Lateral"
        
        # Formatear RSI
        rsi = market_conditions.get('rsi', 50)
        if rsi > 70:
            rsi_text = "Sobrecomprado"
        elif rsi < 30:
            rsi_text = "Sobrevendido"
        else:
            rsi_text = "Neutral"
        
        message = f"""📊 *ACTUALIZACIÓN DE MERCADO - SOL* 📊

💵 *Precio actual de SOL:* `{self.format_price(current_price)} USDT`
1 SOL = {self.format_price(current_price)} USDT

*Condiciones del mercado:*
• {emojis['trend']} *Tendencia:* {trend_text} (`{trend:.2f}`)
• {emojis['volatility']} *Volatilidad:* `{market_conditions.get('volatility', 0)*100:.1f}%`
• {emojis['rsi']} *RSI:* `{rsi:.1f}` ({rsi_text})
• {emojis['volume']} *Cambio volumen:* `{market_conditions.get('volume_change', 0)*100:.1f}%`

*Predicción próximas velas:*
{self._get_prediction_text(market_conditions)}

⏰ *Actualizado:* {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        return self.send_message(message)
    
    def _get_prediction_text(self, market_conditions):
        """
        Genera texto de predicción basado en condiciones de mercado.
        
        Args:
            market_conditions (dict): Condiciones del mercado.
            
        Returns:
            str: Texto de predicción.
        """
        trend = market_conditions.get('trend_strength', 0)
        volatility = market_conditions.get('volatility', 0.5)
        rsi = market_conditions.get('rsi', 50)
        
        if trend > 0.3 and rsi < 70:
            return "🟢 Probable continuación alcista"
        elif trend < -0.3 and rsi > 30:
            return "🔴 Probable continuación bajista"
        elif trend > 0 and rsi > 70:
            return "🟠 Posible agotamiento alcista"
        elif trend < 0 and rsi < 30:
            return "🟠 Posible rebote técnico"
        elif volatility > 0.7:
            return "🟡 Alta volatilidad, movimiento fuerte probable"
        else:
            return "⚪ Consolidación probable"
    
    def notify_trade_entry(self, trade, current_price):
        """
        Notifica entrada en operación.
        
        Args:
            trade (dict): Información de la operación.
            current_price (float): Precio actual de SOL.
            
        Returns:
            bool: True si se envió correctamente, False en caso contrario.
        """
        # Obtener emojis
        market_conditions = trade.get('market_conditions', {})
        emojis = self.get_market_emoji(market_conditions)
        
        # Calcular valor en USDT
        position_size = trade.get('position_size', 0)
        position_value = position_size * current_price
        
        message = f"""🔵 *NUEVA OPERACIÓN - BOT SOL* 🔵

*Tipo:* {'Compra 📈' if trade.get('type') == 'long' else 'Venta 📉'}

💰 *Detalles:*
• 🏷️ Precio entrada de SOL: `{self.format_price(trade.get('entry_price', 0))} USDT`
• 📦 Tamaño: `{position_size} SOL` = `{self.format_price(position_value)} USDT`
• 🛑 Stop Loss: `{self.format_price(trade.get('stop_loss', 0))} USDT` ({trade.get('stop_loss_pct', 0)*100:.2f}%)
• 🎯 Take Profit: `{self.format_price(trade.get('take_profit', 0))} USDT` ({trade.get('adaptive_tp_pct', 0)*100:.2f}%)

📊 *Condiciones de mercado:*
• {emojis['trend']} *Tendencia:* `{market_conditions.get('trend_strength', 0):.2f}`
• {emojis['volatility']} *Volatilidad:* `{market_conditions.get('volatility', 0)*100:.1f}%`
• {emojis['rsi']} *RSI:* `{market_conditions.get('rsi', 0):.1f}`

⚙️ *Modo aprendizaje:* {'✅ Activo' if trade.get('learning_mode', False) else '❌ Inactivo'}

⏰ *Entrada:* {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        return self.send_message(message)
    
    def notify_trade_exit(self, trade, current_balance, performance_metrics):
        """
        Notifica salida de operación.
        
        Args:
            trade (dict): Información de la operación cerrada.
            current_balance (float): Balance actual.
            performance_metrics (dict): Métricas de rendimiento.
            
        Returns:
            bool: True si se envió correctamente, False en caso contrario.
        """
        # Determinar emoji según resultado
        profit_loss = trade.get('profit_loss', 0)
        if profit_loss > 0:
            result_emoji = "🟢"
            result_text = "GANANCIA"
        else:
            result_emoji = "🔴"
            result_text = "PÉRDIDA"
        
        # Determinar razón de salida
        exit_reasons = {
            'stop_loss': "🛑 Stop Loss",
            'take_profit': "🎯 Take Profit",
            'trailing_stop': "🔄 Trailing Stop",
            'signal': "📊 Señal de salida"
        }
        exit_reason = exit_reasons.get(trade.get('exit_reason', ''), "❓ Desconocida")
        
        # Calcular duración de la operación
        try:
            entry_time = datetime.fromisoformat(trade.get('entry_time', '').replace('Z', '+00:00'))
            exit_time = datetime.fromisoformat(trade.get('exit_time', '').replace('Z', '+00:00'))
            duration = exit_time - entry_time
            hours, remainder = divmod(duration.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            duration_text = f"{int(hours)}h {int(minutes)}m"
        except:
            duration_text = "Desconocida"
        
        # Calcular valor en USDT
        position_size = trade.get('position_size', 0)
        entry_value = position_size * trade.get('entry_price', 0)
        exit_value = position_size * trade.get('exit_price', 0)
        
        message = f"""{result_emoji} *OPERACIÓN CERRADA - {result_text}* {result_emoji}

*Tipo:* {'Compra 📈' if trade.get('type') == 'long' else 'Venta 📉'}

💰 *Detalles:*
• 🏷️ Precio entrada de SOL: `{self.format_price(trade.get('entry_price', 0))} USDT`
• 🏷️ Precio salida de SOL: `{self.format_price(trade.get('exit_price', 0))} USDT`
• 📦 Tamaño: `{position_size} SOL`
• 📊 Entrada: `{self.format_price(entry_value)} USDT`
• 📊 Salida: `{self.format_price(exit_value)} USDT`

💵 *Resultado:*
• 💸 P/L: `{self.format_price(profit_loss)} USDT` ({trade.get('profit_loss_pct', 0):.2f}%)
• ⏱️ Duración: `{duration_text}`
• 🚪 Razón de salida: {exit_reason}

📈 *Rendimiento:*
• 💼 Balance actual: `{self.format_price(current_balance)} USDT`
• 📊 Win rate: `{performance_metrics.get('win_rate', 0):.1f}%`
• 📊 Profit factor: `{performance_metrics.get('profit_factor', 0):.2f}`
• 📊 Total P/L: `{self.format_price(performance_metrics.get('total_profit', 0))} USDT`

⏰ *Salida:* {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        return self.send_message(message)
    
    def notify_parameter_update(self, params, market_conditions):
        """
        Notifica actualización de parámetros.
        
        Args:
            params (dict): Parámetros actualizados.
            market_conditions (dict): Condiciones del mercado.
            
        Returns:
            bool: True si se envió correctamente, False en caso contrario.
        """
        # Obtener emojis
        emojis = self.get_market_emoji(market_conditions)
        
        message = f"""⚙️ *PARÁMETROS ACTUALIZADOS - BOT SOL* ⚙️

*Take Profit adaptativo:*
• 🎯 Nuevo TP: `{params.get('take_profit_pct', 0)*100:.2f}%`
• 🛑 Stop Loss: `{params.get('stop_loss_pct', 0)*100:.2f}%` (fijo)

*Condiciones de mercado:*
• {emojis['trend']} *Tendencia:* `{market_conditions.get('trend_strength', 0):.2f}`
• {emojis['volatility']} *Volatilidad:* `{market_conditions.get('volatility', 0)*100:.1f}%`
• {emojis['rsi']} *RSI:* `{market_conditions.get('rsi', 0):.1f}`

⚠️ *Ajuste basado en:* {self._get_adjustment_reason(market_conditions)}

⏰ *Actualizado:* {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        return self.send_message(message)
    
    def _get_adjustment_reason(self, market_conditions):
        """
        Genera texto de razón de ajuste basado en condiciones de mercado.
        
        Args:
            market_conditions (dict): Condiciones del mercado.
            
        Returns:
            str: Texto de razón de ajuste.
        """
        trend = market_conditions.get('trend_strength', 0)
        volatility = market_conditions.get('volatility', 0.5)
        rsi = market_conditions.get('rsi', 50)
        
        if abs(trend) > 0.5:
            return f"Tendencia {'alcista' if trend > 0 else 'bajista'} fuerte"
        elif volatility > 0.7:
            return "Alta volatilidad del mercado"
        elif rsi > 70:
            return "Mercado sobrecomprado"
        elif rsi < 30:
            return "Mercado sobrevendido"
        else:
            return "Análisis periódico del mercado"

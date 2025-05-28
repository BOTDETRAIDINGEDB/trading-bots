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
import traceback

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
    
    def verify_connection(self):
        """
        Verifica la conexión con la API de Telegram.
        
        Returns:
            bool: True si la conexión es exitosa, False en caso contrario.
        """
        if not self.token or not self.chat_id:
            logger.warning("Notificador de Telegram no está habilitado: faltan credenciales")
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.token}/getMe"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                logger.info("Conexión con Telegram verificada exitosamente")
                return True
            else:
                logger.error(f"Error al verificar conexión con Telegram: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error al verificar conexión con Telegram: {str(e)}")
            return False
    
    # Este método ha sido movido y mejorado más abajo en el archivo
    # Ver la implementación completa de send_trade_notification
    
    def send_error_notification(self, error_message):
        """Envía una notificación de error a Telegram.
        
        Args:
            error_message (str): Mensaje de error a enviar.
            
        Returns:
            bool: True si se envió correctamente, False en caso contrario.
        """
        try:
            message = f"🚨 *ERROR EN EL BOT* 🚨\n\n{error_message}\n\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            return self.send_message(message)
        except Exception as e:
            logger.error(f"Error al enviar notificación de error: {str(e)}")
            # Intentar enviar un mensaje simplificado como último recurso
            try:
                simple_message = f"🚨 ERROR: {error_message[:50]}..."
                return self.send_message(simple_message)
            except:
                logger.error("No se pudo enviar ni siquiera el mensaje de error simplificado")
                return False
    
    def send_status_update(self, status_data):
        """Envía una actualización de estado a Telegram.
        
        Args:
            status_data (dict): Datos del estado actual del bot.
            
        Returns:
            bool: True si se envió correctamente, False en caso contrario.
        """
        try:
            # Construir mensaje de estado
            message = f"📊 *ACTUALIZACIÓN DE ESTADO* 📊\n\n"
            
            # Añadir información básica
            message += f"• 🤖 *Bot:* SOL Trading Bot\n"
            message += f"• ⏰ *Timestamp:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # Añadir información del estado
            if 'balance' in status_data:
                message += f"💰 *Balance:* `{status_data['balance']} USDT`\n"
            
            if 'current_price' in status_data:
                message += f"💲 *Precio actual:* `{status_data['current_price']} USDT`\n"
            
            if 'active_trades' in status_data:
                message += f"🔄 *Operaciones activas:* `{status_data['active_trades']}`\n"
            
            if 'profit_today' in status_data:
                profit = status_data['profit_today']
                emoji = "🟢" if profit >= 0 else "🔴"
                message += f"{emoji} *Beneficio hoy:* `{profit} USDT`\n"
            
            # Añadir información adicional si existe
            for key, value in status_data.items():
                if key not in ['balance', 'current_price', 'active_trades', 'profit_today']:
                    message += f"• *{key}:* `{value}`\n"
            
            return self.send_message(message)
        except Exception as e:
            logger.error(f"Error al enviar actualización de estado: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def send_trade_notification(self, trade_type=None, symbol=None, price=None, size=None, profit_loss=None, trade_data=None):
        """
        Envía una notificación sobre una operación de trading.
        
        Esta función acepta dos formas de llamada:
        1. Con parámetros individuales (trade_type, symbol, price, size, profit_loss)
        2. Con un diccionario trade_data que contiene toda la información
        
        Args:
            trade_type (str, optional): Tipo de operación ('buy' o 'sell').
            symbol (str, optional): Símbolo del par de trading.
            price (float, optional): Precio de la operación.
            size (float, optional): Tamaño de la operación.
            profit_loss (float, optional): Beneficio o pérdida de la operación (solo para ventas).
            trade_data (dict, optional): Diccionario con toda la información de la operación.
            
        Returns:
            bool: True si se envió correctamente, False en caso contrario.
        """
        try:
            # Si se proporciona trade_data, usamos la versión nueva
            if trade_data is not None:
                # Construir mensaje según el tipo de operación
                if trade_data['type'] == 'buy':
                    message = self._format_buy_message(trade_data)
                elif trade_data['type'] == 'sell':
                    message = self._format_sell_message(trade_data)
                else:
                    message = f"⚠️ Operación desconocida: {trade_data}"
            else:
                # Compatibilidad con la versión anterior
                # Determinar emoji según el tipo de operación
                emoji = "🟢" if trade_type.lower() == "buy" else "🔴"
                operation = "Compra" if trade_type.lower() == "buy" else "Venta"
                
                # Construir mensaje básico
                message = f"{emoji} *{operation} de {symbol}* {emoji}\n\n"
                message += f"💰 *Precio:* `{price} USDT`\n"
                message += f"📊 *Cantidad:* `{size} {symbol.replace('USDT', '')}`\n"
                message += f"💵 *Total:* `{price * size:.2f} USDT`\n"
                
                # Añadir información de beneficio/pérdida si es una venta
                if trade_type.lower() == "sell" and profit_loss is not None:
                    pl_emoji = "🟢" if profit_loss >= 0 else "🔴"
                    message += f"\n{pl_emoji} *P/L:* `{profit_loss:.2f} USDT`\n"
                    message += f"{pl_emoji} *P/L %:* `{(profit_loss / (price * size)) * 100:.2f}%`\n"
                
                # Añadir timestamp
                message += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Enviar mensaje
            return self.send_message(message)
        except Exception as e:
            logger.error(f"Error al enviar notificación de operación: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
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
        try:
            # Obtener precio fresco directamente de Binance
            symbol = config['symbol']
            fallback_price = config.get('current_price', 0)
            fresh_price = self._get_fresh_price(symbol, fallback_price)
            
            message = f"""🤖 *BOT SOL ADAPTATIVO INICIADO* 🤖

*⚙️ Configuración:*
• 📊 Par: `{config['symbol']}`
• ⏰️ Intervalo: `{config['interval']}`
• 🛑 Stop Loss: `{config['stop_loss']*100}%` (fijo)
• 🏁 Take Profit: Adaptativo
• ⚠️ Riesgo inicial: `{config['risk']*100}%`
• 🧪 Simulación: {'\u2705' if config['simulation'] else '\u274c'}
• 🧠 ML activado: {'\u2705' if config['use_ml'] else '\u274c'}

💰 *Balance inicial:* `{config['balance']} USDT`

📈 *Precio actual SOL:* `{self.format_price(fresh_price)} USDT`

🔍 *Modo aprendizaje:* Activo (operaciones al 50% hasta alcanzar 55% win rate)

⏰ *Iniciado:* {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

_Bot esperando señales de entrada..._
"""
            return self.send_message(message)
        except Exception as e:
            logger.error(f"Error al enviar notificación de inicio: {str(e)}")
            # Intentar enviar un mensaje simplificado en caso de error
            try:
                simple_message = f"""🤖 *BOT SOL ADAPTATIVO INICIADO* 🤖

⏰ *Iniciado:* {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

_Bot esperando señales de entrada..._
"""
                return self.send_message(simple_message)
            except:
                pass
            return False
    
    def _get_fresh_price(self, symbol, fallback_price):
        """
        Obtiene el precio más actualizado directamente de Binance.
        
        Args:
            symbol (str): Símbolo a consultar (ej. 'SOLUSDT').
            fallback_price (float): Precio de respaldo si no se puede obtener el actual.
            
        Returns:
            float: Precio actualizado o el precio de respaldo si hay error.
        """
        try:
            # Intentar obtener el precio directamente de la API pública de Binance
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if 'price' in data:
                    fresh_price = float(data['price'])
                    logger.info(f"Precio fresco obtenido de Binance API: {fresh_price}")
                    return fresh_price
            
            # Si falla, intentar con otro endpoint
            url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if 'lastPrice' in data:
                    fresh_price = float(data['lastPrice'])
                    logger.info(f"Precio fresco obtenido de Binance 24hr API: {fresh_price}")
                    return fresh_price
                    
            # Si todo falla, usar el precio de respaldo
            logger.warning(f"No se pudo obtener precio fresco, usando precio de respaldo: {fallback_price}")
            return fallback_price
        except Exception as e:
            logger.warning(f"Error al obtener precio fresco de Binance: {str(e)}")
            return fallback_price
    
    def notify_market_update(self, market_conditions, current_price):
        """
        Notifica actualización de condiciones del mercado.
        
        Args:
            market_conditions (dict): Condiciones del mercado.
            current_price (float): Precio actual de SOL.
            
        Returns:
            bool: True si se envió correctamente, False en caso contrario.
        """
        try:
            # Obtener precio fresco directamente de Binance
            fresh_price = self._get_fresh_price('SOLUSDT', current_price)
            
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

💵 *Precio actual de SOL:* `{self.format_price(fresh_price)} USDT`
1 SOL = {self.format_price(fresh_price)} USDT

*Condiciones del mercado:*
• {emojis['trend']} *Tendencia:* {trend_text} (`{trend:.2f}`)
• {emojis['volatility']} *Volatilidad:* `{market_conditions.get('volatility', 0)*100:.1f}%`
• {emojis['rsi']} *RSI:* `{rsi:.1f}` ({rsi_text})
• {emojis['volume']} *Cambio volumen:* `{market_conditions.get('volume_change', 0)*100:.1f}%`

*Predicción próximas velas:*
{self._get_prediction_text(market_conditions)}

⏰ *Actualizado:* {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"""
            
            return self.send_message(message)
        except Exception as e:
            logger.error(f"Error al enviar notificación de mercado: {str(e)}")
            # Intentar enviar un mensaje simplificado en caso de error
            try:
                simple_message = f"""📊 *ACTUALIZACIÓN DE MERCADO - SOL* 📊

💵 *Precio actual de SOL:* `{self.format_price(current_price)} USDT`

⏰ *Actualizado:* {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"""
                return self.send_message(simple_message)
            except:
                pass
            return False
    
    def _get_prediction_text(self, market_conditions):
        """
        Genera texto de predicción basado en condiciones de mercado.
        
        Args:
            market_conditions (dict): Condiciones del mercado.
            
        Returns:
            str: Texto de predicción.
        """
        # Obtener valores con valores predeterminados seguros
        trend = float(market_conditions.get('trend_strength', 0))
        volatility = float(market_conditions.get('volatility', 0.5))
        rsi = float(market_conditions.get('rsi', 50))
        volume_change = float(market_conditions.get('volume_change', 0))
        
        # Registrar los valores para depuración
        logger.debug(f"Predicción con: trend={trend}, volatility={volatility}, rsi={rsi}, volume_change={volume_change}")
        
        # Lógica de predicción mejorada
        if abs(trend) < 0.05:  # Tendencia muy lateral
            if volatility > 0.4:
                return "🟡 Posible ruptura de rango"
            else:
                return "⚪ Consolidación probable"
        
        if trend > 0.2:  # Tendencia alcista clara
            if rsi > 75:
                return "🟠 Sobrecompra, posible corrección"
            elif volume_change > 0.2:
                return "🟢 Continuación alcista con volumen"
            else:
                return "🟢 Probable continuación alcista"
                
        if trend < -0.2:  # Tendencia bajista clara
            if rsi < 25:
                return "🟠 Sobreventa, posible rebote"
            elif volume_change > 0.2:
                return "🔴 Continuación bajista con volumen"
            else:
                return "🔴 Probable continuación bajista"
        
        # Casos mixtos
        if rsi > 70:
            return "🟠 Posible agotamiento alcista"
        elif rsi < 30:
            return "🟠 Posible rebote técnico"
        elif volatility > 0.6:
            return "🟡 Alta volatilidad, movimiento fuerte probable"
        
        # Caso predeterminado
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

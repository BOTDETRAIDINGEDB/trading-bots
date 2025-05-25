#!/usr/bin/env python3
import os
import json
import time
import logging
import requests
from datetime import datetime
from dotenv import load_dotenv
from src.utils.telegram_notifier import TelegramNotifier

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("telegram_monitor_xrp.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def read_json_file(file_path):
    """Lee un archivo JSON y devuelve su contenido"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        logger.error(f"Error al leer {file_path}: {str(e)}")
        return None

def main():
    # Cargar variables de entorno
    load_dotenv()
    
    # Inicializar notificador de Telegram
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not telegram_token or not telegram_chat_id:
        logger.error("Credenciales de Telegram no encontradas. Saliendo.")
        return
    
    telegram = TelegramNotifier(telegram_token, telegram_chat_id)
    logger.info("Notificador de Telegram inicializado para XRP")
    
    # Intentar obtener el precio actual de XRP
    try:
        # Obtener precio de XRP
        response = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=XRPUSDT")
        if response.status_code == 200:
            data = response.json()
            xrp_price = float(data["price"])
            setattr(telegram, 'last_xrp_price', xrp_price)
            logger.info(f"Precio actual de XRP obtenido: {xrp_price} USDT")
    except Exception as e:
        logger.error(f"Error al obtener el precio de XRP: {str(e)}")
    
    # Enviar mensaje de inicio
    telegram.send_message("🤖 *Monitor de Trading XRP Iniciado* 🤖\nEl monitor está ahora en funcionamiento y enviará actualizaciones periódicas para XRP.")
    
    # Variables para seguimiento
    last_state_xrp = None
    last_balance_xrp = None
    last_summary_day = None
    
    # Bucle principal
    while True:
        try:
            # Leer archivos de estado para XRP
            bot_state_xrp = read_json_file("bot_state_xrp_30m.json")
            simulation_state_xrp = read_json_file("simulation_state_xrp_30m.json")
            
            # Verificar si hay cambios en el estado de XRP
            if simulation_state_xrp and (not last_state_xrp or simulation_state_xrp != last_state_xrp):
                # Actualización de balance
                if "balance" in simulation_state_xrp:
                    balance = simulation_state_xrp["balance"]
                    pnl = bot_state_xrp.get("performance_metrics", {}).get("total_pnl", 0) if bot_state_xrp else 0
                    
                    # Mensaje mejorado para XRP
                    message = f"📊 *ACTUALIZACIÓN DE BALANCE - BOT XRP* 📊\n\n"
                    
                    # Mostrar cada activo con su valor en USD
                    total_usd_value = 0
                    for asset, amount in balance.items():
                        if asset == "USDT":
                            message += f"💵 {asset}: `{amount}`\n"
                            total_usd_value += amount
                        else:
                            asset_price = getattr(telegram, f'last_{asset.lower()}_price', None)
                            if asset_price:
                                usd_value = amount * asset_price
                                message += f"🪙 {asset}: `{amount}` ≈ `{usd_value:.2f} USDT`\n"
                                total_usd_value += usd_value
                            else:
                                message += f"🪙 {asset}: `{amount}`\n"
                    
                    # Mostrar valor total del portafolio
                    message += f"\n💰 *Valor Total*: `{total_usd_value:.2f} USDT`\n"
                    
                    if pnl is not None:
                        emoji = "📈" if pnl >= 0 else "📉"
                        pnl_percentage = (pnl / (total_usd_value - pnl)) * 100 if total_usd_value > pnl else 0
                        message += f"{emoji} *PnL*: `{pnl:.2f} USDT ({pnl_percentage:.2f}%)`\n"
                    
                    message += f"\n⏰ *Fecha*: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
                    
                    telegram.send_message(message)
                    last_balance_xrp = balance
                
                # Verificar si hay nuevas operaciones
                if "orders" in simulation_state_xrp:
                    orders = simulation_state_xrp["orders"]
                    if last_state_xrp and "orders" in last_state_xrp:
                        new_orders = [order for order in orders if order not in last_state_xrp["orders"]]
                        
                        # Actualizar el precio para cálculos futuros
                        for order in new_orders:
                            if order.get("symbol", "").endswith("USDT"):
                                asset = order.get("symbol", "")[:-4].lower()
                                price = float(order.get("price", 0))
                                if price > 0:
                                    setattr(telegram, f'last_{asset}_price', price)
                        
                        # Enviar notificaciones de nuevas órdenes
                        for order in new_orders:
                            symbol = order.get("symbol", "")
                            side = order.get("side", "")
                            price = float(order.get("price", 0))
                            quantity = float(order.get("executedQty", 0))
                            total_value = float(order.get("cummulativeQuoteQty", 0))
                            
                            # Mensaje mejorado para operaciones de XRP
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            emoji = "🟢 COMPRA" if side == "BUY" else "🔴 VENTA"
                            message = f"*{emoji}: {symbol} - BOT XRP*\n"
                            message += f"📊 Precio: `{price} USDT`\n"
                            message += f"📈 Cantidad: `{quantity} {symbol[:-4]}`\n"
                            message += f"💰 Valor Total: `{total_value} USDT`\n"
                            message += f"⏰ Fecha: `{timestamp}`"
                            
                            telegram.send_message(message)
                
                last_state_xrp = simulation_state_xrp
            
            # Enviar resumen diario a medianoche
            current_time = datetime.now()
            current_day = current_time.strftime("%Y-%m-%d")
            
            if current_time.hour == 0 and current_day != last_summary_day:
                # Resumen para XRP si hay datos disponibles
                if bot_state_xrp:
                    metrics_xrp = bot_state_xrp.get("performance_metrics", {})
                    balance_xrp = simulation_state_xrp.get("balance", {}) if simulation_state_xrp else {}
                    
                    telegram.send_message("📈 *Resumen Diario - BOT XRP* 📈")
                    telegram.send_daily_summary(
                        metrics_xrp.get("total_trades", 0),
                        metrics_xrp.get("win_rate", 0),
                        metrics_xrp.get("avg_profit", 0),
                        metrics_xrp.get("avg_loss", 0),
                        metrics_xrp.get("profit_factor", 0),
                        metrics_xrp.get("total_pnl", 0),
                        balance_xrp
                    )
                
                last_summary_day = current_day
            
            # Actualizar el precio de XRP cada 5 minutos
            try:
                # Actualizar precio de XRP
                response = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=XRPUSDT")
                if response.status_code == 200:
                    data = response.json()
                    xrp_price = float(data["price"])
                    setattr(telegram, 'last_xrp_price', xrp_price)
                    logger.info(f"Precio de XRP actualizado: {xrp_price} USDT")
            except Exception as e:
                logger.error(f"Error al actualizar el precio de XRP: {str(e)}")
            
            # Esperar antes del próximo ciclo (5 minutos)
            time.sleep(300)
            
        except Exception as e:
            logger.error(f"Error en el bucle principal: {str(e)}")
            time.sleep(60)  # Esperar un minuto antes de reintentar

if __name__ == "__main__":
    main()

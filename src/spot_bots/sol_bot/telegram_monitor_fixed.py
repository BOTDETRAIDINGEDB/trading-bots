#!/usr/bin/env python3
"""Monitor de Telegram para el bot de trading de SOL.

Este script maneja las notificaciones y monitoreo del bot de trading,
proporcionando actualizaciones en tiempo real sobre operaciones,
balance y estado del bot.
"""

import os
import json
import time
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Union
from tenacity import retry, stop_after_attempt, wait_exponential
from requests.exceptions import RequestException
from dotenv import load_dotenv

from src.utils.telegram_notifier import TelegramNotifier

# Constantes
UPDATE_INTERVAL = 300  # 5 minutos
BALANCE_UPDATE_INTERVAL = 1800  # 30 minutos
MAX_RETRIES = 3
BASE_WAIT = 2
MAX_WAIT = 10
API_TIMEOUT = 10

# Configurar logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(
            filename="telegram_monitor_sol.log",
            encoding='utf-8',
            mode='a'
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Asegurar que las librerías no generen demasiados logs
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

@retry(stop=stop_after_attempt(MAX_RETRIES),
       wait=wait_exponential(multiplier=BASE_WAIT, max=MAX_WAIT))
def read_json_file(file_path: str) -> Dict:
    """Lee un archivo JSON.
    
    Args:
        file_path (str): Ruta al archivo JSON
        
    Returns:
        Dict: Contenido del archivo JSON
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        json.JSONDecodeError: Si el archivo no es un JSON válido
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Archivo no encontrado: {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Formato JSON inválido en: {file_path}")
        raise

def get_sol_price() -> Optional[float]:
    """Obtiene el precio actual de SOL desde Binance.
    
    Returns:
        float: Precio actual de SOL en USDT
        None: Si hay un error al obtener el precio
    """
    try:
        url = "https://api.binance.com/api/v3/ticker/price?symbol=SOLUSDT"
        response = requests.get(url, timeout=API_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        price = float(data["price"])
        logger.info(f"Precio actual de SOL: {price} USDT")
        return price
    except (RequestException, KeyError, ValueError) as e:
        logger.error(f"Error al obtener precio de SOL: {str(e)}")
        return None

class TelegramMonitor:
    """Clase principal para el monitoreo de trading a través de Telegram."""
    
    def __init__(self):
        """Inicializa el monitor de Telegram."""
        # Cargar variables de entorno
        load_dotenv()
        
        # Validar credenciales
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if not self.telegram_token or not self.telegram_chat_id:
            raise ValueError("Credenciales de Telegram no encontradas")
            
        # Inicializar componentes
        self.telegram = TelegramNotifier(self.telegram_token, self.telegram_chat_id)
        self.last_update_time = None
        self.last_balance_update = None
        self.last_state = None
        self.last_summary_day = None
        
        # Inicializar registro de órdenes procesadas
        self.processed_orders = set()
        
        # Archivo de estado del bot
        self.bot_state_file = 'bot_state.json'
        self.simulation_state_file = 'simulation_state.json'
        
    def update_sol_price(self) -> None:
        """Actualiza el precio de SOL."""
        price = get_sol_price()
        if price:
            self.telegram.last_sol_price = price
    
    def check_and_send_daily_summary(self) -> None:
        """Envía el resumen diario si corresponde."""
        current_time = datetime.now()
        current_day = current_time.strftime("%Y-%m-%d")
        
        # Enviar resumen diario a las 00:05
        if current_time.hour == 0 and current_time.minute >= 5 and current_day != self.last_summary_day:
            try:
                # Obtener datos para el resumen
                bot_state = self._get_bot_state()
                simulation_state = self._get_simulation_state()
                
                if bot_state and simulation_state:
                    # Preparar datos para el resumen
                    balance = simulation_state.get("balance", {})
                    metrics = bot_state.get("performance_metrics", {})
                    
                    # Enviar resumen
                    self.telegram.send_daily_summary(balance, metrics)
                    self.last_summary_day = current_day
                    logger.info(f"Resumen diario enviado para {current_day}")
                    
            except Exception as e:
                logger.error(f"Error al enviar resumen diario: {str(e)}")
    
    def _get_bot_state(self) -> Optional[Dict]:
        """Obtiene el estado actual del bot."""
        try:
            if os.path.exists(self.bot_state_file):
                return read_json_file(self.bot_state_file)
            return None
        except Exception as e:
            logger.error(f"Error al leer estado del bot: {str(e)}")
            return None
    
    def _get_simulation_state(self) -> Optional[Dict]:
        """Obtiene el estado actual de la simulación."""
        try:
            if os.path.exists(self.simulation_state_file):
                return read_json_file(self.simulation_state_file)
            return None
        except Exception as e:
            logger.error(f"Error al leer estado de simulación: {str(e)}")
            return None
    
    def check_and_send_updates(self) -> None:
        """Verifica y envía actualizaciones de estado y balance."""
        try:
            current_time = datetime.now()
            
            # Verificar si es momento de actualizar
            if self.last_update_time and (current_time - self.last_update_time).seconds < UPDATE_INTERVAL:
                return
                
            self.last_update_time = current_time
            
            # Obtener estados actuales
            simulation_state = self._get_simulation_state()
            bot_state = self._get_bot_state()
            
            if not simulation_state:
                logger.warning("No se pudo obtener estado de simulación")
                return
                
            send_update = False
            
            # Determinar si debemos enviar una actualización
            if not self.last_state or self._state_has_changed(simulation_state, self.last_state):
                send_update = True
                logger.info("Detectado cambio en el estado del bot")
            elif not self.last_balance_update or \
                 (current_time - self.last_balance_update).seconds >= BALANCE_UPDATE_INTERVAL:
                send_update = True
                logger.info("Enviando actualización periódica de balance")
            
            if send_update and "balance" in simulation_state:
                self._send_balance_update(simulation_state, bot_state)
                self.last_state = simulation_state.copy()
                self.last_balance_update = current_time
                
            # Verificar nuevas operaciones
            self._check_new_orders(simulation_state)
            
        except Exception as e:
            logger.error(f"Error al procesar actualizaciones: {str(e)}")
    
    def _state_has_changed(self, new_state: Dict, old_state: Dict) -> bool:
        """Verifica si el estado ha cambiado significativamente."""
        if "balance" not in new_state or "balance" not in old_state:
            return True
            
        # Verificar cambios en el balance
        new_balance = new_state["balance"]
        old_balance = old_state["balance"]
        
        for key in new_balance:
            if key not in old_balance or abs(new_balance[key] - old_balance[key]) > 0.001:
                return True
                
        return False
    
    def _send_balance_update(self, simulation_state: Dict, bot_state: Optional[Dict]) -> None:
        """Envía una actualización del balance actual."""
        try:
            balance = simulation_state.get("balance", {})
            pnl = bot_state.get("performance_metrics", {}).get("total_pnl", 0) if bot_state else 0
            
            # Validar datos antes de enviar
            if not balance:
                logger.warning("Balance vacío, no se enviará actualización")
                return
                
            self.telegram.send_balance_update(balance, pnl)
            logger.info(f"Actualización de balance enviada: {balance}")
        except Exception as e:
            logger.error(f"Error al enviar actualización de balance: {str(e)}")
    
    def _check_new_orders(self, simulation_state: Dict) -> None:
        """Verifica y notifica sobre nuevas órdenes."""
        if not simulation_state or "orders" not in simulation_state:
            return
            
        try:
            # Crear un identificador único para cada orden
            current_order_ids = set()
            
            for order in simulation_state.get("orders", []):
                # Crear un identificador único basado en los datos de la orden
                order_id = self._generate_order_id(order)
                current_order_ids.add(order_id)
                
                # Verificar si es una nueva orden
                if order_id not in self.processed_orders:
                    self._process_new_order(order)
                    self.processed_orders.add(order_id)
            
            # Limitar el tamaño del conjunto de órdenes procesadas
            if len(self.processed_orders) > 1000:
                # Mantener solo las órdenes actuales y las 100 más recientes
                self.processed_orders = current_order_ids.union(
                    set(list(self.processed_orders - current_order_ids)[-100:])
                )
                
        except Exception as e:
            logger.error(f"Error al verificar nuevas órdenes: {str(e)}")
    
    def _generate_order_id(self, order: Dict) -> str:
        """Genera un identificador único para una orden."""
        # Usar una combinación de campos que hagan única la orden
        symbol = order.get("symbol", "")
        side = order.get("side", "")
        price = str(order.get("price", 0))
        quantity = str(order.get("quantity", order.get("executedQty", 0)))
        timestamp = str(order.get("time", order.get("transactTime", "")))
        
        return f"{symbol}_{side}_{price}_{quantity}_{timestamp}"
    
    def _process_new_order(self, order: Dict) -> None:
        """Procesa y notifica sobre una nueva orden."""
        try:
            # Extraer información de la orden con validación
            symbol = order.get("symbol", "")
            if not symbol:
                logger.warning("Orden sin símbolo, no se procesará")
                return
                
            side = order.get("side", "")
            if not side:
                logger.warning(f"Orden de {symbol} sin lado (compra/venta), no se procesará")
                return
            
            # Obtener precio con validación
            try:
                price = float(order.get("price", 0))
                if price <= 0:
                    # Intentar obtener precio de otros campos
                    price = float(order.get("avgPrice", order.get("cummulativeQuoteQty", 0)))
            except (ValueError, TypeError):
                logger.warning(f"Precio inválido en orden de {symbol}, no se procesará")
                return
                
            # Obtener cantidad con validación
            try:
                quantity = float(order.get("quantity", order.get("executedQty", 0)))
                if quantity <= 0:
                    # Intentar calcular la cantidad a partir de otros campos
                    quote_qty = float(order.get("cummulativeQuoteQty", 0))
                    if quote_qty > 0 and price > 0:
                        quantity = quote_qty / price
            except (ValueError, TypeError, ZeroDivisionError):
                logger.warning(f"Cantidad inválida en orden de {symbol}, no se procesará")
                return
                
            # Calcular valor total
            total_value = price * quantity
            
            # Validación final
            if price > 0 and quantity > 0:
                logger.info(f"Enviando notificación de operación: {symbol} {side} {price} {quantity}")
                self.telegram.send_trade_notification(
                    symbol, side, price, quantity, total_value
                )
            else:
                logger.warning(f"Valores inválidos en orden: precio={price}, cantidad={quantity}")
                
        except Exception as e:
            logger.error(f"Error al procesar nueva orden: {str(e)}")
    
    def run(self) -> None:
        """Ejecuta el monitor de Telegram."""
        logger.info("Monitor de Trading SOL Iniciado")
        self.telegram.send_message(
            "🤖 *Monitor de Trading SOL Iniciado* 🤖\n"
            "El monitor está ahora en funcionamiento y enviará actualizaciones periódicas."
        )
        
        # Obtener precio inicial
        self.update_sol_price()
        
        while True:
            try:
                # Verificar actualizaciones
                self.check_and_send_updates()
                
                # Verificar resumen diario
                self.check_and_send_daily_summary()
                
                # Actualizar precio de SOL periódicamente
                self.update_sol_price()
                
                # Esperar antes de la próxima verificación
                time.sleep(60)  # Verificar cada minuto
                
            except KeyboardInterrupt:
                logger.info("Monitor detenido por el usuario")
                self.telegram.send_message("⚠️ Monitor de Trading SOL detenido manualmente")
                break
            except Exception as e:
                logger.error(f"Error en el ciclo principal del monitor: {str(e)}")
                time.sleep(300)  # Esperar 5 minutos antes de reintentar

def main() -> None:
    """Función principal del monitor."""
    try:
        # Cargar variables de entorno
        load_dotenv()
        
        # Verificar credenciales
        telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if not telegram_token or not telegram_chat_id:
            logger.error("Credenciales de Telegram no encontradas en variables de entorno")
            return
            
        # Inicializar y ejecutar monitor
        monitor = TelegramMonitor()
        monitor.run()
        
    except KeyboardInterrupt:
        logger.info("Monitor detenido por el usuario")
    except Exception as e:
        logger.error(f"Error crítico en el monitor: {str(e)}")

if __name__ == "__main__":
    main()

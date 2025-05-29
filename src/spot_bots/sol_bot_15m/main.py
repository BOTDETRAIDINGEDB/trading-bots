# main.py
import os
import time
import logging
import logging.handlers
import argparse
import json
import sys
import signal as sys_signal
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Ajustar el PYTHONPATH para encontrar los módulos correctamente
bot_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(bot_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Importar módulos del bot
from utils.binance_client import BinanceAPI
from utils.enhanced_telegram_notifier import EnhancedTelegramNotifier
from utils.api_client import APIClient
from data.processor import DataProcessor
from strategies.technical_strategy import TechnicalStrategy

# Cargar variables de entorno
load_dotenv()

# Configurar logging
def setup_logging(log_file='sol_bot_15m.log'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Limpiar handlers existentes
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Crear handler para archivo con rotación
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10485760,  # 10MB
        backupCount=5,
        mode='a'
    )
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Crear handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Añadir handlers al logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def parse_arguments():
    parser = argparse.ArgumentParser(description='Bot de trading para SOL con intervalo de 15 minutos')
    parser.add_argument('--interval', type=str, default='15m', help='Intervalo de tiempo para las velas (ej. 15m, 1h, 4h)')
    parser.add_argument('--symbol', type=str, default='SOLUSDT', help='Par de trading')
    parser.add_argument('--simulation', action='store_true', help='Ejecutar en modo simulación')
    parser.add_argument('--balance', type=float, default=1000.0, help='Balance inicial para simulación')
    parser.add_argument('--risk', type=float, default=0.03, help='Riesgo por operación (0.03 = 3%)')
    parser.add_argument('--status-interval', type=int, default=6, help='Intervalo para enviar actualizaciones de estado (horas)')
    parser.add_argument('--use-ml', action='store_true', help='Usar modelo de aprendizaje automático')
    parser.add_argument('--retrain-interval', type=int, default=20, help='Intervalo para reentrenar el modelo (minutos)')
    return parser.parse_args()

def get_interval_seconds(interval):
    """Convierte un intervalo de tiempo en segundos."""
    interval_map = {'m': 60, 'h': 3600, 'd': 86400, 'w': 604800}
    unit = interval[-1]
    value = int(interval[:-1])
    return value * interval_map.get(unit, 3600)

# Capturar señales de terminación para una salida limpia
import sys

def signal_handler(sig, frame):
    logger = logging.getLogger()
    logger.info("Señal de terminación recibida. Cerrando bot de manera ordenada...")
    # No enviamos mensaje de detención aquí, ya que esto podría ser parte de un reinicio
    sys.exit(0)

def run_trading_bot(args, logger):
    """Ejecuta el bot de trading."""
    # Configurar manejadores de señales para salida limpia
    sys_signal.signal(sys_signal.SIGINT, signal_handler)
    sys_signal.signal(sys_signal.SIGTERM, signal_handler)
    
    # Inicializar componentes
    binance_api = BinanceAPI()
    telegram = EnhancedTelegramNotifier()
    api_client = APIClient()
    data_processor = DataProcessor()
    strategy = TechnicalStrategy(
        symbol=args.symbol,
        risk_per_trade=args.risk,
        use_ml=args.use_ml
    )
    
    # Registrar el uso de ML en el log
    if args.use_ml:
        logger.info(f"Modelo de ML activado con reentrenamiento cada {args.retrain_interval} minutos")
    else:
        logger.info("Modelo de ML desactivado")
    
    # Verificar conexión con Telegram
    telegram_connected = telegram.verify_connection()
    logger.info(f"Conexión con Telegram: {'Exitosa' if telegram_connected else 'Fallida'}")
    
    if telegram_connected:
        # FORZAR envío de mensaje de inicio (eliminamos la verificación de tiempo)
        logger.info("Enviando mensaje de inicio a Telegram...")
        
        # Eliminar archivo .last_startup si existe para forzar el envío
        startup_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.last_startup')
        if os.path.exists(startup_file):
            try:
                os.remove(startup_file)
                logger.info("Archivo .last_startup eliminado para forzar notificación")
            except Exception as e:
                logger.warning(f"No se pudo eliminar archivo .last_startup: {str(e)}")
        
        # Enviar mensaje de inicio con reintentos
        success = False
        for attempt in range(3):  # Intentar hasta 3 veces
            try:
                success = telegram.send_message(f"🤖 *Bot de Trading SOL {args.interval} Iniciado* 🤖\nEl bot está {('en modo simulación' if args.simulation else 'en modo real')}.")
                if success:
                    logger.info("Mensaje de inicio enviado correctamente a Telegram")
                    break
                else:
                    logger.warning(f"Intento {attempt+1}/3: No se pudo enviar mensaje de inicio")
                    time.sleep(2)  # Esperar antes de reintentar
            except Exception as e:
                logger.error(f"Intento {attempt+1}/3: Error al enviar mensaje de inicio: {str(e)}")
                time.sleep(2)  # Esperar antes de reintentar
        
        if not success:
            logger.error("No se pudo enviar mensaje de inicio después de múltiples intentos")
        
        # Guardar timestamp del inicio solo si el envío fue exitoso
        if success:
            try:
                with open(startup_file, 'w') as f:
                    f.write(str(time.time()))
                logger.info("Timestamp de inicio guardado correctamente")
            except Exception as e:
                logger.warning(f"No se pudo guardar timestamp de inicio: {str(e)}")
    else:
        logger.error("No se pudo enviar mensaje de inicio: Telegram no está conectado")
        logger.error("Verifica las credenciales de Telegram en el archivo credentials.json")
        logger.error(f"TELEGRAM_BOT_TOKEN: {'Configurado' if os.getenv('TELEGRAM_BOT_TOKEN') else 'No configurado'}")
        logger.error(f"TELEGRAM_CHAT_ID: {'Configurado' if os.getenv('TELEGRAM_CHAT_ID') else 'No configurado'}")

    
    # Verificar conexión con la API y registrar el bot
    api_connected = api_client.verify_connection()
    if api_connected:
        logger.info("Conexión con API establecida correctamente")
        bot_registered = api_client.register_bot(
            bot_name=f"SOL Bot {args.interval}",
            bot_type="spot",
            symbol=args.symbol,
            interval=args.interval
        )
        if bot_registered:
            logger.info("Bot registrado exitosamente en la API")
            # Actualizar estado inicial
            api_client.update_bot_status(
                status="active",
                metrics={
                    "mode": "simulation" if args.simulation else "real",
                    "interval": args.interval,
                    "symbol": args.symbol
                }
            )
    else:
        logger.warning("No se pudo establecer conexión con la API. El bot funcionará sin integración con la API.")
        api_connected = False
    
    # Establecer balance inicial
    if args.simulation:
        strategy.set_balance(args.balance)
        logger.info(f"Modo simulación activado con balance inicial de {args.balance} USDT")
    else:
        # Obtener balance real de la cuenta
        usdt_balance = binance_api.get_account_balance('USDT')
        if usdt_balance is not None:
            strategy.set_balance(usdt_balance)
            logger.info(f"Balance real de USDT: {usdt_balance}")
        else:
            logger.error("No se pudo obtener el balance real. Usando valor por defecto.")
            strategy.set_balance(args.balance)
    
    # Cargar estado previo si existe
    state_file = f"sol_bot_{args.interval.replace('m', 'min')}_state.json"
    if os.path.exists(state_file):
        if strategy.load_state(state_file):
            logger.info(f"Estado previo cargado desde {state_file}")
            
    # Verificar resultados de validación cruzada y ajustar comportamiento si es necesario
    if args.use_ml:
        if strategy.check_cv_results():
            logger.info("Comportamiento ajustado basado en resultados de validación cruzada")
            # Notificar ajuste por Telegram si está conectado
            if telegram_connected:
                telegram.send_message("⚠️ *Ajuste Automático* ⚠️\nEl comportamiento del bot ha sido ajustado basado en resultados de validación cruzada. Se ha reducido el nivel de riesgo para compensar posible sobreajuste del modelo.")
        else:
            logger.info("No se encontraron resultados de validación cruzada o no fue necesario ajustar el comportamiento")
            # Sobrescribir los parámetros de línea de comandos con los valores del archivo de estado
            # Solo si no se especificaron explícitamente en la línea de comandos
            logger.info(f"Usando parámetros del archivo de estado: risk_per_trade={strategy.risk_per_trade}, stop_loss_pct={strategy.stop_loss_pct}, take_profit_pct={strategy.take_profit_pct}, trailing_percent={strategy.trailing_percent}")
        else:
            logger.warning(f"No se pudo cargar el estado previo desde {state_file}")
    
    # Variables para control de tiempo
    interval_seconds = get_interval_seconds(args.interval)
    last_candle_time = None
    last_status_update = datetime.now()
    last_command_check = datetime.now()
    last_ml_train = datetime.now() if args.use_ml else None
    command_check_interval = 60  # Verificar comandos cada 60 segundos
    ml_retrain_interval = args.retrain_interval * 60  # Convertir minutos a segundos
    
    # Estado del bot
    bot_status = "active"
    
    # Bucle principal
    logger.info(f"Bot iniciado para {args.symbol} con intervalo de {args.interval}")
    
    try:
        while True:
            current_time = datetime.now()
            
            # Verificar comandos de la API
            if api_connected and (current_time - last_command_check).total_seconds() >= command_check_interval:
                commands = api_client.get_bot_commands()
                if commands:
                    for cmd in commands:
                        try:
                            cmd_id = cmd.get('id')
                            cmd_type = cmd.get('type')
                            cmd_params = cmd.get('params', {})
                            
                            logger.info(f"Procesando comando: {cmd_type} con parámetros: {cmd_params}")
                            
                            if cmd_type == 'stop':
                                # Detener el bot
                                bot_status = "stopped"
                                api_client.acknowledge_command(cmd_id, 'success', 'Bot detenido correctamente')
                                api_client.update_bot_status('stopped')
                                logger.info("Bot detenido por comando de API")
                                return
                                
                            elif cmd_type == 'pause':
                                # Pausar el bot
                                bot_status = "paused"
                                api_client.acknowledge_command(cmd_id, 'success', 'Bot pausado correctamente')
                                api_client.update_bot_status('paused')
                                logger.info("Bot pausado por comando de API")
                                
                            elif cmd_type == 'resume':
                                # Reanudar el bot
                                if bot_status == "paused":
                                    bot_status = "active"
                                    api_client.acknowledge_command(cmd_id, 'success', 'Bot reanudado correctamente')
                                    api_client.update_bot_status('active')
                                    logger.info("Bot reanudado por comando de API")
                                else:
                                    api_client.acknowledge_command(cmd_id, 'failed', 'El bot no estaba pausado')
                                    
                            elif cmd_type == 'update_risk':
                                # Actualizar riesgo por operación
                                new_risk = float(cmd_params.get('risk', strategy.risk_per_trade))
                                strategy.risk_per_trade = new_risk
                                api_client.acknowledge_command(cmd_id, 'success', f'Riesgo actualizado a {new_risk}')
                                logger.info(f"Riesgo por operación actualizado a {new_risk}")
                                
                            else:
                                # Comando desconocido
                                api_client.acknowledge_command(cmd_id, 'failed', f'Comando desconocido: {cmd_type}')
                                logger.warning(f"Comando desconocido recibido: {cmd_type}")
                                
                        except Exception as e:
                            logger.error(f"Error al procesar comando: {str(e)}")
                            if cmd_id:
                                api_client.acknowledge_command(cmd_id, 'failed', f'Error: {str(e)}')
                
                last_command_check = current_time
            
            # Si el bot está pausado, esperar y continuar
            if bot_status == "paused":
                time.sleep(60)
                continue
            
            # Obtener precio actual
            current_price = binance_api.get_current_price(args.symbol)
            if current_price:
                logger.info(f"Precio de {args.symbol} actualizado: {current_price} USDT")
                
                # Enviar notificación periódica a Telegram cada 15 minutos (900 segundos)
                if telegram_connected:
                    # Inicializar last_market_update si no existe
                    if not 'last_market_update' in locals():
                        # Establecer tiempo inicial para evitar notificación inmediata después del inicio
                        last_market_update = current_time - timedelta(minutes=10)  # Enviar primera notificación después de 5 minutos
                        
                    # Verificar si ha pasado suficiente tiempo para enviar una nueva notificación
                    if (current_time - last_market_update).total_seconds() >= 900:  # 15 minutos
                        # Asegurarse de que tenemos datos procesados antes de enviar notificaciones
                        if not hasattr(strategy, 'df') or strategy.df is None or strategy.df.empty:
                            # Obtener datos históricos para procesar
                            logger.info("Obteniendo datos históricos para calcular indicadores de mercado...")
                            end_time = int(current_time.timestamp() * 1000)
                            start_time = int((current_time - timedelta(days=5)).timestamp() * 1000)
                            
                            klines = binance_api.get_historical_klines(
                                symbol=args.symbol,
                                interval=args.interval,
                                start_time=start_time,
                                end_time=end_time,
                                limit=500
                            )
                            
                            if klines:
                                # Procesar datos
                                df = data_processor.klines_to_dataframe(klines)
                                df = data_processor.calculate_indicators(df)
                                df = data_processor.generate_signals(df)
                                
                                # Guardar el DataFrame en la estrategia
                                strategy.df = df
                                logger.info(f"Datos procesados para indicadores de mercado: {len(df)} filas")
                        
                        # Obtener precio actualizado justo antes de enviar la notificación
                        fresh_price = binance_api.get_current_price(args.symbol)
                        if not fresh_price:
                            fresh_price = current_price  # Usar el precio actual si no se puede obtener uno nuevo
                        
                        # Crear un diccionario de condiciones de mercado para la notificación
                        # Intentar obtener valores reales de los indicadores si están disponibles
                        market_conditions = {}
                        
                        # Obtener tendencia del mercado
                        if hasattr(strategy, 'get_trend_strength') and callable(getattr(strategy, 'get_trend_strength')):
                            market_conditions['trend_strength'] = strategy.get_trend_strength()
                        else:
                            # Calcular tendencia básica basada en las últimas velas si hay datos disponibles
                            if 'df' in locals() and not df.empty and len(df) > 20:
                                closes = df['close'].tail(20).values
                                if len(closes) > 1:
                                    trend = (closes[-1] - closes[0]) / closes[0]
                                    market_conditions['trend_strength'] = trend
                                else:
                                    market_conditions['trend_strength'] = 0
                            else:
                                market_conditions['trend_strength'] = 0
                        
                        # Obtener volatilidad del mercado
                        if hasattr(strategy, 'get_volatility') and callable(getattr(strategy, 'get_volatility')):
                            market_conditions['volatility'] = strategy.get_volatility()
                        else:
                            # Calcular volatilidad básica si hay datos disponibles
                            if 'df' in locals() and not df.empty and len(df) > 20:
                                closes = df['close'].tail(20).values
                                if len(closes) > 1:
                                    volatility = df['high'].tail(20).max() / df['low'].tail(20).min() - 1
                                    market_conditions['volatility'] = min(1.0, max(0.0, volatility))
                                else:
                                    market_conditions['volatility'] = 0.5
                            else:
                                market_conditions['volatility'] = 0.5
                        
                        # Obtener RSI
                        if 'df' in locals() and not df.empty and 'rsi' in df.columns:
                            market_conditions['rsi'] = float(df['rsi'].iloc[-1])
                        else:
                            market_conditions['rsi'] = 50.0
                        
                        # Obtener cambio de volumen
                        if 'df' in locals() and not df.empty and 'volume' in df.columns and len(df) > 20:
                            avg_vol = df['volume'].tail(20).mean()
                            last_vol = df['volume'].iloc[-1]
                            if avg_vol > 0:
                                vol_change = (last_vol - avg_vol) / avg_vol
                                market_conditions['volume_change'] = max(-1.0, min(1.0, vol_change))
                            else:
                                market_conditions['volume_change'] = 0.0
                        else:
                            market_conditions['volume_change'] = 0.0
                        
                        # Enviar la notificación con datos actualizados
                        telegram.notify_market_update(market_conditions, fresh_price)
                        last_market_update = current_time
                        logger.info(f"Enviada notificación de actualización de mercado a Telegram con precio {fresh_price} USDT")
                
                # Actualizar precio en la API
                if api_connected:
                    api_client.update_bot_status(
                        status=bot_status,
                        metrics={
                            "current_price": current_price,
                            "updated_at": current_time.isoformat()
                        }
                    )
            else:
                logger.error(f"No se pudo obtener el precio actual de {args.symbol}")
                time.sleep(60)  # Esperar un minuto antes de reintentar
                continue
            
            # Determinar si es momento de procesar una nueva vela
            if last_candle_time is None or (current_time - last_candle_time).total_seconds() >= interval_seconds:
                logger.info(f"Procesando nueva vela para {args.symbol}")
                
                # Obtener datos históricos
                end_time = int(current_time.timestamp() * 1000)
                start_time = int((current_time - timedelta(days=3)).timestamp() * 1000)
                
                klines = binance_api.get_historical_klines(
                    symbol=args.symbol,
                    interval=args.interval,
                    start_time=start_time,
                    end_time=end_time,
                    limit=500
                )
                
                if klines:
                    # Procesar datos
                    df = data_processor.klines_to_dataframe(klines)
                    df = data_processor.calculate_indicators(df)
                    df = data_processor.generate_signals(df)
                    
                    # Asignar DataFrame a la estrategia para acceso a indicadores
                    strategy.df = df.copy()
                    
                    # Procesar con ML si está habilitado
                    ml_prediction = None
                    retrain_metrics = None
                    if args.use_ml:
                        ml_prediction, retrain_metrics = strategy.process_with_ml(df)
                        
                        # Si se reentrenó el modelo, actualizar métricas en la API
                        if api_connected and retrain_metrics:
                            api_client.update_bot_status(
                                status=bot_status,
                                metrics={
                                    'ml_metrics': retrain_metrics,
                                    'ml_retrained_at': datetime.now().isoformat()
                                }
                            )
                    
                    # Obtener la última señal
                    if not df.empty:
                        last_row = df.iloc[-1]
                        trade_signal = last_row['signal']
                        
                        # Verificar si ya estamos en una posición
                        if strategy.position != 0:
                            # Verificar si debemos salir de la posición
                            if strategy.should_exit_trade(current_price, current_time):
                                trade = strategy.exit_trade(current_price, current_time)
                                if trade:
                                    # Notificar por Telegram
                                    if telegram_connected:
                                        telegram.send_trade_notification(
                                            'exit',
                                            args.symbol,
                                            current_price,
                                            trade['position_size'],
                                            trade['profit_loss']
                                        )
                                    
                                    # Reportar a la API
                                    if api_connected:
                                        trade_data = {
                                            'type': 'exit',
                                            'symbol': args.symbol,
                                            'price': current_price,
                                            'quantity': trade['position_size'],
                                            'profit_loss': trade['profit_loss'],
                                            'profit_loss_pct': trade['profit_loss_pct'],
                                            'timestamp': current_time.isoformat(),
                                            'trade_id': trade['id']
                                        }
                                        api_client.report_trade(trade_data)
                                        
                                        # Actualizar métricas en la API
                                        metrics_update = {
                                            'current_balance': strategy.current_balance,
                                            'performance_metrics': strategy.get_performance_metrics()
                                        }
                                        
                                        # Incluir información de ML si está disponible
                                        if args.use_ml and ml_prediction is not None:
                                            metrics_update['ml_prediction'] = int(ml_prediction)
                                            
                                        api_client.update_bot_status(
                                            status=bot_status,
                                            metrics=metrics_update
                                        )
                        else:
                            # Verificar si debemos entrar en una posición
                            available_balance = strategy.current_balance if args.simulation else binance_api.get_account_balance('USDT') or 0
                            
                            if strategy.should_enter_trade(trade_signal, current_price, available_balance, ml_prediction):
                                trade = strategy.enter_trade(current_price, current_time)
                                if trade:
                                    # Notificar por Telegram
                                    if telegram_connected:
                                        telegram.send_trade_notification(
                                            'entry',
                                            args.symbol,
                                            current_price,
                                            trade['position_size']
                                        )
                                    
                                    # Reportar a la API
                                    if api_connected:
                                        trade_data = {
                                            'type': 'entry',
                                            'symbol': args.symbol,
                                            'price': current_price,
                                            'quantity': trade['position_size'],
                                            'stop_loss': trade['stop_loss'],
                                            'take_profit': trade['take_profit'],
                                            'timestamp': current_time.isoformat(),
                                            'trade_id': trade['id']
                                        }
                                        api_client.report_trade(trade_data)
                    
                    # Guardar estado
                    strategy.save_state(state_file)
                    
                    last_candle_time = current_time
                else:
                    logger.warning(f"No se pudieron obtener datos históricos para {args.symbol}")
            
            # Enviar actualización de estado periódicamente
            status_interval_seconds = args.status_interval * 3600  # Convertir horas a segundos
            if (current_time - last_status_update).total_seconds() >= status_interval_seconds:
                metrics = strategy.get_performance_metrics()
                
                # Enviar actualización por Telegram
                if telegram_connected:
                    telegram.send_status_update(
                        strategy.current_balance,
                        metrics,
                        args.symbol
                    )
                
                # Enviar actualización a la API
                if api_connected:
                    metrics_update = {
                        'current_balance': strategy.current_balance,
                        'performance_metrics': metrics,
                        'last_price': current_price,
                        'updated_at': current_time.isoformat(),
                        'mode': 'simulation' if args.simulation else 'real',
                        'uptime_hours': (current_time - datetime.now()).total_seconds() / 3600
                    }
                    
                    # Incluir información de ML si está habilitado
                    if args.use_ml:
                        metrics_update['ml_enabled'] = True
                        metrics_update['ml_retrain_interval'] = args.retrain_interval
                        
                        if strategy.ml_model and strategy.ml_model.last_trained:
                            metrics_update['ml_last_trained'] = strategy.ml_model.last_trained.isoformat()
                    
                    api_client.update_bot_status(
                        status=bot_status,
                        metrics=metrics_update
                    )
                
                logger.info(f"Actualización de estado: Balance={strategy.current_balance:.2f}, "
                           f"Operaciones={metrics['total_trades']}, Win Rate={metrics['win_rate']:.2f}%")
                
                last_status_update = current_time
            
            # Calcular tiempo hasta la próxima vela
            if last_candle_time:
                next_candle_time = last_candle_time + timedelta(seconds=interval_seconds)
                time_to_next_candle = (next_candle_time - datetime.now()).total_seconds()
                
                # Si falta más de un minuto para la próxima vela, esperar
                if time_to_next_candle > 60:
                    sleep_time = min(time_to_next_candle, 300)  # Máximo 5 minutos
                    logger.debug(f"Esperando {sleep_time:.0f} segundos hasta la próxima actualización")
                    time.sleep(sleep_time)
                else:
                    # Esperar un tiempo corto para no sobrecargar la API
                    time.sleep(10)
            else:
                # Si aún no hemos procesado ninguna vela, esperar un tiempo corto
                time.sleep(10)
    
    except KeyboardInterrupt:
        logger.info("Bot detenido manualmente")
        if api_connected:
            api_client.update_bot_status('stopped', {'reason': 'manual_stop'})
    except Exception as e:
        logger.exception(f"Error inesperado: {str(e)}")
        if telegram_connected:
            telegram.send_error_notification(f"Bot detenido por error: {str(e)}")
        if api_connected:
            api_client.update_bot_status('error', {
                'error': str(e),
                'traceback': logging.traceback.format_exc(),
                'timestamp': datetime.now().isoformat()
            })
    finally:
        # Guardar estado final
        strategy.save_state(state_file)
        logger.info(f"Estado final guardado en {state_file}")
        
        # Notificar detención del bot
        if telegram_connected:
            telegram.send_message("⚠️ *Bot de Trading Detenido* ⚠️\nEl bot ha sido detenido.")
        
        # Asegurarse de que la API sepa que el bot está detenido
        if api_connected and bot_status != 'error':
            api_client.update_bot_status('stopped', {
                'final_balance': strategy.current_balance,
                'performance_metrics': strategy.get_performance_metrics(),
                'stopped_at': datetime.now().isoformat()
            })

def main():
    args = parse_arguments()
    logger = setup_logging(f"sol_bot_{args.interval.replace('m', 'min')}.log")
    
    logger.info(f"Iniciando bot de trading SOL con intervalo {args.interval}")
    logger.info(f"Modo: {'Simulación' if args.simulation else 'Real'}")
    logger.info(f"ML: {'Activado' if args.use_ml else 'Desactivado'}")
    
    if args.use_ml:
        logger.info(f"Intervalo de reentrenamiento: {args.retrain_interval} minutos")
    
    run_trading_bot(args, logger)

if __name__ == "__main__":
    main()

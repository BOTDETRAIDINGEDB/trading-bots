#!/usr/bin/env python3
# Script para verificar la configuración de Telegram y probar el envío de mensajes

import os
import sys
import requests
import logging
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verify_telegram_config():
    """Verifica la configuración de Telegram y prueba el envío de mensajes."""
    # Cargar variables de entorno
    load_dotenv()
    
    # Obtener token y chat_id
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    # Verificar que existan las variables
    if not token:
        logger.error("❌ TELEGRAM_BOT_TOKEN no está configurado en el archivo .env")
        return False
    
    if not chat_id:
        logger.error("❌ TELEGRAM_CHAT_ID no está configurado en el archivo .env")
        return False
    
    logger.info(f"✅ Variables de entorno encontradas:")
    logger.info(f"   - TELEGRAM_BOT_TOKEN: {token[:5]}...{token[-5:]} (longitud: {len(token)})")
    logger.info(f"   - TELEGRAM_CHAT_ID: {chat_id}")
    
    # Verificar conexión con la API de Telegram
    base_url = f"https://api.telegram.org/bot{token}"
    
    try:
        logger.info("Verificando conexión con la API de Telegram...")
        response = requests.get(f"{base_url}/getMe", timeout=10)
        
        if response.status_code == 200:
            bot_info = response.json()['result']
            logger.info(f"✅ Conexión exitosa con el bot: @{bot_info['username']} ({bot_info['first_name']})")
        else:
            logger.error(f"❌ Error al conectar con Telegram: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ Excepción al conectar con Telegram: {str(e)}")
        return False
    
    # Intentar enviar un mensaje de prueba
    try:
        logger.info(f"Enviando mensaje de prueba al chat {chat_id}...")
        message = "🧪 Este es un mensaje de prueba del bot de trading SOL. Si puedes ver este mensaje, la configuración de Telegram está funcionando correctamente."
        
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }
        
        response = requests.post(f"{base_url}/sendMessage", json=payload, timeout=10)
        
        if response.status_code == 200:
            logger.info("✅ Mensaje enviado exitosamente")
            return True
        else:
            logger.error(f"❌ Error al enviar mensaje: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ Excepción al enviar mensaje: {str(e)}")
        return False

def main():
    """Función principal."""
    logger.info("=== Verificación de configuración de Telegram ===")
    
    if verify_telegram_config():
        logger.info("✅ Verificación completada exitosamente. Telegram está correctamente configurado.")
    else:
        logger.error("❌ Verificación fallida. Por favor revisa la configuración de Telegram.")
        logger.info("\nPasos para solucionar problemas:")
        logger.info("1. Verifica que el archivo .env contenga las variables TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID")
        logger.info("2. Asegúrate de que el token del bot sea válido")
        logger.info("3. Confirma que el chat_id sea correcto")
        logger.info("4. Verifica que el bot tenga permisos para enviar mensajes al chat")
        logger.info("5. Comprueba la conexión a internet")

if __name__ == "__main__":
    main()

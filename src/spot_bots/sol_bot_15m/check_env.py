#!/usr/bin/env python3
"""
Script simple para verificar las variables de entorno disponibles
"""

import os
import sys

def main():
    print("Verificando variables de entorno relacionadas con Binance y Telegram:")
    
    # Variables de Binance
    binance_api_key = os.getenv('BINANCE_API_KEY')
    binance_api_secret = os.getenv('BINANCE_API_SECRET')
    
    # Variables de Telegram
    telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    print(f"BINANCE_API_KEY: {'Configurado' if binance_api_key else 'No configurado'}")
    if binance_api_key:
        print(f"  Primeros 4 caracteres: {binance_api_key[:4]}...")
        
    print(f"BINANCE_API_SECRET: {'Configurado' if binance_api_secret else 'No configurado'}")
    if binance_api_secret:
        print(f"  Primeros 4 caracteres: {binance_api_secret[:4]}...")
    
    print(f"TELEGRAM_BOT_TOKEN: {'Configurado' if telegram_bot_token else 'No configurado'}")
    print(f"TELEGRAM_CHAT_ID: {'Configurado' if telegram_chat_id else 'No configurado'}")
    
    # Verificar si hay un archivo .env en el directorio actual
    if os.path.exists('.env'):
        print("\nSe encontró un archivo .env en el directorio actual.")
        print("Contenido del archivo .env (sin mostrar valores completos):")
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    if value:
                        print(f"  {key}: Configurado")
                        if len(value) > 4:
                            print(f"    Primeros 4 caracteres: {value[:4]}...")
                    else:
                        print(f"  {key}: No configurado")
    else:
        print("\nNo se encontró un archivo .env en el directorio actual.")
    
    # Verificar si hay un archivo credentials.json en el directorio actual o en el directorio padre
    credentials_files = [
        'credentials.json',
        '../credentials.json',
        '../../credentials.json',
        '../../../credentials.json',
        '../../../../credentials.json',
        os.path.expanduser('~/.credentials.json'),
        os.path.expanduser('~/credentials.json'),
        os.path.expanduser('~/new-trading-bots-api/credentials.json')
    ]
    
    found_credentials = False
    for cred_file in credentials_files:
        if os.path.exists(cred_file):
            found_credentials = True
            print(f"\nSe encontró un archivo de credenciales en: {cred_file}")
            print(f"Tamaño del archivo: {os.path.getsize(cred_file)} bytes")
    
    if not found_credentials:
        print("\nNo se encontró ningún archivo credentials.json en las ubicaciones probadas.")
    
    # Verificar cómo se inicia el bot (script de inicio)
    bot_scripts = ['start_bot.sh', 'cleanup_bot_sessions.sh', 'start_cloud_simulation.sh']
    for script in bot_scripts:
        if os.path.exists(script):
            print(f"\nSe encontró el script {script}. Verificando cómo se configuran las variables de entorno:")
            with open(script, 'r') as f:
                content = f.read()
                env_lines = [line for line in content.split('\n') if 'BINANCE_API' in line or 'TELEGRAM_' in line or 'export ' in line or 'source ' in line or '.env' in line]
                if env_lines:
                    print("Líneas relacionadas con variables de entorno:")
                    for line in env_lines:
                        print(f"  {line}")
                else:
                    print("No se encontraron referencias a variables de entorno en el script.")

if __name__ == "__main__":
    main()

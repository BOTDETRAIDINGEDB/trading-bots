#!/usr/bin/env python3
# Script para corregir todas las referencias de 20m a 15m en todos los archivos

import os
import sys
import re
import glob
import json

def fix_file(file_path, is_python=False, is_json=False, is_env=False):
    """Corrige las referencias en un archivo"""
    if not os.path.exists(file_path):
        print(f"Error: El archivo {file_path} no existe.")
        return False
    
    try:
        with open(file_path, 'r') as f:
            if is_json:
                try:
                    content = json.load(f)
                    # Convertir a string para buscar y reemplazar
                    content_str = json.dumps(content, indent=4)
                    modified = False
                    
                    # Reemplazar referencias
                    if "20m" in content_str:
                        content_str = content_str.replace("20m", "15m")
                        modified = True
                    if "20min" in content_str:
                        content_str = content_str.replace("20min", "15min")
                        modified = True
                    if "sol_bot_20m" in content_str:
                        content_str = content_str.replace("sol_bot_20m", "sol_bot_15m")
                        modified = True
                    
                    if modified:
                        # Convertir de nuevo a objeto JSON
                        content = json.loads(content_str)
                        with open(file_path, 'w') as f:
                            json.dump(content, f, indent=4)
                        print(f"Archivo JSON {file_path} actualizado.")
                        return True
                    else:
                        print(f"No se encontraron referencias a corregir en {file_path}")
                        return False
                except json.JSONDecodeError:
                    print(f"Error: El archivo {file_path} no es un JSON válido.")
                    return False
            else:
                content = f.read()
    except Exception as e:
        print(f"Error al leer el archivo {file_path}: {e}")
        return False
    
    modified = False
    
    # Reemplazar referencias
    if "20m" in content:
        content = content.replace("20m", "15m")
        modified = True
    if "20min" in content:
        content = content.replace("20min", "15min")
        modified = True
    if "sol_bot_20m" in content:
        content = content.replace("sol_bot_20m", "sol_bot_15m")
        modified = True
    
    # Corrección específica para el error de pandas en processor.py
    if is_python and "df.loc[:200, 'signal'] = 0" in content:
        content = content.replace("df.loc[:200, 'signal'] = 0", "df.iloc[:200, df.columns.get_loc('signal')] = 0")
        print("Se corrigió el error de pandas en el archivo.")
        modified = True
    
    # Corrección para el intervalo de reentrenamiento
    if is_python and "retrain_interval = 20" in content:
        content = content.replace("retrain_interval = 20", "retrain_interval = 15")
        modified = True
    
    # Corrección para archivos .env
    if is_env:
        if "BOT_ID=sol_bot_20m" in content:
            content = content.replace("BOT_ID=sol_bot_20m", "BOT_ID=sol_bot_15m")
            modified = True
    
    if modified:
        try:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"Archivo {file_path} actualizado.")
            return True
        except Exception as e:
            print(f"Error al escribir el archivo {file_path}: {e}")
            return False
    else:
        print(f"No se encontraron referencias a corregir en {file_path}")
        return False

def fix_all_references(bot_dir):
    """Corrige todas las referencias en todos los archivos del bot"""
    if not os.path.exists(bot_dir):
        print(f"Error: El directorio {bot_dir} no existe.")
        return False
    
    print(f"Buscando archivos en {bot_dir}...")
    
    # Corregir archivos Python
    python_files = glob.glob(f"{bot_dir}/**/*.py", recursive=True)
    for file in python_files:
        print(f"Procesando archivo Python: {file}")
        fix_file(file, is_python=True)
    
    # Corregir archivos de shell script
    shell_files = glob.glob(f"{bot_dir}/**/*.sh", recursive=True)
    for file in shell_files:
        print(f"Procesando archivo shell: {file}")
        fix_file(file)
    
    # Corregir archivos JSON
    json_files = glob.glob(f"{bot_dir}/**/*.json", recursive=True)
    for file in json_files:
        print(f"Procesando archivo JSON: {file}")
        fix_file(file, is_json=True)
    
    # Corregir archivos .env
    env_files = glob.glob(f"{bot_dir}/**/.env", recursive=True)
    for file in env_files:
        print(f"Procesando archivo .env: {file}")
        fix_file(file, is_env=True)
    
    print("Proceso completado.")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        bot_dir = sys.argv[1]
    else:
        bot_dir = "/home/edisonbautistaruiz2025/new-trading-bots"
    
    api_dir = "/home/edisonbautistaruiz2025/trading-bots-api"
    
    print("=== Corrigiendo referencias en el directorio del bot ===")
    fix_all_references(bot_dir)
    
    print("\n=== Corrigiendo referencias en el directorio de la API ===")
    fix_all_references(api_dir)
    
    print("\n=== Corrección completada ===")
    print("Para aplicar los cambios, reinicia el bot y la API:")
    print("1. Detén el bot: screen -S sol_bot_15m -X quit")
    print("2. Detén la API: screen -S trading_api -X quit")
    print("3. Inicia el bot: ~/new-trading-bots/start_sol_bot_15m.sh")
    print("4. Inicia la API: cd ~/trading-bots-api && screen -dmS trading_api python app.py")

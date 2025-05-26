#!/usr/bin/env python3
# Script para corregir el error en el procesador de datos

import os
import sys

def fix_processor_file(file_path):
    """Corrige el error en el archivo processor.py"""
    if not os.path.exists(file_path):
        print(f"Error: El archivo {file_path} no existe.")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Reemplazar la línea problemática
    if "df.loc[:200, 'signal'] = 0" in content:
        content = content.replace("df.loc[:200, 'signal'] = 0", "df.iloc[:200, df.columns.get_loc('signal')] = 0")
        print("Se encontró y corrigió la línea problemática.")
    else:
        print("No se encontró la línea problemática en el archivo.")
        return False
    
    # Guardar el archivo corregido
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Archivo {file_path} actualizado correctamente.")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        processor_file = sys.argv[1]
    else:
        processor_file = "/home/edisonbautistaruiz2025/new-trading-bots/src/spot_bots/sol_bot_15m/src/data/processor.py"
    
    if fix_processor_file(processor_file):
        print("Corrección completada. Reinicia el bot para aplicar los cambios.")
    else:
        print("No se pudo completar la corrección. Verifica la ruta del archivo.")

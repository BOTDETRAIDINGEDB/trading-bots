#!/usr/bin/env python3
"""
Script para eliminar todos los archivos de respaldo en el proyecto.

Este script busca y elimina archivos con extensiones o patrones típicos de respaldo como:
.bak, .backup, .old, .tmp, .temp, ~, etc.

También elimina directorios de respaldo que contengan 'backup', 'old', 'temp', etc. en su nombre.

Uso:
    python delete_backup_files.py [--dry-run] [--dir PROJECT_DIR]

Opciones:
    --dry-run: Muestra qué archivos se eliminarían sin eliminarlos realmente
    --dir: Directorio del proyecto a limpiar (por defecto, el directorio actual)
"""

import os
import sys
import re
import shutil
import argparse
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DELETE_BACKUPS")

# Patrones para identificar archivos de respaldo
BACKUP_FILE_PATTERNS = [
    r'.*\.bak$',
    r'.*\.backup$',
    r'.*\.old$',
    r'.*\.tmp$',
    r'.*\.temp$',
    r'.*~$',
    r'.*\.swp$',
    r'.*\.swo$',
    r'.*\.pyc$',
    r'.*\.pyo$',
    r'.*\.orig$',
    r'.*\.rej$',
    r'.*\.#.*$',
    r'.*%.*$',
    r'.*\.save$',
    r'.*\.copy$',
]

# Patrones para identificar directorios de respaldo
BACKUP_DIR_PATTERNS = [
    r'.*backup.*',
    r'.*\.bak.*',
    r'.*\.old.*',
    r'.*\.tmp.*',
    r'.*\.temp.*',
    r'.*~.*',
    r'.*\.save.*',
    r'.*\.copy.*',
    r'.*_archive.*',
    r'.*_respaldo.*',
    r'.*_copia.*',
]

def is_backup_file(file_path):
    """Determina si un archivo es un archivo de respaldo."""
    file_name = os.path.basename(file_path)
    return any(re.match(pattern, file_name) for pattern in BACKUP_FILE_PATTERNS)

def is_backup_dir(dir_path):
    """Determina si un directorio es un directorio de respaldo."""
    dir_name = os.path.basename(dir_path)
    return any(re.match(pattern, dir_name) for pattern in BACKUP_DIR_PATTERNS)

def find_backup_files(project_dir):
    """Encuentra todos los archivos de respaldo en el proyecto."""
    backup_files = []
    
    for root, dirs, files in os.walk(project_dir):
        # Verificar si el directorio actual es un directorio de respaldo
        if is_backup_dir(root):
            backup_files.append(root)
            # Saltar este directorio ya que lo eliminaremos completo
            dirs[:] = []
            continue
        
        # Verificar archivos individuales
        for file in files:
            file_path = os.path.join(root, file)
            if is_backup_file(file_path):
                backup_files.append(file_path)
    
    return backup_files

def delete_backup_files(backup_files, dry_run=False):
    """Elimina los archivos y directorios de respaldo."""
    deleted_count = 0
    error_count = 0
    
    for path in backup_files:
        try:
            if os.path.isdir(path):
                if dry_run:
                    logger.info(f"[DRY RUN] Se eliminaría directorio: {path}")
                else:
                    shutil.rmtree(path)
                    logger.info(f"Directorio eliminado: {path}")
            else:
                if dry_run:
                    logger.info(f"[DRY RUN] Se eliminaría archivo: {path}")
                else:
                    os.remove(path)
                    logger.info(f"Archivo eliminado: {path}")
            
            deleted_count += 1
        except Exception as e:
            logger.error(f"Error al eliminar {path}: {str(e)}")
            error_count += 1
    
    return deleted_count, error_count

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Elimina todos los archivos de respaldo en el proyecto")
    parser.add_argument("--dry-run", action="store_true", help="Muestra qué archivos se eliminarían sin eliminarlos realmente")
    parser.add_argument("--dir", help="Directorio del proyecto a limpiar (por defecto, el directorio actual)")
    
    args = parser.parse_args()
    
    # Determinar el directorio del proyecto
    project_dir = args.dir if args.dir else os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    
    logger.info(f"Buscando archivos de respaldo en: {project_dir}")
    
    # Encontrar archivos de respaldo
    backup_files = find_backup_files(project_dir)
    
    if not backup_files:
        logger.info("No se encontraron archivos o directorios de respaldo")
        return 0
    
    logger.info(f"Se encontraron {len(backup_files)} archivos/directorios de respaldo")
    
    # Mostrar los archivos encontrados
    for i, path in enumerate(backup_files, 1):
        if os.path.isdir(path):
            logger.info(f"{i}. Directorio: {path}")
        else:
            logger.info(f"{i}. Archivo: {path}")
    
    # Eliminar archivos de respaldo
    deleted_count, error_count = delete_backup_files(backup_files, args.dry_run)
    
    if args.dry_run:
        logger.info(f"[DRY RUN] Se eliminarían {deleted_count} archivos/directorios de respaldo")
    else:
        logger.info(f"Se eliminaron {deleted_count} archivos/directorios de respaldo")
        if error_count > 0:
            logger.warning(f"Hubo {error_count} errores durante la eliminación")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

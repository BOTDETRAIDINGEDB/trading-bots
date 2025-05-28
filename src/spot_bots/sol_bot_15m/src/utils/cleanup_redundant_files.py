#!/usr/bin/env python3
"""
Script para analizar y limpiar archivos redundantes en el proyecto.

Este script realiza las siguientes tareas:
1. Identifica archivos duplicados o con nombres similares
2. Detecta archivos de respaldo o temporales
3. Busca archivos que ya no son referenciados por ningún otro archivo
4. Genera un informe de archivos que pueden ser eliminados
5. Permite eliminar archivos redundantes de forma segura

Uso:
    python cleanup_redundant_files.py [--analyze] [--clean] [--dry-run]

Opciones:
    --analyze: Solo analiza y muestra un informe (predeterminado)
    --clean: Elimina los archivos redundantes identificados
    --dry-run: Muestra qué archivos se eliminarían sin eliminarlos realmente
"""

import os
import sys
import hashlib
import re
import argparse
import logging
import json
import shutil
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CLEANUP")

# Patrones para identificar archivos temporales o de respaldo
BACKUP_PATTERNS = [
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
    r'.*__pycache__.*',
    r'.*\.git.*',
    r'.*\.DS_Store$',
    r'.*\.idea.*',
    r'.*\.vscode.*',
    r'.*\.ipynb_checkpoints.*',
]

# Extensiones de archivos a ignorar completamente
IGNORE_EXTENSIONS = [
    '.pyc', '.pyo', '.pyd', '.so', '.dll', '.exe', '.obj', '.o',
    '.a', '.lib', '.swp', '.swo', '.swn', '.swm', '.swl',
    '.DS_Store', '.git', '.gitignore', '.gitmodules', '.gitattributes',
    '.ipynb_checkpoints', '.idea', '.vscode', '__pycache__'
]

# Archivos importantes que nunca deben ser eliminados
CRITICAL_FILES = [
    'requirements.txt',
    'setup.py',
    'README.md',
    'LICENSE',
    'CONTRIBUTING.md',
    'CHANGELOG.md',
    'Dockerfile',
    'docker-compose.yml',
    '.env.example',
    'credentials_template.json',
    'config.json',
    'config_template.json',
    'main.py',
    'app.py',
    'run.py',
    'start.sh',
    'stop.sh',
    'restart.sh',
    'deploy.sh',
    'deploy_to_cloud.py',
    'cloud_init.py',
    'cloud_optimizer.py',
    'lstm_model.py',
    'lstm_model_part2.py',
    'lstm_model_part3.py',
    '__init__.py',
]

def get_file_hash(file_path: str) -> str:
    """Calcula el hash SHA256 de un archivo."""
    try:
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    except Exception as e:
        logger.warning(f"No se pudo calcular el hash para {file_path}: {str(e)}")
        return ""

def is_backup_file(file_path: str) -> bool:
    """Determina si un archivo es un archivo de respaldo o temporal."""
    file_name = os.path.basename(file_path)
    for pattern in BACKUP_PATTERNS:
        if re.match(pattern, file_name):
            return True
    return False

def should_ignore_file(file_path: str) -> bool:
    """Determina si un archivo debe ser ignorado."""
    file_name = os.path.basename(file_path)
    _, ext = os.path.splitext(file_name)
    
    # Ignorar archivos con extensiones específicas
    if ext.lower() in IGNORE_EXTENSIONS:
        return True
    
    # Ignorar archivos críticos
    if file_name in CRITICAL_FILES:
        return True
    
    # Ignorar directorios específicos
    for ignore_dir in ['.git', '__pycache__', '.idea', '.vscode', '.ipynb_checkpoints']:
        if ignore_dir in file_path.split(os.sep):
            return True
    
    return False

def find_all_files(root_dir: str) -> List[str]:
    """Encuentra todos los archivos en un directorio y sus subdirectorios."""
    all_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Ignorar directorios específicos
        dirnames[:] = [d for d in dirnames if d not in ['.git', '__pycache__', '.idea', '.vscode', '.ipynb_checkpoints']]
        
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if not should_ignore_file(file_path):
                all_files.append(file_path)
    
    return all_files

def find_duplicate_files(files: List[str]) -> Dict[str, List[str]]:
    """Encuentra archivos duplicados basados en su contenido (hash)."""
    hash_to_files = defaultdict(list)
    
    for file_path in files:
        file_hash = get_file_hash(file_path)
        if file_hash:
            hash_to_files[file_hash].append(file_path)
    
    # Filtrar para mantener solo los hashes con más de un archivo
    return {h: files for h, files in hash_to_files.items() if len(files) > 1}

def find_similar_named_files(files: List[str]) -> List[List[str]]:
    """Encuentra archivos con nombres similares que podrían ser versiones diferentes."""
    # Agrupar archivos por nombre base (sin extensión)
    base_name_to_files = defaultdict(list)
    
    for file_path in files:
        dir_name = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        base_name, _ = os.path.splitext(file_name)
        
        # Eliminar sufijos comunes como _old, _backup, _v1, etc.
        base_name = re.sub(r'_v\d+$|_old$|_backup$|_bak$|_new$|_temp$|_tmp$', '', base_name)
        
        # Agrupar por directorio + nombre base
        key = os.path.join(dir_name, base_name)
        base_name_to_files[key].append(file_path)
    
    # Filtrar para mantener solo los grupos con más de un archivo
    return [files for files in base_name_to_files.values() if len(files) > 1]

def find_backup_files(files: List[str]) -> List[str]:
    """Encuentra archivos de respaldo o temporales."""
    return [file_path for file_path in files if is_backup_file(file_path)]

def find_unreferenced_files(files: List[str], root_dir: str) -> List[str]:
    """Encuentra archivos que no son referenciados por ningún otro archivo."""
    # Construir un conjunto de todos los archivos
    all_files_set = set(files)
    
    # Conjunto de archivos referenciados
    referenced_files = set()
    
    # Buscar referencias en archivos Python
    for file_path in files:
        if file_path.endswith('.py'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Buscar patrones de importación
                import_patterns = [
                    r'from\s+(\S+)\s+import',
                    r'import\s+(\S+)',
                    r'__import__\([\'"](\S+)[\'"]\)'
                ]
                
                for pattern in import_patterns:
                    for match in re.finditer(pattern, content):
                        module_name = match.group(1)
                        # Convertir el nombre del módulo a una ruta de archivo relativa
                        module_path = module_name.replace('.', os.sep)
                        
                        # Buscar posibles archivos que coincidan con este módulo
                        for potential_file in all_files_set:
                            rel_path = os.path.relpath(potential_file, root_dir)
                            rel_path_no_ext, _ = os.path.splitext(rel_path)
                            
                            if rel_path_no_ext == module_path or rel_path_no_ext.endswith(module_path):
                                referenced_files.add(potential_file)
                
                # Buscar referencias a archivos directamente
                file_patterns = [
                    r'open\([\'"](.+?)[\'"]\)',
                    r'with\s+open\([\'"](.+?)[\'"]\)',
                    r'os\.path\.join\(.+?[\'"](.+?)[\'"]\)',
                    r'[\'"](.+?\.(py|json|yaml|yml|txt|md|csv|xlsx|html|js|css))[\'"]\)'
                ]
                
                for pattern in file_patterns:
                    for match in re.finditer(pattern, content):
                        ref_path = match.group(1)
                        # Intentar resolver la ruta relativa
                        abs_ref_path = os.path.abspath(os.path.join(os.path.dirname(file_path), ref_path))
                        if abs_ref_path in all_files_set:
                            referenced_files.add(abs_ref_path)
            
            except Exception as e:
                logger.warning(f"Error al analizar {file_path}: {str(e)}")
    
    # Archivos que no están referenciados
    unreferenced = all_files_set - referenced_files
    
    # Excluir archivos críticos y archivos en la raíz del proyecto
    result = []
    for file_path in unreferenced:
        file_name = os.path.basename(file_path)
        if file_name not in CRITICAL_FILES and not os.path.dirname(file_path) == root_dir:
            result.append(file_path)
    
    return result

def analyze_project(root_dir: str) -> Dict[str, Any]:
    """Analiza el proyecto y encuentra archivos redundantes."""
    logger.info(f"Analizando proyecto en {root_dir}...")
    
    # Encontrar todos los archivos
    all_files = find_all_files(root_dir)
    logger.info(f"Encontrados {len(all_files)} archivos para analizar")
    
    # Encontrar archivos duplicados
    duplicate_files = find_duplicate_files(all_files)
    logger.info(f"Encontrados {len(duplicate_files)} grupos de archivos duplicados")
    
    # Encontrar archivos con nombres similares
    similar_named_files = find_similar_named_files(all_files)
    logger.info(f"Encontrados {len(similar_named_files)} grupos de archivos con nombres similares")
    
    # Encontrar archivos de respaldo
    backup_files = find_backup_files(all_files)
    logger.info(f"Encontrados {len(backup_files)} archivos de respaldo o temporales")
    
    # Encontrar archivos no referenciados
    unreferenced_files = find_unreferenced_files(all_files, root_dir)
    logger.info(f"Encontrados {len(unreferenced_files)} archivos no referenciados")
    
    # Crear informe
    report = {
        "timestamp": datetime.now().isoformat(),
        "root_directory": root_dir,
        "total_files": len(all_files),
        "duplicate_files": {hash_val: files for hash_val, files in duplicate_files.items()},
        "similar_named_files": similar_named_files,
        "backup_files": backup_files,
        "unreferenced_files": unreferenced_files,
        "recommended_for_removal": []
    }
    
    # Generar lista de archivos recomendados para eliminar
    for hash_val, files in duplicate_files.items():
        # Mantener el archivo con la ruta más corta o más simple
        files_sorted = sorted(files, key=lambda x: (len(x.split(os.sep)), len(x)))
        for file_path in files_sorted[1:]:  # Todos excepto el primero
            report["recommended_for_removal"].append({
                "file_path": file_path,
                "reason": "Duplicado de " + files_sorted[0],
                "category": "duplicate"
            })
    
    for backup_file in backup_files:
        report["recommended_for_removal"].append({
            "file_path": backup_file,
            "reason": "Archivo de respaldo o temporal",
            "category": "backup"
        })
    
    for unreferenced_file in unreferenced_files:
        # Excluir archivos importantes que podrían no ser referenciados directamente
        file_name = os.path.basename(unreferenced_file)
        if not any(file_name.startswith(critical) for critical in CRITICAL_FILES):
            report["recommended_for_removal"].append({
                "file_path": unreferenced_file,
                "reason": "Archivo no referenciado por ningún otro archivo",
                "category": "unreferenced"
            })
    
    return report

def print_report(report: Dict[str, Any]) -> None:
    """Imprime un informe de análisis en formato legible."""
    print("\n" + "="*80)
    print(f"INFORME DE ANÁLISIS DE ARCHIVOS REDUNDANTES")
    print(f"Fecha: {report['timestamp']}")
    print(f"Directorio: {report['root_directory']}")
    print("="*80)
    
    print(f"\nTotal de archivos analizados: {report['total_files']}")
    print(f"Grupos de archivos duplicados: {len(report['duplicate_files'])}")
    print(f"Grupos de archivos con nombres similares: {len(report['similar_named_files'])}")
    print(f"Archivos de respaldo o temporales: {len(report['backup_files'])}")
    print(f"Archivos no referenciados: {len(report['unreferenced_files'])}")
    print(f"Archivos recomendados para eliminar: {len(report['recommended_for_removal'])}")
    
    if report['recommended_for_removal']:
        print("\nARCHIVOS RECOMENDADOS PARA ELIMINAR:")
        print("-"*80)
        for i, item in enumerate(report['recommended_for_removal'], 1):
            print(f"{i}. {item['file_path']}")
            print(f"   Razón: {item['reason']}")
            print(f"   Categoría: {item['category']}")
            print()
    
    print("="*80)
    print("Para eliminar estos archivos, ejecute el script con la opción --clean")
    print("Para ver qué archivos se eliminarían sin eliminarlos, use --dry-run")
    print("="*80 + "\n")

def save_report(report: Dict[str, Any], output_file: str) -> None:
    """Guarda el informe en un archivo JSON."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Informe guardado en {output_file}")
    except Exception as e:
        logger.error(f"Error al guardar el informe: {str(e)}")

def clean_redundant_files(report: Dict[str, Any], dry_run: bool = False) -> None:
    """Elimina los archivos redundantes identificados en el informe."""
    if not report['recommended_for_removal']:
        logger.info("No hay archivos para eliminar")
        return
    
    # Crear directorio de respaldo
    backup_dir = os.path.join(report['root_directory'], f"backup_files_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    if not dry_run:
        os.makedirs(backup_dir, exist_ok=True)
        logger.info(f"Creado directorio de respaldo: {backup_dir}")
    
    # Eliminar archivos
    for item in report['recommended_for_removal']:
        file_path = item['file_path']
        try:
            if dry_run:
                logger.info(f"[DRY RUN] Se eliminaría: {file_path}")
            else:
                # Crear estructura de directorios en el backup
                rel_path = os.path.relpath(file_path, report['root_directory'])
                backup_path = os.path.join(backup_dir, rel_path)
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                
                # Copiar archivo al backup
                shutil.copy2(file_path, backup_path)
                logger.info(f"Archivo respaldado en: {backup_path}")
                
                # Eliminar archivo original
                os.remove(file_path)
                logger.info(f"Archivo eliminado: {file_path}")
        except Exception as e:
            logger.error(f"Error al procesar {file_path}: {str(e)}")
    
    if not dry_run:
        logger.info(f"Eliminados {len(report['recommended_for_removal'])} archivos redundantes")
        logger.info(f"Los archivos fueron respaldados en: {backup_dir}")

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Analiza y limpia archivos redundantes en el proyecto")
    parser.add_argument("--analyze", action="store_true", help="Solo analiza y muestra un informe (predeterminado)")
    parser.add_argument("--clean", action="store_true", help="Elimina los archivos redundantes identificados")
    parser.add_argument("--dry-run", action="store_true", help="Muestra qué archivos se eliminarían sin eliminarlos realmente")
    parser.add_argument("--output", help="Archivo de salida para el informe (JSON)")
    parser.add_argument("--dir", help="Directorio raíz del proyecto a analizar")
    
    args = parser.parse_args()
    
    # Si no se especifica ninguna opción, usar --analyze
    if not (args.analyze or args.clean or args.dry_run):
        args.analyze = True
    
    # Determinar el directorio raíz
    root_dir = args.dir if args.dir else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Analizar el proyecto
    report = analyze_project(root_dir)
    
    # Guardar informe si se especificó un archivo de salida
    if args.output:
        save_report(report, args.output)
    
    # Imprimir informe
    print_report(report)
    
    # Limpiar archivos redundantes si se solicitó
    if args.clean or args.dry_run:
        clean_redundant_files(report, dry_run=args.dry_run or not args.clean)

if __name__ == "__main__":
    main()

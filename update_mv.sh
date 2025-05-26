#!/bin/bash
# Script para actualizar ambos repositorios en la máquina virtual
# Autor: BOTDETRAIDINGEDB
# Fecha: 26-05-2025

# Colores para mensajes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}==================================================${NC}"
echo -e "${YELLOW}  ACTUALIZACIÓN DE BOTS DE TRADING EN MÁQUINA VIRTUAL ${NC}"
echo -e "${YELLOW}==================================================${NC}"

# Verificar que estamos en la máquina virtual
if [[ "$(hostname)" != *"vm"* && "$(hostname)" != *"VM"* ]]; then
  echo -e "${YELLOW}ADVERTENCIA: Este script está diseñado para ejecutarse en la máquina virtual.${NC}"
  echo -e "${YELLOW}Parece que estás ejecutándolo en un entorno local.${NC}"
  read -p "¿Deseas continuar de todos modos? (s/n): " confirm
  if [[ $confirm != "s" && $confirm != "S" ]]; then
    echo -e "${RED}Operación cancelada.${NC}"
    exit 1
  fi
fi

# Función para actualizar un repositorio
update_repo() {
  local repo_path=$1
  local repo_name=$2
  
  echo -e "\n${YELLOW}Actualizando repositorio: ${repo_name}${NC}"
  
  # Verificar que el directorio existe
  if [ ! -d "$repo_path" ]; then
    echo -e "${RED}Error: El directorio $repo_path no existe.${NC}"
    return 1
  fi
  
  # Cambiar al directorio del repositorio
  cd "$repo_path"
  
  # Verificar que es un repositorio git
  if [ ! -d ".git" ]; then
    echo -e "${RED}Error: $repo_path no es un repositorio Git.${NC}"
    return 1
  fi
  
  # Guardar cambios locales si existen
  if [[ $(git status --porcelain) ]]; then
    echo -e "${YELLOW}Se detectaron cambios locales. Guardando en stash...${NC}"
    git stash save "Cambios automáticos antes de actualizar - $(date)"
  fi
  
  # Obtener la rama actual
  current_branch=$(git symbolic-ref --short HEAD)
  echo -e "${YELLOW}Rama actual: $current_branch${NC}"
  
  # Actualizar desde el repositorio remoto
  echo -e "${YELLOW}Obteniendo cambios del repositorio remoto...${NC}"
  git fetch origin
  
  # Verificar si hay cambios para evitar conflictos
  if git diff --quiet HEAD origin/$current_branch; then
    echo -e "${GREEN}No hay cambios nuevos en el repositorio remoto.${NC}"
  else
    echo -e "${YELLOW}Se detectaron cambios en el repositorio remoto. Actualizando...${NC}"
    
    # Realizar el pull
    if git pull origin $current_branch; then
      echo -e "${GREEN}Repositorio $repo_name actualizado correctamente.${NC}"
    else
      echo -e "${RED}Error al actualizar el repositorio $repo_name.${NC}"
      echo -e "${YELLOW}Puede haber conflictos que requieran resolución manual.${NC}"
      return 1
    fi
  fi
  
  return 0
}

# Actualizar el repositorio trading-bots
TRADING_BOTS_PATH="/home/usuario/trading-bots"  # Ajusta esta ruta según tu entorno
echo -e "\n${YELLOW}==================================================${NC}"
echo -e "${YELLOW}  ACTUALIZANDO TRADING BOTS ${NC}"
echo -e "${YELLOW}==================================================${NC}"
update_repo "$TRADING_BOTS_PATH" "trading-bots"
TRADING_BOTS_RESULT=$?

# Actualizar el repositorio trading-bots-api
TRADING_BOTS_API_PATH="/home/usuario/trading-bots-api"  # Ajusta esta ruta según tu entorno
echo -e "\n${YELLOW}==================================================${NC}"
echo -e "${YELLOW}  ACTUALIZANDO TRADING BOTS API ${NC}"
echo -e "${YELLOW}==================================================${NC}"
update_repo "$TRADING_BOTS_API_PATH" "trading-bots-api"
TRADING_BOTS_API_RESULT=$?

# Instalar dependencias actualizadas si es necesario
if [ $TRADING_BOTS_RESULT -eq 0 ]; then
  echo -e "\n${YELLOW}Instalando dependencias actualizadas para trading-bots...${NC}"
  cd "$TRADING_BOTS_PATH"
  if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}Dependencias de trading-bots instaladas correctamente.${NC}"
  else
    echo -e "${YELLOW}No se encontró archivo requirements.txt en trading-bots.${NC}"
  fi
fi

if [ $TRADING_BOTS_API_RESULT -eq 0 ]; then
  echo -e "\n${YELLOW}Instalando dependencias actualizadas para trading-bots-api...${NC}"
  cd "$TRADING_BOTS_API_PATH"
  if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}Dependencias de trading-bots-api instaladas correctamente.${NC}"
  else
    echo -e "${YELLOW}No se encontró archivo requirements.txt en trading-bots-api.${NC}"
  fi
fi

# Resumen final
echo -e "\n${YELLOW}==================================================${NC}"
echo -e "${YELLOW}  RESUMEN DE LA ACTUALIZACIÓN ${NC}"
echo -e "${YELLOW}==================================================${NC}"

if [ $TRADING_BOTS_RESULT -eq 0 ]; then
  echo -e "${GREEN}✓ Repositorio trading-bots actualizado correctamente.${NC}"
else
  echo -e "${RED}✗ Error al actualizar el repositorio trading-bots.${NC}"
fi

if [ $TRADING_BOTS_API_RESULT -eq 0 ]; then
  echo -e "${GREEN}✓ Repositorio trading-bots-api actualizado correctamente.${NC}"
else
  echo -e "${RED}✗ Error al actualizar el repositorio trading-bots-api.${NC}"
fi

echo -e "\n${YELLOW}Nota: Asegúrate de verificar que los bots funcionen correctamente después de la actualización.${NC}"
echo -e "${YELLOW}Si encuentras algún problema, revisa los logs o contacta al administrador.${NC}"

# Finalizar
echo -e "\n${GREEN}Proceso de actualización completado.${NC}"

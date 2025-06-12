#!/bin/bash
# =============================================================================
# SOL BOT - SCRIPT DE INICIO PARA MODO REAL (GOOGLE CLOUD)
# =============================================================================
# 
# Descripción: Script profesional para iniciar el bot SOL en modo real
# Autor: Edison Bautista
# Versión: 1.0.0
# Fecha: 2025-06-12
# 
# ADVERTENCIA: Este script opera con dinero real. Use con precaución.
# =============================================================================

set -euo pipefail  # Salir en caso de error, variables no definidas o pipes fallidos

# =============================================================================
# CONFIGURACIÓN Y COLORES
# =============================================================================

# Colores para mejor legibilidad
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly WHITE='\033[1;37m'
readonly NC='\033[0m' # No Color

# Configuración del bot
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly BOT_NAME="sol_bot_real"
readonly SYMBOL="SOLUSDT"
readonly INTERVAL="15m"
readonly RISK="0.015"              # 1.5% riesgo (conservador para real)
readonly RETRAIN_INTERVAL="90"     # 90 minutos reentrenamiento
readonly STATUS_INTERVAL="4"       # 4 horas actualización de estado
readonly MIN_BALANCE="50.0"        # Balance mínimo requerido en USDT

# Archivos y directorios
readonly LOGS_DIR="$SCRIPT_DIR/logs"
readonly CREDENTIALS_FILE="$HOME/.binance_credentials"
readonly PID_FILE="$SCRIPT_DIR/${BOT_NAME}.pid"
readonly LOG_FILE="$LOGS_DIR/${BOT_NAME}_$(date +%Y%m%d_%H%M%S).log"

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

# Función para mostrar mensajes con timestamp
log_message() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "$level" in
        "INFO")  echo -e "${GREEN}[${timestamp}] INFO: ${message}${NC}" ;;
        "WARN")  echo -e "${YELLOW}[${timestamp}] WARN: ${message}${NC}" ;;
        "ERROR") echo -e "${RED}[${timestamp}] ERROR: ${message}${NC}" ;;
        "DEBUG") echo -e "${BLUE}[${timestamp}] DEBUG: ${message}${NC}" ;;
        *)       echo -e "${WHITE}[${timestamp}] ${message}${NC}" ;;
    esac
}

# Función para mostrar banner profesional
show_banner() {
    echo -e "${CYAN}"
    echo "============================================================================="
    echo "                        🤖 SOL TRADING BOT - MODO REAL 🤖"
    echo "============================================================================="
    echo -e "${WHITE}Versión: 1.0.0                    Fecha: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo -e "${RED}⚠️  ADVERTENCIA: ESTE BOT OPERA CON DINERO REAL ⚠️${NC}"
    echo -e "${CYAN}=============================================================================${NC}"
}

# Función para validar dependencias
validate_dependencies() {
    log_message "INFO" "Validando dependencias del sistema..."
    
    local dependencies=("python3" "screen" "curl")
    for dep in "${dependencies[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            log_message "ERROR" "Dependencia faltante: $dep"
            exit 1
        fi
    done
    
    log_message "INFO" "✅ Todas las dependencias están instaladas"
}

# Función para validar credenciales
validate_credentials() {
    log_message "INFO" "Validando credenciales de Binance..."
    
    if [[ ! -f "$CREDENTIALS_FILE" ]]; then
        log_message "ERROR" "Archivo de credenciales no encontrado: $CREDENTIALS_FILE"
        log_message "ERROR" "Configure sus credenciales de Binance antes de continuar"
        exit 1
    fi
    
    # Verificar que el archivo contiene las claves necesarias
    if ! grep -q "api_key" "$CREDENTIALS_FILE" || ! grep -q "api_secret" "$CREDENTIALS_FILE"; then
        log_message "ERROR" "Credenciales incompletas en: $CREDENTIALS_FILE"
        exit 1
    fi
    
    log_message "INFO" "✅ Credenciales validadas correctamente"
}

# Función para verificar balance mínimo
verify_minimum_balance() {
    log_message "INFO" "Verificando balance mínimo requerido ($MIN_BALANCE USDT)..."
    
    # Nota: En un entorno real, aquí se haría una llamada a la API de Binance
    # para verificar el balance actual. Por ahora, mostramos el mensaje.
    log_message "WARN" "⚠️  Verifique manualmente que tiene al menos $MIN_BALANCE USDT en su cuenta"
    log_message "WARN" "⚠️  El bot se detendrá automáticamente si el balance es insuficiente"
}

# Función para verificar si el bot ya está ejecutándose
check_existing_bot() {
    log_message "INFO" "Verificando instancias previas del bot..."
    
    # Lista de posibles nombres de sesiones del bot
    local bot_sessions=("sol_bot_real" "sol_bot" "sol_bot_15m" "sol_bot_sim")
    local found_sessions=()
    
    # Buscar todas las sesiones relacionadas con el bot SOL
    for session in "${bot_sessions[@]}"; do
        if screen -ls | grep -q "$session"; then
            found_sessions+=("$session")
        fi
    done
    
    # Si encontramos sesiones, manejarlas automáticamente
    if [ ${#found_sessions[@]} -gt 0 ]; then
        log_message "WARN" "⚠️  Se encontraron ${#found_sessions[@]} sesión(es) del bot ejecutándose:"
        for session in "${found_sessions[@]}"; do
            log_message "WARN" "   - $session"
        done
        
        echo -e "${YELLOW}"
        echo "============================================================================="
        echo "                     🔄 LIMPIEZA AUTOMÁTICA REQUERIDA"
        echo "============================================================================="
        echo "Se detectaron bots previos ejecutándose. Para evitar conflictos,"
        echo "se cerrarán automáticamente las sesiones anteriores antes de iniciar"
        echo "el bot en modo REAL."
        echo ""
        echo -e "${RED}¿Desea cerrar automáticamente las sesiones previas? (escriba 'SI')${NC}"
        
        read -r cleanup_confirmation
        if [[ "$cleanup_confirmation" != "SI" ]]; then
            log_message "INFO" "Operación cancelada por el usuario"
            log_message "INFO" "Use './stop.sh' o 'screen -S <nombre_sesion> -X quit' para cerrar manualmente"
            exit 0
        fi
        
        # Cerrar sesiones automáticamente
        log_message "INFO" "🧹 Iniciando limpieza automática de sesiones..."
        
        for session in "${found_sessions[@]}"; do
            log_message "INFO" "Cerrando sesión: $session"
            screen -S "$session" -X quit 2>/dev/null || true
            sleep 1
        done
        
        # Esperar a que las sesiones se cierren completamente
        log_message "INFO" "Esperando 5 segundos para completar el cierre..."
        sleep 5
        
        # Verificar que todas las sesiones se cerraron
        local remaining_sessions=()
        for session in "${found_sessions[@]}"; do
            if screen -ls | grep -q "$session"; then
                remaining_sessions+=("$session")
            fi
        done
        
        if [ ${#remaining_sessions[@]} -gt 0 ]; then
            log_message "ERROR" "❌ No se pudieron cerrar todas las sesiones:"
            for session in "${remaining_sessions[@]}"; do
                log_message "ERROR" "   - $session (aún activa)"
            done
            log_message "ERROR" "Cierre manualmente con: screen -S <nombre_sesion> -X quit"
            exit 1
        fi
        
        log_message "INFO" "✅ Todas las sesiones previas fueron cerradas exitosamente"
    fi
    
    # Verificar y limpiar archivos PID huérfanos
    local pid_files=("sol_bot_real.pid" "sol_bot.pid" "sol_bot_15m.pid")
    for pid_file in "${pid_files[@]}"; do
        if [[ -f "$pid_file" ]]; then
            local pid=$(cat "$pid_file" 2>/dev/null || echo "")
            if [[ -n "$pid" ]] && ! ps -p "$pid" > /dev/null 2>&1; then
                log_message "WARN" "Limpiando archivo PID huérfano: $pid_file"
                rm -f "$pid_file"
            fi
        fi
    done
    
    # Verificación final
    if screen -ls | grep -qE "(sol_bot|bot)" 2>/dev/null; then
        log_message "ERROR" "❌ Aún existen sesiones de bot activas:"
        screen -ls | grep -E "(sol_bot|bot)" || true
        log_message "ERROR" "Cierre todas las sesiones manualmente antes de continuar"
        exit 1
    fi
    
    log_message "INFO" "✅ Sistema listo para iniciar el bot en modo REAL"
}

# Función para crear directorios necesarios
setup_directories() {
    log_message "INFO" "Configurando directorios..."
    
    mkdir -p "$LOGS_DIR"
    
    log_message "INFO" "✅ Directorios configurados"
}

# Función para mostrar configuración
show_configuration() {
    echo -e "${PURPLE}"
    echo "============================================================================="
    echo "                           CONFIGURACIÓN DEL BOT"
    echo "============================================================================="
    echo -e "${WHITE}• Símbolo:                    ${SYMBOL}${NC}"
    echo -e "${WHITE}• Intervalo:                  ${INTERVAL}${NC}"
    echo -e "${WHITE}• Riesgo por operación:       ${RISK}% (conservador)${NC}"
    echo -e "${WHITE}• Intervalo de reentrenamiento: ${RETRAIN_INTERVAL} minutos${NC}"
    echo -e "${WHITE}• Intervalo de estado:        ${STATUS_INTERVAL} horas${NC}"
    echo -e "${WHITE}• Balance mínimo requerido:   ${MIN_BALANCE} USDT${NC}"
    echo -e "${WHITE}• Archivo de logs:            ${LOG_FILE}${NC}"
    echo -e "${RED}• Modo:                       🚨 REAL (DINERO REAL) 🚨${NC}"
    echo -e "${PURPLE}=============================================================================${NC}"
}

# Función para confirmar inicio
confirm_start() {
    echo -e "${RED}"
    echo "⚠️⚠️⚠️  CONFIRMACIÓN REQUERIDA  ⚠️⚠️⚠️"
    echo ""
    echo "Está a punto de iniciar el bot en MODO REAL con dinero real."
    echo "El bot realizará operaciones de compra y venta automáticamente."
    echo ""
    echo -e "${YELLOW}¿Está seguro de que desea continuar? (escriba 'SI' en mayúsculas)${NC}"
    
    read -r confirmation
    if [[ "$confirmation" != "SI" ]]; then
        log_message "INFO" "Operación cancelada por el usuario"
        exit 0
    fi
    
    log_message "INFO" "✅ Confirmación recibida. Iniciando bot..."
}

# Función para iniciar el bot
start_bot() {
    log_message "INFO" "Iniciando bot SOL en modo real..."
    
    cd "$SCRIPT_DIR"
    
    # Comando para iniciar el bot SIN --simulation (modo real)
    screen -dmS "$BOT_NAME" bash -c "
        python3 main.py \\
            --symbol '$SYMBOL' \\
            --interval '$INTERVAL' \\
            --use-ml \\
            --retrain-interval '$RETRAIN_INTERVAL' \\
            --risk '$RISK' \\
            --status-interval '$STATUS_INTERVAL' \\
            2>&1 | tee '$LOG_FILE'
    "
    
    # Obtener y guardar PID
    sleep 2
    local screen_pid=$(screen -ls | grep "$BOT_NAME" | awk '{print $1}' | cut -d. -f1)
    
    if [[ -n "$screen_pid" ]]; then
        echo "$screen_pid" > "$PID_FILE"
        log_message "INFO" "✅ Bot iniciado exitosamente"
        log_message "INFO" "PID de la sesión screen: $screen_pid"
    else
        log_message "ERROR" "❌ Error al iniciar el bot"
        exit 1
    fi
}

# Función para mostrar información post-inicio
show_post_start_info() {
    echo -e "${GREEN}"
    echo "============================================================================="
    echo "                        🚀 BOT INICIADO EXITOSAMENTE 🚀"
    echo "============================================================================="
    echo -e "${WHITE}Comandos útiles:${NC}"
    echo -e "${CYAN}• Ver logs en tiempo real:    ${WHITE}screen -r $BOT_NAME${NC}"
    echo -e "${CYAN}• Desconectarse sin detener:  ${WHITE}Ctrl+A, D${NC}"
    echo -e "${CYAN}• Ver logs guardados:         ${WHITE}tail -f $LOG_FILE${NC}"
    echo -e "${CYAN}• Detener el bot:             ${WHITE}./stop.sh${NC}"
    echo -e "${CYAN}• Estado de sesiones:         ${WHITE}screen -ls${NC}"
    echo -e "${GREEN}=============================================================================${NC}"
    
    echo -e "${RED}⚠️ RECORDATORIO: EL BOT ESTÁ OPERANDO CON DINERO REAL ⚠️${NC}"
    echo -e "${YELLOW}Monitoree regularmente las operaciones y el balance de su cuenta.${NC}"
}

# Función de limpieza en caso de error
cleanup_on_error() {
    log_message "ERROR" "Error detectado. Ejecutando limpieza..."
    
    # Matar sesión screen si existe
    if screen -ls | grep -q "$BOT_NAME"; then
        screen -S "$BOT_NAME" -X quit 2>/dev/null || true
    fi
    
    # Remover archivo PID
    rm -f "$PID_FILE"
    
    log_message "ERROR" "Limpieza completada"
    exit 1
}

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

main() {
    # Configurar trap para limpieza en caso de error
    trap cleanup_on_error ERR
    
    # Verificar que estamos en el directorio correcto
    if [[ ! -f "main.py" ]]; then
        log_message "ERROR" "Script debe ejecutarse desde el directorio del bot SOL"
        exit 1
    fi
    
    # Ejecutar validaciones y configuración
    show_banner
    validate_dependencies
    validate_credentials
    check_existing_bot
    setup_directories
    verify_minimum_balance
    show_configuration
    confirm_start
    start_bot
    show_post_start_info
    
    log_message "INFO" "🎉 Inicio completado exitosamente"
}

# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

# Ejecutar función principal solo si el script se ejecuta directamente
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

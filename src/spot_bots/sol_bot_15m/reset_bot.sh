#!/bin/bash
# reset_bot.sh - Script para reiniciar completamente el bot, limpiar estado y comenzar de cero

# Colores para mejor legibilidad
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== REINICIO COMPLETO DEL BOT SOL ===${NC}"

# 1. Detener el bot si está en ejecución
echo -e "${YELLOW}Deteniendo bot si está en ejecución...${NC}"
./stop.sh

# 2. Hacer copia de seguridad del estado actual
echo -e "${YELLOW}Creando copia de seguridad del estado...${NC}"
if [ -f sol_bot_15min_state.json ]; then
    cp sol_bot_15min_state.json "sol_bot_15min_state_backup_$(date +%Y%m%d_%H%M%S).json"
    echo -e "${GREEN}Copia de seguridad creada.${NC}"
    
    # 3. Extraer configuraciones importantes para preservarlas
    echo -e "${YELLOW}Extrayendo configuraciones importantes...${NC}"
    RISK=$(grep -o '"risk_per_trade": [0-9.]*' sol_bot_15min_state.json | cut -d' ' -f2)
    SL=$(grep -o '"stop_loss_pct": [0-9.]*' sol_bot_15min_state.json | cut -d' ' -f2)
    TP=$(grep -o '"take_profit_pct": [0-9.]*' sol_bot_15min_state.json | cut -d' ' -f2)
    TRAILING=$(grep -o '"trailing_percent": [0-9.]*' sol_bot_15min_state.json | cut -d' ' -f2)
    MAX_TRADES=$(grep -o '"max_trades": [0-9]*' sol_bot_15min_state.json | cut -d' ' -f2)
    USE_ML=$(grep -o '"use_ml": [a-z]*' sol_bot_15min_state.json | cut -d' ' -f2)
    BALANCE=$(grep -o '"current_balance": [0-9.]*' sol_bot_15min_state.json | cut -d' ' -f2)
    
    # 4. Crear nuevo archivo de estado limpio
    echo -e "${YELLOW}Creando nuevo estado limpio...${NC}"
    cat > sol_bot_15min_state.json << EOL
{
    "symbol": "SOLUSDT",
    "risk_per_trade": ${RISK:-0.03},
    "stop_loss_pct": ${SL:-0.06},
    "take_profit_pct": ${TP:-0.04},
    "max_trades": ${MAX_TRADES:-3},
    "use_ml": ${USE_ML:-true},
    "position": 0,
    "entry_price": 0,
    "position_size": 0,
    "position_amount": 0,
    "stop_loss": 0,
    "take_profit": 0,
    "trailing_stop": 0.0,
    "trailing_active": false,
    "trailing_percent": ${TRAILING:-0.015},
    "highest_price": 0,
    "trades": [],
    "current_balance": ${BALANCE:-100.0},
    "initial_balance": ${BALANCE:-100.0},
    "performance_metrics": {
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "win_rate": 0.0,
        "avg_profit": 0.0,
        "avg_loss": 0.0,
        "profit_factor": 0.0,
        "max_drawdown": 0.0,
        "current_drawdown": 0.0,
        "total_profit": 0.0
    },
    "market_conditions": {
        "trend_strength": 0,
        "volatility": 0,
        "rsi": 50,
        "volume_change": 0
    },
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")"
}
EOL
    echo -e "${GREEN}Nuevo estado limpio creado.${NC}"
else
    echo -e "${RED}No se encontró archivo de estado. Se creará uno nuevo al iniciar el bot.${NC}"
fi

# 5. Limpiar otros archivos temporales
echo -e "${YELLOW}Limpiando archivos temporales...${NC}"
rm -f *.pid
rm -f .last_startup
echo -e "${GREEN}Archivos temporales eliminados.${NC}"

# 6. Reiniciar el bot
echo -e "${YELLOW}Reiniciando el bot...${NC}"
./cleanup_bot_sessions.sh

echo -e "${GREEN}=== REINICIO COMPLETO FINALIZADO ===${NC}"
echo -e "${GREEN}El bot se ha reiniciado con un estado limpio, preservando las configuraciones.${NC}"
echo -e "${BLUE}Para ver los logs: tail -f logs/sol_bot_15m_cloud_simulation_*.log${NC}"

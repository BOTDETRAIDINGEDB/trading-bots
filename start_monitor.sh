#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
source venv/bin/activate
python3 -m src.spot_bots.sol_bot.telegram_monitor

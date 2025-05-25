# Cryptocurrency Trading Bots

Automated trading bots for cryptocurrency spot and futures markets, with support for XRP trading pairs.

## Features

- Spot Trading Bots:
  - XRP Trading Bot
- Futures Trading Bots (Coming Soon)
- Real-time market data analysis
- Telegram notifications
- Secure credential management

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your credentials

## Usage

Each bot can be run independently:

```bash
python src/spot_bots/xrp_bot/main.py
python src/spot_bots/sol_bot/main.py
```

## Security

- Never commit sensitive data to this repository
- Store all API keys and credentials in `.env` file
- Keep backup of credentials in secure storage

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
Bots comerciales automatizados para mercados al contado y de futuros de criptomonedas

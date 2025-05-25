from setuptools import setup, find_packages

setup(
    name="trading-bots",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    python_requires=">=3.8",
    author="Edison",
    description="Trading Bots with Telegram Integration",
    entry_points={
        "console_scripts": [
            "sol-monitor=src.spot_bots.sol_bot.telegram_monitor:main",
        ],
    },
)

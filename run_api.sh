#!/bin/bash
echo "Lancement du serveur API LiquidAI..."

if [ ! -f "venv/bin/activate" ]; then
    echo "Environnement virtuel non détecté. Lancez d'abord ./install.sh"
    exit 1
fi

# shellcheck disable=SC1091
source venv/bin/activate

python3 api.py

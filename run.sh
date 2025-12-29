#!/bin/bash
echo "Lancement de l'interface graphique LiquidAI..."

# Vérifier si l'environnement virtuel existe. Sinon, lancer l'installation.
if [ ! -f "venv/bin/activate" ]; then
    echo "Environnement virtuel non détecté. Lancement de l'installation..."
    chmod +x install.sh # Assurer que le script d'install est exécutable
    ./install.sh
    exit $?
fi

# Activer l'environnement virtuel
source venv/bin/activate

# Lancer l'application principale
python3 main.py

#!/bin/bash

echo "#################################################################"
echo "#                                                               #"
echo "#         Installateur pour l'application LiquidAI              #"
echo "#                                                               #"
echo "#################################################################"
echo

# Vérifier les prérequis
command -v python3 >/dev/null 2>&1 || { echo >&2 "Python 3 n'est pas installé. Abandon."; exit 1; }
command -v pip3 >/dev/null 2>&1 || { echo >&2 "pip3 n'est pas installé. Abandon."; exit 1; }
command -v git >/dev/null 2>&1 || { echo >&2 "Git n'est pas installé. Abandon."; exit 1; }

echo "Prérequis vérifiés."

# Créer et activer l'environnement virtuel
if [ ! -d "venv" ]; then
    echo "Création de l'environnement virtuel..."
    python3 -m venv venv
fi
source venv/bin/activate

# Mettre à jour pip et installer les dépendances
pip install --upgrade pip
echo "Installation des dépendances..."
pip install torch PyQt6 accelerate fastapi uvicorn[standard] markdown2
pip install "transformers @ git+https://github.com/huggingface/transformers.git@main"

echo
echo "#################################################################"
echo "#                                                               #"
echo "#            Installation terminée !                            #"
echo "#                                                               #"
echo "#################################################################"
echo
echo "Pour utiliser l'environnement, exécutez :"
echo "source venv/bin/activate"
echo
echo "Puis lancez l'application avec :"
echo "./run.sh"
echo

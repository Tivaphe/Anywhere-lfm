#!/bin/bash

echo "#################################################################"
echo "#                                                               #"
echo "#         Installateur pour l'application LiquidAI              #"
echo "#                                                               #"
echo "#################################################################"
echo

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 n'est pas installé. Abandon." >&2
    exit 1
fi

if ! command -v git >/dev/null 2>&1; then
    echo "Git n'est pas installé. Abandon." >&2
    exit 1
fi

echo "Prérequis vérifiés."

if [ ! -d "venv" ]; then
    echo "Création de l'environnement virtuel..."
    python3 -m venv venv || {
        echo "Impossible de créer l'environnement virtuel." >&2
        exit 1
    }
fi

# shellcheck disable=SC1091
source venv/bin/activate

echo "Mise à jour de pip..."
python3 -m pip install --upgrade pip

echo "Installation des dépendances depuis requirements.txt..."
echo "Cela peut prendre plusieurs minutes..."
python3 -m pip install -r requirements.txt || {
    echo "L'installation des dépendances a échoué." >&2
    exit 1
}

echo
echo "#################################################################"
echo "#                                                               #"
echo "#            Installation terminée !                            #"
echo "#                                                               #"
echo "#################################################################"
echo
echo "Pour relancer plus tard : ./run.sh   (GUI)  ou  ./run_api.sh  (API)"
echo

if [ "${SKIP_LAUNCH:-0}" != "1" ]; then
    chmod +x run.sh run_api.sh 2>/dev/null || true
    exec ./run.sh
fi

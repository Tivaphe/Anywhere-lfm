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

# Mettre à jour pip et installer les dépendances de base
pip install --upgrade pip
echo "Installation des dépendances de base (PyQt, Transformers, etc.)..."
pip install torch PyQt6 accelerate
pip install "transformers @ git+https://github.com/huggingface/transformers.git@main"

# --- Installation de Llama.cpp ---
echo
echo "Tentative d'installation de Llama.cpp (pour les modèles GGUF)..."

# Fonction pour gérer l'échec
handle_llama_failure() {
    echo
    echo "#################################################################"
    echo "#                                                               #"
    echo "#   AVERTISSEMENT : L'installation de Llama.cpp a échoué.       #"
    echo "#                                                               #"
    echo "#################################################################"
    echo
    echo "Cela signifie que vous ne pourrez PAS utiliser les modèles locaux .gguf."
    echo "L'application fonctionnera toujours avec les modèles Hugging Face ([HF])."
    echo
    echo "Pour résoudre ce problème, assurez-vous d'avoir un compilateur C++ installé."
    echo "- Sur Debian/Ubuntu: sudo apt-get install build-essential"
    echo "- Sur macOS: xcode-select --install"
    echo
}

# Tenter l'installation
pip install llama-cpp-python --no-cache-dir
if [ $? -ne 0 ]; then
    echo "L'installation simple a échoué. Tentative de compilation avec détection du matériel..."
    UNAME_S=$(uname -s)
    if [ "$UNAME_S" == "Darwin" ]; then
        echo "Système macOS détecté. Compilation avec le support Metal..."
        CMAKE_ARGS="-DLLAMA_METAL=on" pip install --force-reinstall --no-cache-dir llama-cpp-python || handle_llama_failure
    elif [ "$UNAME_S" == "Linux" ]; then
        if command -v nvidia-smi &> /dev/null; then
            echo "GPU NVIDIA détecté. Compilation avec le support CUDA..."
            CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install --force-reinstall --no-cache-dir llama-cpp-python || handle_llama_failure
        else
            echo "Pas de GPU NVIDIA détecté. Compilation pour CPU..."
            pip install --force-reinstall --no-cache-dir llama-cpp-python || handle_llama_failure
        fi
    else
        handle_llama_failure
    fi
fi

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

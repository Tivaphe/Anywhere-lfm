#!/bin/bash

echo "#################################################################"
echo "#                                                               #"
echo "#         Installateur pour l'application LiquidAI              #"
echo "#                                                               #"
echo "#################################################################"
echo

# Vérifier si Python 3 est installé
if ! command -v python3 &> /dev/null
then
    echo "Python 3 n'est pas installé. Veuillez l'installer."
    exit 1
fi

# Vérifier si pip est installé
if ! command -v pip3 &> /dev/null
then
    echo "pip3 n'est pas installé. Veuillez l'installer."
    exit 1
fi

# Vérifier si Git est installé
if ! command -v git &> /dev/null
then
    echo "Git n'est pas installé. Veuillez l'installer."
    exit 1
fi

echo "Prérequis vérifiés."

# Créer un environnement virtuel
if [ ! -d "venv" ]; then
    echo "Création de l'environnement virtuel..."
    python3 -m venv venv
fi

echo "Activation de l'environnement virtuel..."
source venv/bin/activate

# Mettre à jour pip
pip install --upgrade pip

# Installer les dépendances de base
echo "Installation des dépendances de base..."
pip install torch PyQt6 accelerate

echo "Installation de la dernière version de Transformers depuis GitHub..."
pip install "transformers @ git+https://github.com/huggingface/transformers.git@main"

# Installation de llama-cpp-python avec détection du matériel
echo "Installation de Llama.cpp..."
UNAME_S=$(uname -s)
if [ "$UNAME_S" == "Darwin" ]; then
    # macOS - compiler avec Metal (Apple Silicon)
    echo "Système macOS détecté. Compilation avec le support Metal..."
    CMAKE_ARGS="-DLLAMA_METAL=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
elif [ "$UNAME_S" == "Linux" ]; then
    # Linux - vérifier la présence de NVIDIA-SMI pour CUDA
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU NVIDIA détecté. Compilation avec le support CUDA..."
        CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
    else
        echo "Pas de GPU NVIDIA détecté. Compilation pour CPU..."
        pip install --force-reinstall --no-cache-dir llama-cpp-python
    fi
else
    echo "Système d'exploitation non pris en charge par ce script. Installation standard de Llama.cpp..."
    pip install --force-reinstall --no-cache-dir llama-cpp-python
fi


echo
echo "#################################################################"
echo "#                                                               #"
echo "#            Installation terminée avec succès!                 #"
echo "#                                                               #"
echo "#################################################################"
echo
echo "Pour utiliser l'environnement, exécutez :"
echo "source venv/bin/activate"
echo
echo "Puis lancez l'application avec :"
echo "./run.sh"
echo

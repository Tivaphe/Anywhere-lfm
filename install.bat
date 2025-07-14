@echo off
echo #################################################################
echo #                                                               #
echo #         Installateur pour l'application LiquidAI              #
echo #                                                               #
echo #################################################################
echo.

REM Vérifier si Python est installé
echo Vérification de l'installation de Python...
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo Python n'est pas installé ou n'est pas dans le PATH.
    echo Veuillez l'installer depuis https://www.python.org/downloads/
    echo Assurez-vous de cocher "Add Python to PATH" lors de l'installation.
    pause
    exit
)
echo Python est installé.

REM Vérifier si Git est installé
echo Vérification de l'installation de Git...
git --version >nul 2>nul
if %errorlevel% neq 0 (
    echo Git n'est pas installé ou n'est pas dans le PATH.
    echo Veuillez l'installer depuis https://git-scm.com/downloads
    pause
    exit
)
echo Git est installé.

REM Créer un environnement virtuel
if not exist venv (
    echo Création de l'environnement virtuel...
    python -m venv venv
)

echo Activation de l'environnement virtuel...
call venv\\Scripts\\activate

REM Mettre à jour pip
python -m pip install --upgrade pip

REM Installer les dépendances de base
echo Installation des dépendances de base (PyQt, Transformers, etc.)...
pip install torch PyQt6 accelerate
pip install "transformers @ git+https://github.com/huggingface/transformers.git@main"

REM --- Installation de Llama.cpp ---
echo.
echo Tentative d'installation de Llama.cpp (pour les modèles GGUF)...

pip install llama-cpp-python --no-cache-dir
if %errorlevel% neq 0 (
    echo.
    echo L'installation simple de llama-cpp-python a échoué. Tentative de compilation...
    set CMAKE_ARGS="-DLLAMA_CUBLAS=on"
    set FORCE_CMAKE=1
    pip install llama-cpp-python --no-cache-dir
    if %errorlevel% neq 0 (
        echo.
        echo #################################################################
        echo #                                                               #
        echo #   AVERTISSEMENT : L'installation de Llama.cpp a échoué.       #
        echo #                                                               #
        echo #################################################################
        echo.
        echo Cela signifie que vous ne pourrez PAS utiliser les modèles locaux .gguf.
        echo L'application fonctionnera toujours avec les modèles Hugging Face ([HF]).
        echo.
        echo Pour résoudre ce problème, vous devez installer les "Build Tools for Visual Studio":
        echo 1. Allez sur : https://visualstudio.microsoft.com/fr/downloads/
        echo 2. Trouvez "Outils pour Visual Studio" et téléchargez les "Build Tools".
        echo 3. Lors de l'installation, cochez la case "Développement desktop en C++".
        echo 4. Une fois l'installation terminée, relancez ce script install.bat.
        echo.
    )
)

echo.
echo #################################################################
echo #                                                               #
echo #            Installation terminée !                            #
echo #                                                               #
echo #################################################################
echo.
echo Vous pouvez maintenant lancer l'application en exécutant run.bat
echo.
pause

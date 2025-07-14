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

REM Installer les dépendances
echo Installation des dépendances...
pip install torch PyQt6 accelerate

echo Installation de la dernière version de Transformers depuis GitHub...
pip install git+https://github.com/huggingface/transformers.git@main

echo.
echo #################################################################
echo #                                                               #
echo #            Installation terminée avec succès!                 #
echo #                                                               #
echo #################################################################
echo.
echo Vous pouvez maintenant lancer l'application en exécutant run.bat
echo.
pause

@echo off
echo #################################################################
echo #                                                               #
echo #         Installateur pour l'application LiquidAI              #
echo #                                                               #
echo #################################################################
echo.

REM Vérifier les prérequis importants
echo Verification des prerequis...
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo Python n'est pas installé ou n'est pas dans le PATH.
    echo Veuillez l'installer depuis https://www.python.org/downloads/
    pause
    exit
)
git --version >nul 2>nul
if %errorlevel% neq 0 (
    echo Git n'est pas installe ou n'est pas dans le PATH.
    echo Veuillez l'installer depuis https://git-scm.com/downloads
    pause
    exit
)
echo Prerequis OK.

REM Creer un environnement virtuel
if not exist venv (
    echo Creation de l'environnement virtuel...
    python -m venv venv
)

echo Activation de l'environnement virtuel...
call venv\\Scripts\\activate

REM Creer le dossier pour les paquets locaux s'il n'existe pas
if not exist local_packages (
    echo Creation du dossier pour les paquets locaux...
    mkdir local_packages
)

REM Telecharger les paquets dans le dossier local
echo Telechargement des dependances dans le cache local...
pip download -r requirements.txt -d local_packages

REM Mettre à jour pip et installer les dependances depuis le cache local
echo Installation des dependances depuis le cache local...
pip install --upgrade pip
pip install --no-index --find-links=local_packages -r requirements.txt

echo.
echo #################################################################
echo #                                                               #
echo #            Installation terminee !                            #
echo #                                                               #
echo #################################################################
echo.
echo Lancement de l'application...
call run.bat

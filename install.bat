@echo off
echo #################################################################
echo #                                                               #
echo #         Installateur pour l'application LiquidAI              #
echo #                                                               #
echo #################################################################
echo.

REM Vérifier les prérequis importants
echo Vérification des prérequis...
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo Python n'est pas installé ou n'est pas dans le PATH.
    echo Veuillez l'installer depuis https://www.python.org/downloads/
    pause
    exit
)
git --version >nul 2>nul
if %errorlevel% neq 0 (
    echo Git n'est pas installé ou n'est pas dans le PATH.
    echo Veuillez l'installer depuis https://git-scm.com/downloads
    pause
    exit
)
echo Prérequis OK.

REM Créer un environnement virtuel
if not exist venv (
    echo Création de l'environnement virtuel...
    python -m venv venv
)

echo Activation de l'environnement virtuel...
call venv\\Scripts\\activate

REM Créer le dossier pour les paquets locaux s'il n'existe pas
if not exist local_packages (
    echo Création du dossier pour les paquets locaux...
    mkdir local_packages
)

REM Télécharger les paquets dans le dossier local
echo Téléchargement des dépendances dans le cache local...
pip download -r requirements.txt -d local_packages

REM Mettre à jour pip et installer les dépendances depuis le cache local
echo Installation des dépendances depuis le cache local...
pip install --upgrade pip
pip install --no-index --find-links=local_packages -r requirements.txt

echo.
echo #################################################################
echo #                                                               #
echo #            Installation terminée !                            #
echo #                                                               #
echo #################################################################
echo.
echo Lancement de l'application...
call run.bat

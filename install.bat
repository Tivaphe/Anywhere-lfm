@echo off
echo #################################################################
echo #                                                               #
echo #         Installateur pour l'application LiquidAI              #
echo #                                                               #
echo #################################################################
echo.

REM Verifier les prerequis importants
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

REM Mettre à jour pip et installer les dépendances
echo Installation des dépendances...
pip install --upgrade pip
pip install torch PyQt6 accelerate
pip install "transformers @ git+https://github.com/huggingface/transformers.git@main"

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

@echo off
echo #################################################################
echo #                                                               #
echo #         Installateur pour l'application LiquidAI              #
echo #         ========================================              #
echo #                                                               #
echo # Ce script va configurer l'environnement et installer          #
echo # toutes les dependances necessaires.                           #
echo #                                                               #
echo #################################################################
echo.
pause

REM Etape 1: Verifier les prerequis
echo [ETAPE 1/4] Verification de Python...
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo ERREUR: Python n'est pas installe ou n'est pas dans votre PATH.
    echo Veuillez l'installer (et cochez "Add to PATH") depuis:
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)
echo Python trouve.
echo.

REM Etape 2: Creer un environnement virtuel
echo [ETAPE 2/4] Configuration de l'environnement virtuel...
if not exist venv (
    echo Creation du dossier 'venv'...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERREUR: Impossible de creer l'environnement virtuel.
        pause
        exit /b 1
    )
) else (
    echo Le dossier 'venv' existe deja.
)
echo Environnement virtuel configure.
echo.

REM Etape 3: Activer l'environnement et installer les dependances
echo [ETAPE 3/4] Activation et installation des dependances...
call venv\\Scripts\\activate

echo Mise a jour de pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo ERREUR: Impossible de mettre a jour pip.
    pause
    exit /b 1
)

echo Installation des dependances depuis requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERREUR: L'installation des dependances a echoue.
    echo Verifiez les messages d'erreur ci-dessus.
    pause
    exit /b 1
)
echo Installation des dependances terminee.
echo.

REM Etape 4: Verifier l'installation
echo [ETAPE 4/4] Verification de l'installation de PyQt6...
pip show PyQt6 >nul 2>nul
if %errorlevel% neq 0 (
    echo ERREUR CRITIQUE: PyQt6 n'a pas pu etre installe correctement.
    echo L'application ne peut pas demarrer.
    pause
    exit /b 1
)
echo PyQt6 a ete installe avec succes.
echo.

echo #################################################################
echo #                                                               #
echo #            Installation terminee avec succes!                 #
echo #                                                               #
echo #################################################################
echo.
echo Vous pouvez maintenant lancer l'application en executant run.bat
echo.
pause

REM Lancement de l'application
call run.bat

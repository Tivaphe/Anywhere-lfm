@echo off
title Installateur LiquidAI

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
echo Lancement de l'installation...
echo.

REM Etape 1: Verifier les prerequis
echo [ETAPE 1/4] Verification de Python...
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo.
    echo ===============================================================
    echo = ERREUR: Python n'est pas installe ou n'est pas dans votre PATH.
    echo ===============================================================
    echo.
    echo Veuillez l'installer (et cochez "Add to PATH" pendant l'installation) depuis:
    echo https://www.python.org/downloads/
    echo.
    echo Le script va maintenant se fermer.
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
        echo.
        echo ===============================================================
        echo = ERREUR: Impossible de creer l'environnement virtuel.
        echo ===============================================================
        echo.
        echo Verifiez votre installation de Python.
        echo Le script va maintenant se fermer.
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
    echo.
    echo ===============================================================
    echo = ERREUR: Impossible de mettre a jour pip.
    echo ===============================================================
    echo.
    echo Le script va maintenant se fermer.
    pause
    exit /b 1
)

echo Installation des dependances depuis requirements.txt...
echo Cela peut prendre plusieurs minutes...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo ===============================================================
    echo = ERREUR: L'installation des dependances a echoue.
    echo ===============================================================
    echo.
    echo Verifiez les messages d'erreur ci-dessus.
    echo Le script va maintenant se fermer.
    pause
    exit /b 1
)
echo Installation des dependances terminee.
echo.

REM Etape 4: Verifier l'installation
echo [ETAPE 4/4] Verification de l'installation de PyQt6...
pip show PyQt6 >nul 2>nul
if %errorlevel% neq 0 (
    echo.
    echo ===============================================================
    echo = ERREUR CRITIQUE: PyQt6 n'a pas pu etre installe correctement.
    echo ===============================================================
    echo.
    echo L'application ne peut pas demarrer.
    echo Le script va maintenant se fermer.
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
echo Le script va maintenant tenter de lancer l'application pour vous.
pause

REM Lancement de l'application
call run.bat

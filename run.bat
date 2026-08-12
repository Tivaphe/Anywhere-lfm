@echo off
echo Lancement de l'interface graphique LiquidAI...

REM Vérifier si l'environnement virtuel existe. Sinon, lancer l'installation.
if not exist venv\\Scripts\\activate (
    echo Environnement virtuel non détecté. Lancement de l'installation...
    call install.bat
    exit /b %errorlevel%
)

REM Activer l'environnement virtuel
call venv\\Scripts\\activate

REM Lancer l'application principale
python main.py

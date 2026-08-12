@echo off
echo Lancement du serveur API LiquidAI...

if not exist venv\\Scripts\\activate (
    echo Environnement virtuel non detecte. Lancez d'abord install.bat
    pause
    exit /b 1
)

call venv\\Scripts\\activate

python api.py

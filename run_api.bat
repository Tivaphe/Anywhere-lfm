@echo off
echo Lancement du serveur API LiquidAI...

REM Activer l'environnement virtuel
call venv\\Scripts\\activate

REM Lancer le serveur API
python api.py

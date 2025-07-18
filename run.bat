@echo off
echo Lancement de l'application LiquidAI...
venv\Scripts\python.exe -m uvicorn api:app --host 0.0.0.0 --port 8000
pause

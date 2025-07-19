@echo off
echo Installation des dependances...

REM Verifier si Python est installe
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python n'est pas installe ou n'est pas dans le PATH.
    echo Veuillez installer Python 3 et l'ajouter a votre PATH.
    pause
    exit /b 1
)

REM Installer les dependances Python
pip install -r requirements.txt

echo Telechargement des modeles...
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium', cache_dir='./models'); AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium', cache_dir='./models')"

echo Installation terminee.
pause

@echo off
echo --- 1. Configurare Python ---
python -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt

echo.
echo --- 2. Configurare Frontend ---
cd muzica_UI
call npm install
cd ..

echo.
echo --- GATA! Totul este instalat. ---
pause
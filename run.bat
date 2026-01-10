@echo off
echo Pornesc aplicatia...

:: 1. Pornim Backend-ul intr-o fereastra noua
start "Backend Server" cmd /k "call .venv\Scripts\activate && python backend.py"

:: 2. Asteptam 3 secunde sa porneasca backend-ul
timeout /t 3 /nobreak >nul

:: 3. Pornim Frontend-ul intr-o fereastra noua
cd muzica_UI
start "Frontend Interface" cmd /k "npm run dev"

:: 4. Deschidem Browserul automat (asteptam putin sa incarce vite)
timeout /t 4 /nobreak >nul
start http://localhost:5173

echo Aplicatia ruleaza! Nu inchide ferestrele negre.
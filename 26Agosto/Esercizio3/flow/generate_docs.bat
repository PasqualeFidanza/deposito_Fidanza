@echo off
echo Generazione documentazione Flow CrewAI...
echo.

REM Verifica che siamo nella directory corretta
if not exist "docs\source\conf.py" (
    echo ERRORE: Esegui questo script dalla directory root del progetto flow
    pause
    exit /b 1
)

REM Genera la documentazione
echo Generando documentazione HTML...
sphinx-build -b html docs\source docs\build

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Documentazione generata con successo!
    echo.
    echo Per visualizzare la documentazione:
    echo 1. Apri docs\build\index.html nel browser
    echo 2. Oppure esegui: cd docs\build ^&^& python -m http.server 8000
    echo.
) else (
    echo.
    echo ❌ Errore durante la generazione della documentazione
    echo.
)

pause


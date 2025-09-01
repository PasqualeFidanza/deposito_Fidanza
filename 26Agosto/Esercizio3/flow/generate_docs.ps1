# Script PowerShell per generare la documentazione Flow CrewAI

Write-Host "Generazione documentazione Flow CrewAI..." -ForegroundColor Green
Write-Host ""

# Verifica che siamo nella directory corretta
if (-not (Test-Path "docs\source\conf.py")) {
    Write-Host "ERRORE: Esegui questo script dalla directory root del progetto flow" -ForegroundColor Red
    Read-Host "Premi Invio per uscire"
    exit 1
}

# Genera la documentazione
Write-Host "Generando documentazione HTML..." -ForegroundColor Yellow
$result = sphinx-build -b html docs\source docs\build

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Documentazione generata con successo!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Per visualizzare la documentazione:" -ForegroundColor Cyan
    Write-Host "1. Apri docs\build\index.html nel browser" -ForegroundColor White
    Write-Host "2. Oppure esegui: cd docs\build; python -m http.server 8000" -ForegroundColor White
    Write-Host ""
    
    # Chiedi se aprire la documentazione
    $open = Read-Host "Vuoi aprire la documentazione nel browser? (s/n)"
    if ($open -eq "s" -or $open -eq "S" -or $open -eq "si" -or $open -eq "SI") {
        Start-Process "docs\build\index.html"
    }
} else {
    Write-Host ""
    Write-Host "❌ Errore durante la generazione della documentazione" -ForegroundColor Red
    Write-Host ""
}

Read-Host "Premi Invio per uscire"

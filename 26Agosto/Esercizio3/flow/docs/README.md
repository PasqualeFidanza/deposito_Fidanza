# Documentazione Flow CrewAI

Questa directory contiene la documentazione completa del progetto Flow CrewAI generata con Sphinx.

## Struttura

- `source/` - File sorgente della documentazione (RST)
- `build/` - Documentazione HTML generata
- `qa_cani.md` - Documenti di esempio per il sistema RAG

## Come visualizzare la documentazione

### Opzione 1: Aprire direttamente il file HTML
Apri il file `build/index.html` nel tuo browser web preferito.

### Opzione 2: Server locale (raccomandato)
```bash
cd docs/build
python -m http.server 8000
```
Poi apri http://localhost:8000 nel browser.

## Come rigenerare la documentazione

Se modifichi il codice sorgente e vuoi aggiornare la documentazione:

```bash
# Dalla directory root del progetto
sphinx-build -b html docs/source docs/build
```

## Contenuto della documentazione

La documentazione include:

1. **Panoramica del progetto** - Descrizione generale e architettura
2. **Documentazione API completa** - Tutti i moduli, classi e metodi
3. **Docstring dettagliate** - Documentazione inline del codice
4. **Indici** - Indice dei moduli e ricerca

## Note

- La documentazione è in italiano
- Utilizza il tema Alabaster di Sphinx
- Include supporto per Google-style e NumPy-style docstrings
- Genera automaticamente la documentazione da tutti i moduli Python

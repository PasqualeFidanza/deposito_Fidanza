Flow CrewAI Documentation
=========================

Benvenuto nella documentazione del progetto Flow CrewAI! Questo progetto implementa un sistema di flusso intelligente che utilizza CrewAI per gestire domande tramite RAG (Retrieval-Augmented Generation) o ricerca web.

Panoramica del Progetto
-----------------------

Il progetto Flow CrewAI è un sistema che:

- **Analizza le domande** dell'utente per determinare se sono relative a un argomento specifico (cani)
- **Instrada automaticamente** le domande al sistema appropriato:
  - RAG per domande sui cani (utilizzando documenti locali)
  - Ricerca web per altre domande
- **Fornisce risposte** basate su documenti locali o ricerche web

Architettura
------------

Il sistema è composto da:

1. **Flow principale** (`main.py`): Gestisce il flusso di lavoro e il routing delle domande
2. **Crew RAG** (`rag_crew.py`): Gestisce le domande sui cani utilizzando documenti locali
3. **Crew Search** (`search_crew.py`): Gestisce le altre domande tramite ricerca web
4. **Tools**: Strumenti specializzati per RAG e ricerca web

.. toctree::
   :maxdepth: 2
   :caption: Documentazione API:

   modules


# Application Functionality

**Riferimenti normativi**: EU AI Act – Art. 11, Art. 13; Allegato IV (par. 1–3)

## Panoramica del funzionamento
Il sistema riceve una domanda dall’utente e la instrada tramite **CrewAI** verso l’agente più appropriato:
- **RAG Agent**: se la domanda è legata al dominio gestito dalla knowledge base interna (documenti indicizzati tramite embeddings).
- **Web Agent**: se la domanda è fuori dominio, viene effettuata una ricerca online tramite DuckDuckGo e la risposta è generata a partire dai risultati.

L’intero processo è orchestrato come un **flow CrewAI**, in cui lo stato della conversazione e il routing degli agenti vengono gestiti automaticamente.

## Input richiesti
- **Domanda utente** (testo libero).  
- Eventuali parametri di configurazione lato sviluppatore (es. tipo di modello Azure da usare, top-k documenti da recuperare, ecc.).

## Output prodotti
- **Risposta testuale** generata dal modello Azure OpenAI, arricchita dai documenti (RAG) o dalle ricerche Web.  
- Informazioni su quale agente ha risposto (RAG o Web), per garantire trasparenza.

## Capacità principali
- Routing automatico delle query in base al contesto/domino.  
- Recupero informazioni tramite due modalità complementari:  
  - **Knowledge interna (RAG)**: risposte basate su documenti e embedding.  
  - **Ricerca Web (DuckDuckGo)**: risposte aggiornate in tempo reale.  
- Supporto a modelli **Azure OpenAI** per generazione linguistica.  
- Architettura modulare e estendibile: nuovi agenti possono essere aggiunti al flow.

## Limiti e incertezze
- Le risposte del **Web Agent** dipendono dalla qualità delle fonti trovate: possono essere incomplete o non affidabili.  
- Il **RAG Agent** è limitato ai documenti disponibili nella knowledge base.  
- Non è garantita la **veridicità assoluta** delle risposte: il sistema deve essere inteso come supporto informativo, non come fonte autorevole unica.  

## Architettura funzionale
- **Flow Manager (CrewAI)**: gestisce lo stato e decide a quale agente instradare la query.  
- **RAG Agent**: combina retrieval da un vector store con modelli Azure per rispondere.  
- **Web Agent**: usa DuckDuckGo per cercare informazioni e modelli Azure per sintetizzare la risposta.  
- **LLM (Azure OpenAI)**: backbone linguistico utilizzato da entrambi gli agenti.



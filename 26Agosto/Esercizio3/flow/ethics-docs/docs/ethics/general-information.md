# General Information

## Purpose and Intended Use
Il progetto implementa un sistema conversazionale basato su **CrewAI**, in grado di ricevere una domanda dall’utente e indirizzarla automaticamente verso l’agente più appropriato:
- **RAG Agent** → se la domanda rientra nell’ambito del Retrieval-Augmented Generation, utilizzando conoscenza specifica e documenti indicizzati.
- **Web Agent** → se la domanda è fuori dal dominio del RAG, eseguendo ricerche in tempo reale tramite DuckDuckGo.

L’obiettivo principale è fornire un’infrastruttura modulare e scalabile per orchestrare agenti specializzati, semplificando l’integrazione di diverse fonti di conoscenza.  
Gli utenti previsti sono **sviluppatori e ricercatori** interessati a sistemi multi-agente e tecniche di routing in AI.  

Il sistema **non è destinato** a contesti sensibili (es. decisioni mediche, legali, finanziarie critiche) e non deve essere utilizzato per profilazione o raccolta di dati personali.

---

## Operational Environment
Il sistema è sviluppato in **Python** e si appoggia al framework **CrewAI** per la gestione dei flussi e degli agenti.  
Sono utilizzati **modelli di linguaggio deployati su Azure OpenAI** per l’elaborazione linguistica, garantendo stabilità e gestione delle chiavi/API attraverso variabili d’ambiente.  

Componenti principali:
- **CrewAI**: gestione del flow e routing delle richieste.
- **RAG pipeline**: recupero e utilizzo di conoscenza contestuale (embedding, vector store).
- **Web agent con DuckDuckGo**: recupero informazioni da fonti aperte.
- **Azure OpenAI models**: backend LLM per generazione e comprensione del linguaggio.

Il sistema può essere eseguito sia in **locale** che su **ambiente cloud**, a seconda delle necessità di sviluppo.

---

## Ethical Considerations
Il progetto integra alcune considerazioni etiche di base:
- **Trasparenza**: l’utente è consapevole di quale agente risponde (RAG o Web) e delle fonti usate.  
- **Privacy**: non vengono raccolti dati sensibili o identificativi; le query utente restano confinate al sistema senza persistenza.  
- **Affidabilità**: il Web agent utilizza DuckDuckGo come fonte primaria, quindi le informazioni potrebbero essere incomplete o non verificate; questo limite deve essere chiarito all’utente.  
- **Bias e Limitazioni**: i modelli Azure OpenAI possono riflettere bias intrinseci ai dati di addestramento; è responsabilità dello sviluppatore gestire correttamente l’interpretazione degli output.  
- **Uso previsto**: il sistema non deve essere applicato in scenari ad alto rischio (AI Act – sistemi critici), ma è concepito come strumento di ricerca e supporto informativo.

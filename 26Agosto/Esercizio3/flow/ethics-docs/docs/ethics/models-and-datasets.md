# Models & Datasets

**Riferimenti normativi**: EU AI Act – Art. 11; Allegato IV (par. 2d)

## Modelli

### LLM – Azure OpenAI `gpt-4o`
- **Ruolo**: modello linguistico principale per generazione e comprensione del linguaggio.  
- **Fornitore**: Microsoft Azure (OpenAI Service).  
- **Funzionalità**: interpretazione delle domande, generazione delle risposte, sintesi dei risultati recuperati.  
- **Rischi etici**: possibili bias nei dati di addestramento del modello; rischio di allucinazioni (informazioni inventate).  
- **Misure adottate**: uso limitato a scopi di supporto informativo; trasparenza verso l’utente sulla fonte della risposta (RAG o Web).

### Embeddings – `text-embedding-ada-002`
- **Ruolo**: creazione di rappresentazioni numeriche (embedding) delle frasi per il retrieval semantico.  
- **Fornitore**: OpenAI (via Azure).  
- **Funzionalità**: confronto semantico tra query utente e documenti della knowledge base.  
- **Limiti**: copertura linguistica principalmente in inglese; performance variabile su domini non visti.  

---

## Dataset (Knowledge Base)

### Documenti RAG
- **Contenuto**: documenti di dominio “cani” (informazioni generali, cura, comportamento, addestramento).  
- **Fonte**: testi generati con ChatGPT e adattati per simulare una base di conoscenza tematica.  
- **Scopo**: permettere al sistema di rispondere a domande specifiche su cani.  
- **Qualità e limiti**: i documenti sono sintetici, non derivati da dataset ufficiali; rischio di incompletezza o semplificazioni.  
- **Etica**: i documenti non contengono dati sensibili né riferimenti a persone reali; non presentano rischi per la privacy.  

---

## Considerazioni etiche
- I modelli Azure OpenAI (`gpt-4o` e `ada-002`) possono riflettere bias intrinseci nei dati di addestramento → l’uso è circoscritto a scenari di ricerca/didattici.  
- La knowledge base (dominio cani) non presenta rischi sensibili, ma va chiarito che non sostituisce **fonti scientifiche o veterinarie autorevoli**.  
- L’utente deve essere informato che le risposte RAG sono basate su documenti sintetici, mentre le risposte Web possono attingere a fonti non verificate.

# Incident Management

**Riferimenti normativi**: EU AI Act – Art. 9(2), Art. 15(5)

## Tipologie di incidenti previsti
Durante l’uso del sistema possono verificarsi i seguenti incidenti:
- **Errori di routing** → una domanda viene indirizzata all’agente sbagliato (es. query RAG inviata al Web agent).  
- **Risposte non accurate** → il modello `gpt-4o` genera output scorretto o poco coerente.  
- **Informazioni non verificate** → l’agente Web restituisce contenuti da fonti non attendibili (DuckDuckGo).  
- **Errori infrastrutturali** → mancata risposta da Azure OpenAI o da DuckDuckGo per problemi di rete/API.  
- **Gestione credenziali** → configurazioni errate o chiavi API non valide.  

---

## Runbook di gestione
Per ciascun incidente è prevista una procedura di risposta:

1. **Rilevazione**
   - L’utente segnala un comportamento anomalo (risposta errata, assenza di output).  
   - I log locali segnalano errori di connessione o eccezioni Python.  

2. **Notifica**
   - In ambiente di sviluppo, gli errori vengono mostrati a console.  
   - Non essendo un sistema in produzione, non sono previsti canali automatici di notifica.  

3. **Contenimento**
   - Routing manuale delle query (ripetizione della domanda o forzatura verso l’agente corretto).  
   - Blocco temporaneo dell’esecuzione in caso di errori ripetuti.  

4. **Post-mortem**
   - Analisi dell’errore e individuazione della causa (es. chiavi API scadute, logica CrewAI incompleta, limiti del modello).  
   - Aggiornamento della knowledge base o correzione del codice di routing.  

5. **Azioni correttive**
   - Aggiornamento del flow CrewAI.  
   - Rigenerazione dei documenti RAG se troppo limitati.  
   - Revisione della gestione delle chiavi `.env`.  

---

## Monitoring e logging
- **Console logs** → forniscono informazioni sugli errori durante l’esecuzione.  
- **Messaggi di fallback** → l’utente viene informato quando non è possibile fornire una risposta.  
- **Manual review** → gli sviluppatori possono controllare l’output delle sessioni di test per identificare anomalie.  

---

## Disaster Recovery (DR)
Essendo un sistema sperimentale, non è richiesto un DR formale.  
Le misure minime adottate includono:
- Versionamento del codice su GitHub.  
- Rigenerazione dei documenti RAG a partire da ChatGPT.  
- Possibilità di ripristino rapido reinstallando le dipendenze Python e riconfigurando le chiavi API.  

---

## Conclusione
L’incident management è gestito in modo proporzionato alla natura del progetto (AI a rischio minimo/limitato).  
Gli incidenti più frequenti (routing errato, fonti non verificate, errori API) vengono mitigati tramite supervisione umana, logging e correzioni incrementali.

# Deployment

**Riferimenti normativi**: EU AI Act – Allegato IV (par. 2–3)

## Infrastruttura e ambiente di esecuzione
- **Ambiente di sviluppo**: Python 3.12 con ambiente virtuale dedicato.  
- **Framework principale**: CrewAI per orchestrazione di agenti e gestione dei flow.  
- **LLM provider**: Microsoft Azure OpenAI (endpoint e chiavi configurati via variabili d’ambiente).  
- **Agenti**:
  - **RAG Agent**: utilizza il modello `text-embedding-ada-002` per il retrieval e `gpt-4o` per la generazione.  
  - **Web Agent**: integra ricerche esterne tramite DuckDuckGo API e genera risposte con `gpt-4o`.  

Il sistema può essere eseguito:
- **In locale**, su ambiente di sviluppo (VS Code, terminale).  
- **In cloud**, sfruttando Azure come backend per modelli e potenzialmente container/Docker per distribuire CrewAI.  

---

## Integrazioni esterne
- **Azure OpenAI**: accesso tramite chiavi API (gestite in `.env`, non versionate).  
- **DuckDuckGo Search API**: usata per la raccolta di informazioni real-time.  

---

## Deployment plan
- **Configurazione**: 
  - Variabili d’ambiente per endpoint e chiavi Azure (`AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION`).  
  - Configurazione agenti CrewAI tramite file YAML e Python.  
- **Scalabilità**: architettura modulare → possibile aggiungere nuovi agenti o nuove fonti senza modificare l’intero sistema.  
- **Backup e recovery**: i documenti RAG possono essere rigenerati facilmente; le configurazioni sono versionate in Git.  
- **Rollback**: eventuali modifiche ai flow CrewAI possono essere ripristinate tramite versioning del codice.  

---

## Sicurezza e gestione delle credenziali
- Le chiavi API non vengono salvate nel codice → uso di file `.env` o variabili d’ambiente.  
- Le connessioni a Azure OpenAI sono su HTTPS.  
- Accesso limitato allo sviluppatore (nessun multi-tenant).  

---

## Limitazioni del deployment
- Non è prevista una distribuzione **pubblica di produzione**: il progetto è a scopo di ricerca e sviluppo.  
- Le risorse cloud (Azure) sono utilizzate solo come backend per i modelli LLM/embedding, non per storage o gestione utenti.  

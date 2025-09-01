# Testing & Validation (Accuracy, Robustness, Cybersecurity)

**Riferimenti normativi**: EU AI Act – Art. 15

## Accuracy
L’accuratezza del sistema è stata valutata tramite **verifiche manuali**:
- È stato creato un set di domande di test, differenziate tra quelle appartenenti al dominio del RAG (documenti sui cani) e quelle generiche da inviare al Web agent.
- Per ciascuna domanda si è verificata la correttezza:
  - Del **routing** (la domanda è stata instradata all’agente corretto).  
  - Della **risposta** (coerenza con i documenti RAG o con i risultati del Web search).  

Questa metodologia ha permesso di valutare l’efficacia del sistema pur senza test automatizzati.  
> *Nota*: per applicazioni più estese, si potrebbero introdurre test quantitativi (es. metriche di precision/recall sul routing, valutazioni qualitative con utenti).  

---

## Robustness
La robustezza del sistema è stata verificata testando:
- **Domande fuori dominio**: il sistema ha correttamente indirizzato query non pertinenti al RAG verso il Web agent.  
- **Domande borderline**: in alcuni casi si è osservata incertezza nel routing, evidenziando la necessità di una logica più raffinata.  
- **Errori o assenza di fonti**: quando DuckDuckGo non restituiva risultati rilevanti, il sistema comunicava limiti all’utente.  

Non sono stati effettuati test adversariali (es. input malevoli), data la natura sperimentale del progetto.

---

## Cybersecurity
- Le chiavi API Azure sono gestite tramite file `.env` e **non** vengono versionate nel repository.  
- Le connessioni con Azure OpenAI e DuckDuckGo avvengono tramite **HTTPS**.  
- L’accesso al sistema è limitato all’ambiente di sviluppo; non esiste un deployment pubblico esposto a minacce esterne.  

---

## Conclusione
Le attività di testing hanno confermato che:
- Il routing tra RAG e Web agent funziona nella maggior parte dei casi.  
- Le risposte del RAG sono coerenti con i documenti forniti.  
- Le risposte del Web agent dipendono dalla qualità delle fonti esterne, con margini di incertezza.  

Il sistema, pur basandosi su verifiche manuali, soddisfa i requisiti di **trasparenza** e **robustezza proporzionata al rischio minimo/limitato**.

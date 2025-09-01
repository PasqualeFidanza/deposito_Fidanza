# Lifecycle & Risk Management

**Riferimenti normativi**: EU AI Act – Art. 9, Art. 11; Allegato IV (par. 6)

## Gestione del ciclo di vita
Il sistema segue un ciclo di vita tipico di un progetto AI sperimentale:
1. **Sviluppo iniziale** → definizione del flow CrewAI, configurazione agenti e integrazione con Azure OpenAI.  
2. **Testing locale** → esecuzione in ambiente di sviluppo con domande di esempio e validazione del routing (RAG vs Web).  
3. **Validazione funzionale** → verifica che le risposte siano coerenti con la query e con la logica di routing.  
4. **Iterazioni di miglioramento** → possibilità di aggiornare i documenti RAG o sostituire/aggiungere agenti.  
5. **Manutenzione** → aggiornamento librerie Python, chiavi API e modelli Azure (se cambiano versione).  

---

## Versioning e Change Management
- **Controllo versione**: il codice e la documentazione sono gestiti con GitHub.  
- **Flow CrewAI**: eventuali modifiche al routing vengono tracciate tramite commit e branch dedicati.  
- **Dataset RAG**: i documenti (dominio “cani”) possono essere aggiornati o ampliati in nuove versioni, mantenendo la tracciabilità.  
- **Configurazioni**: endpoint e chiavi Azure vengono gestiti tramite variabili d’ambiente e non inclusi nel versioning.  

---

## Monitoraggio e metriche
- **Funzionalità**: correttezza del routing (percentuale di query classificate correttamente come RAG o Web).  
- **Qualità risposte**: accuratezza percepita delle risposte, valutata con test manuali.  
- **Affidabilità infrastruttura**: monitoraggio degli errori di rete/API con Azure e DuckDuckGo.  

---

## Risk Management System
Il sistema non rientra tra gli **AI ad alto rischio** (EU AI Act), ma viene adottato un approccio di gestione dei rischi per garantire trasparenza e affidabilità.  

### Rischi identificati
- **Bias e allucinazioni** del modello `gpt-4o`.  
- **Fonti Web non verificate** con possibile disinformazione.  
- **Errori di routing** che indirizzano una query all’agente sbagliato.  
- **Dipendenza da fornitori esterni** (Azure, DuckDuckGo).  

### Misure di mitigazione
- **Trasparenza**: l’utente viene informato su quale agente risponde (RAG o Web).  
- **Limitazione d’uso**: il sistema è destinato a ricerca e sviluppo, non a decisioni critiche.  
- **Aggiornamenti periodici**: librerie e modelli monitorati per compatibilità e sicurezza.  
- **Fallback**: in caso di errore di rete/API, il sistema restituisce un messaggio chiaro all’utente.  

---

## Conclusione
La gestione del ciclo di vita è basata su **iterazioni continue** e **versioning controllato**.  
Il Risk Management è proporzionato al livello di rischio minimo/limitato, con focus su trasparenza, robustezza e controllo umano.

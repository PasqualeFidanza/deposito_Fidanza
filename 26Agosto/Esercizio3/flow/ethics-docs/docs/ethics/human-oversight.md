# Human Oversight

**Riferimenti normativi**: EU AI Act – Art. 11(2e), Art. 14

## Meccanismi di supervisione
Il sistema è progettato per essere **assistito e supervisionato da un operatore umano** in tutte le fasi:
- L’utente **inserisce la domanda** e conosce quale agente risponde (RAG o Web).  
- L’utente può **valutare la qualità e l’attendibilità** della risposta.  
- In caso di errore (routing sbagliato, informazione incompleta), l’utente può **interrompere l’uso o correggere manualmente** la query.  

## HITL (Human-in-the-Loop)
Il sistema non prende decisioni autonome critiche:  
- Ogni output viene mostrato all’utente prima di essere utilizzato.  
- L’utente può riformulare la domanda o decidere di non accettare la risposta.  
- Non esistono azioni automatizzate che incidano su diritti o attività sensibili.  

## HOTL (Human-on-the-Loop)
Lo sviluppatore rimane responsabile del monitoraggio:  
- Può aggiornare la knowledge base del RAG (documenti sui cani).  
- Può modificare la logica di routing in CrewAI.  
- Può sostituire i modelli o le fonti in caso di problemi di affidabilità.  

## Limitazioni
- Il sistema **non è pensato per uso autonomo senza supervisione umana**.  
- Le risposte del Web agent dipendono da fonti esterne (DuckDuckGo) e devono essere valutate criticamente dall’utente.  

## Conclusione
La supervisione umana è garantita sia lato **utente finale** (valutazione diretta delle risposte), sia lato **sviluppatore** (controllo e aggiornamento continuo del sistema).  
Questo riduce i rischi e conferma che l’applicazione appartiene alla categoria di **AI a rischio minimo/limitato**.

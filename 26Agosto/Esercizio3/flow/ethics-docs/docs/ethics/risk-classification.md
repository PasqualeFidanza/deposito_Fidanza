# Risk Classification

**Riferimenti normativi**: EU AI Act – Articoli 5–7, Allegato III

## Livello di rischio
Il sistema non rientra tra quelli classificati come **Prohibited AI** (Art. 5) né come **High-risk AI** (Allegato III).  
È quindi da considerarsi come **AI a rischio limitato o minimo**, in quanto:
- Non prende decisioni automatizzate con impatti diretti su diritti fondamentali (es. credito, giustizia, sanità).
- È progettato per scopi di **ricerca, sviluppo e supporto informativo**.
- Le interazioni con l’utente non comportano conseguenze legali, professionali o mediche.

## Motivazione della classificazione
- **Minimal risk**: il sistema si comporta come un assistente che fornisce risposte testuali, analogamente a chatbot generici.  
- **Limited risk**: l’unico elemento che può influenzare l’affidabilità è la componente Web (DuckDuckGo), che può restituire informazioni non sempre verificate.  
- **No High-risk**: non opera in contesti critici (infrastrutture, istruzione, occupazione, biometria, polizia, giustizia).  

## Misure di mitigazione
- L’utente viene informato su quale agente risponde (RAG o Web) e sulla natura delle fonti.  
- È dichiarato esplicitamente che il sistema **non deve essere usato in contesti sensibili o decisionali critici**.  
- Si raccomanda un uso **responsabile e supervisionato** da parte degli sviluppatori.  

## Conclusione
Il sistema può essere correttamente classificato come **AI a rischio minimo/limitato** ai sensi dell’AI Act.  
Il rischio principale riguarda la **qualità delle informazioni** fornite dall’agente Web, mitigato da trasparenza e supervisione umana.

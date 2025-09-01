"""
Modulo principale per il sistema Flow CrewAI.

Questo modulo implementa un flusso di lavoro intelligente che analizza le domande
dell'utente e le instrada automaticamente al sistema appropriato (RAG o ricerca web)
basandosi sul contenuto della domanda.
"""

from pydantic import BaseModel
from crewai.flow import Flow, listen, start, router
from crewai import LLM
from flow.crews.rag_crew.rag_crew import RagCrew
from flow.crews.search_crew.search_crew import SearchCrew


class FlowState(BaseModel):
    """
    Stato del flusso di lavoro che mantiene le informazioni durante l'esecuzione.
    
    Attributes:
        question (str): La domanda inserita dall'utente
        route (str): Il percorso scelto per la domanda ('rag' o 'search')
        rag_topic (str): L'argomento per cui il sistema RAG è configurato (default: 'dogs')
        answer (str): La risposta generata dal sistema
    """
    question: str = ''
    route: str = ''
    rag_topic: str = 'dogs'
    answer: str = ''


class RagOrSearchFlow(Flow[FlowState]):
    """
    Flusso di lavoro principale che gestisce il routing intelligente delle domande.
    
    Questo flusso implementa un sistema che:
    1. Raccoglie la domanda dall'utente
    2. Analizza la domanda per determinare se è relativa all'argomento RAG
    3. Instrada la domanda al sistema appropriato (RAG o ricerca web)
    4. Restituisce la risposta generata
    """

    @start()
    def generate_question(self):
        """
        Punto di partenza del flusso che raccoglie la domanda dall'utente.
        
        Continua a richiedere input finché l'utente non inserisce una domanda valida
        (non vuota).
        """
        while True:
            self.state.question = input('Inserisci la domanda: ')
            if self.state.question.strip():
                break
            else:
                print('Inserisci una domanda valida!')


    @router(generate_question)
    def route_flow(self):
        """
        Router che determina se la domanda deve essere gestita dal sistema RAG o ricerca web.
        
        Utilizza un LLM per analizzare la domanda e determinare se è relativa
        all'argomento configurato per il sistema RAG (default: 'dogs').
        
        Returns:
            str: 'rag' se la domanda è relativa all'argomento RAG, 'search' altrimenti
        """
        llm = LLM(model='azure/gpt-4o')
        messages = [
            {
                "role": "system",
                "content": (
                    f"Analyze the question '{self.state.question}'; "
                    f"if it is related to '{self.state.rag_topic}' return 'true', "
                    f"otherwise return 'false'. Don't give any other explanation."
                )
            }
        ]

        response = llm.call(messages=messages)
        if response is not None:
            response = response.strip().lower()
        else:
            response = ''
        print("Routing decision:", response)
        if response == 'true':
            return 'rag'
        else:
            return 'search'


    @listen('rag')
    def handle_rag(self):
        """
        Gestisce le domande che sono state instradate al sistema RAG.
        
        Crea un'istanza del RagCrew e lo esegue con la domanda dell'utente.
        Il risultato viene salvato nello stato del flusso.
        
        Returns:
            FlowState: Lo stato del flusso aggiornato con la risposta
        """
        crew = RagCrew().crew()
        self.state.answer = crew.kickoff(inputs={
            'question': self.state.question
        })
        return self.state


    @listen('search')
    def handle_search(self):
        """
        Gestisce le domande che sono state instradate al sistema di ricerca web.
        
        Crea un'istanza del SearchCrew e lo esegue con la domanda dell'utente.
        Il risultato viene salvato nello stato del flusso.
        
        Returns:
            FlowState: Lo stato del flusso aggiornato con la risposta
        """
        crew = SearchCrew().crew()
        self.state.answer = crew.kickoff(inputs={
            'question': self.state.question
        })
        return self.state


def kickoff():
    """
    Funzione principale per avviare il flusso di lavoro.
    
    Crea un'istanza del RagOrSearchFlow e lo avvia.
    """
    flow = RagOrSearchFlow()
    flow.kickoff()


def plot():
    """
    Funzione per visualizzare il grafo del flusso di lavoro.
    
    Crea un'istanza del RagOrSearchFlow e genera una visualizzazione
    del grafo del flusso.
    """
    flow = RagOrSearchFlow()
    flow.plot()


if __name__ == "__main__":
    kickoff()
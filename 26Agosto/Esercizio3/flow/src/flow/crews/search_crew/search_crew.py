"""
Modulo per il SearchCrew - Crew specializzato per la ricerca web.

Questo modulo implementa un crew CrewAI che utilizza il DDGSearch tool per
rispondere alle domande effettuando ricerche web in tempo reale.
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from src.flow.tools.search_tool import DDGSearch

@CrewBase
class SearchCrew():
    """
    Crew specializzato per la ricerca web.
    
    Questo crew è progettato per gestire domande che non sono coperte dal sistema RAG,
    utilizzando ricerche web in tempo reale per fornire informazioni aggiornate
    e accurate.
    
    Attributes:
        agents (List[BaseAgent]): Lista degli agenti del crew
        tasks (List[Task]): Lista dei task del crew
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def searcher(self) -> Agent:
        """
        Crea e configura l'agente di ricerca web.
        
        L'agente è configurato per utilizzare il DDGSearch tool e rispondere alle domande
        effettuando ricerche web in tempo reale.
        
        Returns:
            Agent: L'agente di ricerca web configurato
        """
        return Agent(
            config=self.agents_config['searcher'],
            verbose=True,
            tools=[DDGSearch()]
        )

    @task
    def search_task(self) -> Task:
        """
        Crea e configura il task di ricerca web.
        
        Il task definisce l'obiettivo e le istruzioni per l'agente di ricerca
        per elaborare le domande dell'utente utilizzando ricerche web.
        
        Returns:
            Task: Il task di ricerca web configurato
        """
        return Task(
            config=self.tasks_config['search_task'],
        )


    @crew
    def crew(self) -> Crew:
        """
        Crea e configura il crew di ricerca web completo.
        
        Combina l'agente e il task di ricerca in un crew che esegue i task
        in modo sequenziale per fornire risposte basate su ricerche web.
        
        Returns:
            Crew: Il crew di ricerca web configurato e pronto per l'esecuzione
        """
        return Crew(
            agents=self.agents, 
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True,
        )

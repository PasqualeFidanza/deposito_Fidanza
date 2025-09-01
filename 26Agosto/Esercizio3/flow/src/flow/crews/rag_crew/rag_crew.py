"""
Modulo per il RagCrew - Crew specializzato per il Retrieval-Augmented Generation.

Questo modulo implementa un crew CrewAI che utilizza il RagTool per rispondere
alle domande basandosi su documenti locali utilizzando tecniche di RAG.
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from src.flow.tools.rag_tool import RagTool


@CrewBase
class RagCrew():
    """
    Crew specializzato per il Retrieval-Augmented Generation (RAG).
    
    Questo crew è progettato per gestire domande relative a un argomento specifico
    (configurato come 'dogs' di default) utilizzando documenti locali e tecniche
    di RAG per fornire risposte accurate e contestualizzate.
    
    Attributes:
        agents (List[BaseAgent]): Lista degli agenti del crew
        tasks (List[Task]): Lista dei task del crew
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def rag_agent(self) -> Agent:
        """
        Crea e configura l'agente RAG.
        
        L'agente è configurato per utilizzare il RagTool e rispondere alle domande
        basandosi sui documenti locali disponibili.
        
        Returns:
            Agent: L'agente RAG configurato
        """
        return Agent(
            config=self.agents_config['rag_agent'], 
            verbose=True,
            tools=[RagTool()]
        )

    @task
    def rag_task(self) -> Task:
        """
        Crea e configura il task RAG.
        
        Il task definisce l'obiettivo e le istruzioni per l'agente RAG
        per elaborare le domande dell'utente.
        
        Returns:
            Task: Il task RAG configurato
        """
        return Task(
            config=self.tasks_config['rag_task'], 
        )
    
    @crew
    def crew(self) -> Crew:
        """
        Crea e configura il crew RAG completo.
        
        Combina l'agente e il task RAG in un crew che esegue i task
        in modo sequenziale per fornire risposte basate sui documenti locali.
        
        Returns:
            Crew: Il crew RAG configurato e pronto per l'esecuzione
        """
        return Crew(
            agents=self.agents, 
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True,
        )

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from src.flow.tools.rag_tool import RagTool



@CrewBase
class RagCrew():
    """RagCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def rag_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['rag_agent'], 
            verbose=True,
            tools=[RagTool()]
        )

    @task
    def rag_task(self) -> Task:
        return Task(
            config=self.tasks_config['rag_task'], 
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the RagCrew crew"""


        return Crew(
            agents=self.agents, 
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True,
        )

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from src.flow.tools.docs_tool import RagToolSphinx


@CrewBase
class DocsCrew():
    """DocsCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def docs_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['docs_agent'],
            verbose=True,
            tools = [RagToolSphinx()]
        )


    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'], 
        )


    @crew
    def crew(self) -> Crew:
        """Creates the DocsCrew crew"""


        return Crew(
            agents=self.agents, 
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True,
        )

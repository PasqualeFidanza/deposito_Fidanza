from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from src.flow.tools.search_tool import DDGSearch

@CrewBase
class SearchCrew():
    """SearchCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def searcher(self) -> Agent:
        return Agent(
            config=self.agents_config['searcher'],
            verbose=True,
            tools=[DDGSearch()]
        )

    @task
    def search_task(self) -> Task:
        return Task(
            config=self.tasks_config['search_task'],
        )


    @crew
    def crew(self) -> Crew:
        """Creates the SearchCrew crew"""

        return Crew(
            agents=self.agents, 
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True,
        )

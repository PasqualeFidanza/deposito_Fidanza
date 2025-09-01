from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List


@CrewBase
class MathCrew():
    """MathCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'], 
            verbose=True
        )

    @agent
    def math_solver(self) -> Agent:
        return Agent(
            config=self.agents_config['math_solver'], 
            verbose=True
        )


    @task
    def math_task(self) -> Task:
        return Task(
            config=self.tasks_config['math_task'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the MathCrew crew"""

        return Crew(
            agents=self.agents, 
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True,
        )

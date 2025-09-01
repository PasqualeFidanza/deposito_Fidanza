from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai import LLM
from crewai_tools import ScrapeWebsiteTool
from crewai.tools import tool
from src.webscraping_flow.tools.custom_tool import DDGSearch

# @tool('Search Tool')
# def search_tool(query: str) -> list:
#     """Search Internet for relevant information based on a query."""
#     ddgs = DDGS()
#     results = ddgs.text(query=query, region='wt-wt', safesearch='moderate', max_results=5)
#     return results


scrape = ScrapeWebsiteTool()

@CrewBase
class WebscrapingCrew:
    """WebscrapingCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True,
            tools=[DDGSearch()]
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'], # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the WebscrapingCrew crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from flow.tools.docs_tool import RagToolSphinx
from flow.tools.ask_tool import AskUserTool
from crewai_tools import SerperScrapeWebsiteTool

link = "https://aloosley.github.io/techops/template-application-documentation/#general-information"


@CrewBase
class DocsCrew():
    """DocsCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]
    
    @agent
    def web_search_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['web_search_agent'],
            verbose=True,
            tools=[SerperScrapeWebsiteTool()]
        )
    
    @agent
    def docs_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['docs_agent'],
            verbose=True,
            tools = [RagToolSphinx()]
        )
    
    @task
    def extract_template_fields(self) -> Task:
        return Task(
            config=self.tasks_config['extract_template_fields'],
            agent=self.web_search_agent()
        )

    @task
    def generate_ethics_doc(self) -> Task:
        return Task(
            config=self.tasks_config['generate_ethics_doc'],
            agent=self.docs_agent()
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

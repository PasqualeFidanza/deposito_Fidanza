from crewai.flow import Flow, listen, start
from pydantic import BaseModel

from webscraping_flow.crews.webscraping_crew.webscraping_crew import WebscrapingCrew



class WebscrapingState(BaseModel):
    topic: str = ''
    summary: str = ''


class WebscrapingFlow(Flow[WebscrapingState]):

    @start()
    def get_user_input(self):
        self.state.topic = input("Inserisci l'argomento: ")
        return self.state
    

    @listen(get_user_input)
    def scrape_web(self):
        print("\n--- Scraping the web... ---\n")
        topic = self.state.topic
        crew = WebscrapingCrew().crew()
        result = crew.kickoff(inputs={'topic': topic})
        self.state.summary = str(result)
        print(self.state.summary)
        return self.state.summary
        

def kickoff():
    flow = WebscrapingFlow()
    flow.kickoff()

def plot():
    flow = WebscrapingFlow()
    flow.plot("Webscraping Flow diagram")

if __name__ == "__main__":
    kickoff()
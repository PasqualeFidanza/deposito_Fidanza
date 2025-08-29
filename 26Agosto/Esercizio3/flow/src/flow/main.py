from pydantic import BaseModel

from crewai.flow import Flow, listen, start, router
from crewai import LLM
from flow.crews.rag_crew.rag_crew import RagCrew
from flow.crews.search_crew.search_crew import SearchCrew


class FlowState(BaseModel):
    question: str = ''
    route: str = ''
    rag_topic: str = 'dogs'
    answer: str = ''


class RagOrSearchFlow(Flow[FlowState]):

    @start()
    def generate_question(self):
        while True:
            self.state.question = input('Inserisci la domanda: ')
            if self.state.question.strip():
                break
            else:
                print('Inserisci una domanda valida!')

    @router(generate_question)
    def route_flow(self):
        llm = LLM(model='azure/gpt-4o-mini')
        messages = [
            {
                "role": "system",
                "content": (
                    f"Analyze the question '{self.state.question}'; "
                    f"if it is related to '{self.state.rag_topic}' return 'true', "
                    f"otherwise return 'false'. Don't give any explanation."
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
        crew = RagCrew().crew()
        self.state.answer = crew.kickoff(inputs={
            'question': self.state.question.strip()
        })
        print("RAG answer:", self.state.answer)
        return self.state

    @listen('search')
    def handle_search(self):
        crew = SearchCrew().crew()
        self.state.answer = crew.kickoff(inputs={
            'question': self.state.question.strip()
        })
        print("Search answer:", self.state.answer)
        return self.state


def kickoff():
    flow = RagOrSearchFlow()
    flow.kickoff()


def plot():
    flow = RagOrSearchFlow()
    flow.plot()


if __name__ == "__main__":
    kickoff()
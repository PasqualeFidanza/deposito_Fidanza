#!/usr/bin/env python
from random import randint

from pydantic import BaseModel

from crewai.flow import Flow, listen, start, router

from routing.crews.search_crew.search_crew import SearchCrew
from routing.crews.add_crew.add_crew import AddCrew

class RoutingState(BaseModel):
    choice: str = ''
    topic: str = ''
    summary: str = ''
    first_number: str = ''
    second_number: str = ''
    result: int = None

class RoutingFlow(Flow[RoutingState]):

    @start()
    def generate_choice(self):
        while True:
            self.state.choice = input("Choose: \n\t 'SEARCH' to summarize from websites \n\t 'ADD' to sum 2 numbers")
            if self.state.choice.lower() in ['search', 'add']:
                break
            print('Choice not valid, try again!')

    @router(generate_choice)
    def routing_method(self):
        if self.state.choice.lower() == 'search':
            return 'search'
        else:
            return 'add'
        
    @listen('search')
    def searcher(self):
        self.state.topic = input('Inserisci un argomento valido: ')
        crew = SearchCrew().crew()
        result = crew.kickoff(inputs={'topic': self.state.topic})
        self.state.summary = str(result)
        print(self.state.summary)
        return self.state

    @listen('add')
    def adder(self):
        self.state.first_number = input('Inserisci il primo numero: ')
        self.state.second_number = input('Inserisci il secondo numero: ')
        crew = AddCrew().crew()
        self.state.result = self.state.result = crew.kickoff(inputs = {
            'first_number': self.state.first_number,
            'second_number': self.state.second_number
        })
        print(self.state.result)
        return self.state


def kickoff():
    routing_flow = RoutingFlow()
    routing_flow.kickoff()


def plot():
    routing_flow = RoutingFlow()
    routing_flow.plot()


if __name__ == "__main__":
    kickoff()

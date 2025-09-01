from typing import Type
from ddgs import DDGS
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class DDGSearch(BaseTool):
    name: str = 'duckduckgo'
    description: str = 'Uno strumento per ricercare risultati sul web'

    def _run(self, query: str) -> str:
        if not query:
            raise ValueError("Fornisci una query valida.")
        
        with DDGS(verify=False) as ddgs:
            risultati = list(ddgs.text(query, region='it-it', safesearch='on', max_results=3))

        output = []
        for i, r in enumerate(risultati, 1):
            titolo = r.get('title', '')
            url = r.get('href') or r.get('url') or ''
            snippet = r.get('body','')
            output.append(f'{i}. {titolo}\n{url}\n{snippet}')

        return '\n\n'.join(output)
    

class AddNumbers(BaseTool):
    name: str = 'Add Tool'
    description: str = 'Uno strumento per sommare due numeri'
    
    def _run(self, first_number: str, second_number: str):
        if not first_number or not second_number:
            raise ValueError("Fornisci dei numeri validi!")
        first = int(first_number)
        second = int(second_number)
        return first + second
"""
Modulo per il DDGSearch - Strumento CrewAI per ricerca web.

Questo modulo implementa un tool CrewAI che utilizza DuckDuckGo per effettuare
ricerche web in tempo reale e restituire risultati formattati.
"""

from ddgs import DDGS
from crewai.tools import BaseTool


class DDGSearch(BaseTool):
    """
    Tool CrewAI per la ricerca web utilizzando DuckDuckGo.
    
    Questo tool permette di effettuare ricerche web in tempo reale utilizzando
    l'API di DuckDuckGo, restituendo risultati formattati con titolo, URL e snippet.
    
    Attributes:
        name (str): Nome del tool per CrewAI
        description (str): Descrizione del tool per CrewAI
    """
    
    name: str = 'duckduckgo'
    description: str = 'Uno strumento per ricercare risultati sul web'

    def _run(self, query: str) -> str:
        """
        Esegue una ricerca web utilizzando DuckDuckGo.
        
        Effettua una ricerca web con i seguenti parametri:
        - Regione: Italia (it-it)
        - Safe search: attivato
        - Numero massimo di risultati: 3
        
        Args:
            query (str): Query di ricerca da eseguire
            
        Returns:
            str: Risultati formattati con titolo, URL e snippet per ogni risultato
            
        Raises:
            ValueError: Se la query è vuota o None
        """
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



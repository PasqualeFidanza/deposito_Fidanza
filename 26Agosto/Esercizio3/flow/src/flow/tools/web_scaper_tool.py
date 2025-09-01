from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from crewai.tools import tool
from ddgs import DDGS
from typing import Optional, List, Dict, Any
import json
from crewai_tools import RagTool
import os



@tool("duckduckgo_search")
def duckduckgo_search_tool(
    url: str,
    max_results: int = 5,
    region: str = "wt-wt",
    safesearch: str = "moderate",
    timelimit: Optional[str] = None
) -> str:
    """
    Search the web with DuckDuckGo and return a JSON string of results.
    Works with both 'ddgs' and 'duckduckgo_search' packages.

    Args:
      url: url da cui estrarre informazioni
      max_results: number of results (1–50)
      region: e.g., 'it-it', 'us-en', 'wt-wt'
      safesearch: 'off' | 'moderate' | 'strict'
      timelimit: None or 'd','w','m','y' (last day/week/month/year)

    Returns:
      JSON string: [{"title": "...", "href": "...", "body": "..."}, ...]
    """
    # Alcune versioni espongono .text(), altre .search()
    with DDGS() as client:
        if hasattr(client, "text"):
            results = list(client.text(
                url,
                region=region,
                safesearch=safesearch,
                timelimit=timelimit,
                max_results=max_results
            ))
        else:
            # fallback molto compatibile
            results = list(client.search(
                url,
                region=region,
                safesearch=safesearch,
                timelimit=timelimit,
                max_results=max_results
            ))

    cleaned: List[Dict[str, Any]] = [
        {
            "title": r.get("title"),
            "href": r.get("href") or r.get("url"),
            "body": r.get("body") or r.get("snippet"),
        }
        for r in results
    ]
    return json.dumps(cleaned, ensure_ascii=False)



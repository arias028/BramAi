from ddgs import DDGS
from collections import namedtuple

# Define a simple result structure to match what bram_ai.py expects
WebSearchResult = namedtuple('WebSearchResult', ['results'])
Result = namedtuple('Result', ['body', 'url'])

def ddg_search(query: str, aiohttp_session=None, max_results=5) -> WebSearchResult:
    """
    Performs a web search using DuckDuckGo and returns results in a structured format.
    The aiohttp_session is ignored as we are using the synchronous version.
    """
    results = []
    try:
        # Using a context manager for the search client
        with DDGS() as ddgs:
            # Fetch search results using the 'keywords' parameter
            search_results = list(ddgs.text(keywords=query, max_results=max_results))
            if search_results:
                for r in search_results:
                    # Map the fields to the expected 'Result' structure
                    results.append(Result(body=r['body'], url=r['href']))
        
        return WebSearchResult(results=results)

    except Exception as e:
        print(f"❌ An error occurred during web search: {e}")
        return WebSearchResult(results=[])

if __name__ == '__main__':
    # Example usage for testing
    query = "what is an asteroid?"
    search_results = ddg_search(query)
    if search_results and search_results.results:
        print(f"Found {len(search_results.results)} results for '{query}':")
        for i, result in enumerate(search_results.results):
            print(f"\n--- Result {i+1} ---")
            print(f"URL: {result.url}")
            print(f"Body Snippet: {result.body[:200]}...")
    else:
        print(f"No results found for '{query}'.") 
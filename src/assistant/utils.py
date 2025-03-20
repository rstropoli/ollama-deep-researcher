import os
import asyncio
import requests
from typing import Dict, Any, List, Optional
from langsmith import traceable
from tavily import TavilyClient
from duckduckgo_search import DDGS
from langchain_community.utilities import SearxSearchWrapper
from langchain_community.document_loaders import PDFMinerLoader

from crawl4ai import *

def deduplicate_sources(search_response):
    """
    Takes either a single search response or list of responses from search APIs 
    and deduplicates the list by url. Then returns a new list of search results
    in the format { "results": [ { result_obj }, ... ]}.
    Limits the raw_content to approximately max_tokens_per_source.
    include_raw_content specifies whether to include the raw_content from Tavily in the formatted string.
    
    Args:
        search_response: Either:
            - A dict with a 'results' key containing a list of search results
            - A list of dicts, each containing search results
            - A single search result
            
    Returns:
        str: A dict with a 'results' key containing a list of deduplicated search results
    """
    # Convert input to list of results
    if isinstance(search_response, dict):
        sources_list = search_response['results']
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and 'results' in response:
                sources_list.extend(response['results'])
            else:
                sources_list.extend(response)
    else:
        raise ValueError("Input must be either a dict with 'results' or a list of search results")
    
    return {"results": sources_list}


def deduplicate_and_format_sources(search_response, max_tokens_per_source, include_raw_content=False):
    """
    Takes either a single search response or list of responses from search APIs and formats them.
    Limits the raw_content to approximately max_tokens_per_source.
    include_raw_content specifies whether to include the raw_content from Tavily in the formatted string.
    
    Args:
        search_response: Either:
            - A dict with a 'results' key containing a list of search results
            - A list of dicts, each containing search results
            
    Returns:
        str: Formatted string with deduplicated sources
    """
    # Convert input to list of results
    if isinstance(search_response, dict):
        sources_list = search_response['results']
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and 'results' in response:
                sources_list.extend(response['results'])
            else:
                sources_list.extend(response)
    else:
        raise ValueError("Input must be either a dict with 'results' or a list of search results")
    
    # Deduplicate by URL
    unique_sources = {}
    for source in sources_list:
        if source['url'] not in unique_sources:
            unique_sources[source['url']] = source
    
    # Format output
    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source {source['title']}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
                
    return formatted_text.strip()

def format_sources(search_results):
    """Format search results into a bullet-point list of sources.
    
    Args:
        search_results (dict): Tavily search response containing results
        
    Returns:
        str: Formatted string with sources and their URLs
    """
    
    if isinstance(search_results, list) and len(search_results) > 0:
        full_results = ""
        for search_result in search_results:
            full_results += '\n'.join(
                f"* {source['title']} : {source['url']}"
                for source in search_result['results'] )
        return full_results
     
    return '\n'.join(
        f"* {source['title']} : {source['url']}"
        for source in search_results['results']
    )

async def crawl_batch ( urls : list ) -> bool :
    browser_config = BrowserConfig(headless=True, verbose=False)
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        check_robots_txt=True,  # Will respect robots.txt for each URL
        stream=True  # Default: get all results at once
    )
    
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=10,
        monitor=CrawlerMonitor(
            display_mode=DisplayMode.DETAILED,
        )
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
                        
        # Process results as they become available
        async for result in await crawler.arun_many(
            urls=urls,
            config=run_config,
            dispatcher=dispatcher
        ):
            if result.success:
                return result.markdown
            else:
                return f"Failed to crawl {result.url}: {result.error_message}"

    
@traceable
def searxng_search(query: str, max_results: int = 3, fetch_full_page: bool = False) -> Dict[str, List[Dict[str, str]]]:
    """Search the web using SearXng.
    
    Args:
        query (str): The search query to execute
        max_results (int): Maximum number of results to return
        
    Returns:
        dict: Search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str): Same as content since SearXng doesn't provide full page content
    """
       
    try:
        results = []    
        searx = SearxSearchWrapper(searx_host="http://localhost:18080", unsecure=True)
        search_results = searx.results(query, num_results=max_results ) 
                    
        for r in search_results:
            url = r.get('link')
            title = r.get('title')
            content = r.get('snippet')
            
            if not all([url, title, content]):
                print(f"Warning: Incomplete result from SearXng: {r}")
                continue

            raw_content = content
            if fetch_full_page:
                try:
                    
                    if url.endswith('.pdf'):
                        loader = PDFMinerLoader(url)
                        docs = loader.load()
                        raw_content = docs[0].content
                    else:
                        raw_content = asyncio.run( crawl_batch([url]) )
                        
                        # Try to fetch the full page content using curl
                        # import urllib.request
                        # from bs4 import BeautifulSoup

                        # response = urllib.request.urlopen(url)
                        # html = response.read()
                        # soup = BeautifulSoup(html, 'html.parser')
                        # raw_content = soup.get_text()
                    
                    # Try to fetch the full page content using curl
                    # import urllib.request
                    # from bs4 import BeautifulSoup
                    
                    # req = urllib.request.Request(
                    #    url, 
                    #    data=None, 
                    #    headers={
                    #        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
                    #    }
                    #)

                    # response = urllib.request.urlopen(req)
                    # html = response.read()
                    # soup = BeautifulSoup(html, 'html.parser')
                    # raw_content = soup.get_text()
                    
                except Exception as e:
                    print(f"Warning: Failed to fetch full page content for {url}: {str(e)}")
            
            # Add result to list
            result = {
                "title": title,
                "url": url,
                "content": content,
                "raw_content": raw_content
            }
            results.append(result)
        
        return {"results": results}
    except Exception as e:
        print(f"Error in SearXng search: {str(e)}")
        print(f"Full error details: {type(e).__name__}")
        return {"results": []}
    
@traceable
def duckduckgo_search(query: str, max_results: int = 3, fetch_full_page: bool = False) -> Dict[str, List[Dict[str, str]]]:
    """Search the web using DuckDuckGo.
    
    Args:
        query (str): The search query to execute
        max_results (int): Maximum number of results to return
        
    Returns:
        dict: Search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str): Same as content since DDG doesn't provide full page content
    """
    try:
        with DDGS() as ddgs:
            results = []
            search_results = list(ddgs.text(query, max_results=max_results))
            
            for r in search_results:
                url = r.get('href')
                title = r.get('title')
                content = r.get('body')
                
                if not all([url, title, content]):
                    print(f"Warning: Incomplete result from DuckDuckGo: {r}")
                    continue

                raw_content = content
                if fetch_full_page:
                    try:
                        # Try to fetch the full page content using curl
                        import urllib.request
                        from bs4 import BeautifulSoup

                        response = urllib.request.urlopen(url)
                        html = response.read()
                        soup = BeautifulSoup(html, 'html.parser')
                        raw_content = soup.get_text()
                        
                    except Exception as e:
                        print(f"Warning: Failed to fetch full page content for {url}: {str(e)}")
                
                # Add result to list
                result = {
                    "title": title,
                    "url": url,
                    "content": content,
                    "raw_content": raw_content
                }
                results.append(result)
            
            return {"results": results}
    except Exception as e:
        print(f"Error in DuckDuckGo search: {str(e)}")
        print(f"Full error details: {type(e).__name__}")
        return {"results": []}

@traceable
def tavily_search(query, include_raw_content=True, max_results=3):
    """ Search the web using the Tavily API.
    
    Args:
        query (str): The search query to execute
        include_raw_content (bool): Whether to include the raw_content from Tavily in the formatted string
        max_results (int): Maximum number of results to return
        
    Returns:
        dict: Search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str): Full content of the page if available"""
     
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set")
    tavily_client = TavilyClient(api_key=api_key)
    return tavily_client.search(query, 
                         max_results=max_results, 
                         include_raw_content=include_raw_content)

@traceable
def perplexity_search(query: str, perplexity_search_loop_count: int) -> Dict[str, Any]:
    """Search the web using the Perplexity API.
    
    Args:
        query (str): The search query to execute
        perplexity_search_loop_count (int): The loop step for perplexity search (starts at 0)
  
    Returns:
        dict: Search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str): Full content of the page if available
    """

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}"
    }
    
    payload = {
        "model": "sonar-pro",
        "messages": [
            {
                "role": "system",
                "content": "Search the web and provide factual information with sources."
            },
            {
                "role": "user",
                "content": query
            }
        ]
    }
    
    response = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers=headers,
        json=payload
    )
    response.raise_for_status()  # Raise exception for bad status codes
    
    # Parse the response
    data = response.json()
    content = data["choices"][0]["message"]["content"]

    # Perplexity returns a list of citations for a single search result
    citations = data.get("citations", ["https://perplexity.ai"])
    
    # Return first citation with full content, others just as references
    results = [{
        "title": f"Perplexity Search {perplexity_search_loop_count + 1}, Source 1",
        "url": citations[0],
        "content": content,
        "raw_content": content
    }]
    
    # Add additional citations without duplicating content
    for i, citation in enumerate(citations[1:], start=2):
        results.append({
            "title": f"Perplexity Search {perplexity_search_loop_count + 1}, Source {i}",
            "url": citation,
            "content": "See above for full content",
            "raw_content": None
        })
    
    return {"results": results}
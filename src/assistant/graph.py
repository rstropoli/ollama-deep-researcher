import json

from typing_extensions import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph

from assistant.configuration import Configuration, SearchAPI
from assistant.utils import deduplicate_sources, deduplicate_and_format_sources, tavily_search, format_sources, perplexity_search, duckduckgo_search, searxng_search
from assistant.state import SummaryState, SummaryStateInput, SummaryStateOutput
from assistant.prompts import query_writer_instructions, summarizer_instructions, reflection_instructions, article_summarizer_instructions
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os

# Nodes
def generate_query(state: SummaryState, config: RunnableConfig):
    """ Generate a query for web search """

    # Format the prompt
    query_writer_instructions_formatted = query_writer_instructions.format(research_topic=state.research_topic)

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    llm_json_mode = ChatOllama(base_url=configurable.ollama_base_url, model=configurable.local_llm, temperature=0,
                    num_predict=configurable.num_predictions, num_ctx=configurable.num_ctx, format="json")
    result = llm_json_mode.invoke(
        [SystemMessage(content=query_writer_instructions_formatted),
        HumanMessage(content=f"Generate a query for web search:")]
    )
    query = json.loads(result.content)

    return {"search_query": query['query']}

def web_research(state: SummaryState, config: RunnableConfig):
    """ Gather information from the web """

    # Configure
    configurable = Configuration.from_runnable_config(config)

    # Handle both cases for search_api:
    # 1. When selected in Studio UI -> returns a string (e.g. "tavily")
    # 2. When using default -> returns an Enum (e.g. SearchAPI.TAVILY)
    if isinstance(configurable.search_api, str):
        search_api = configurable.search_api
    else:
        search_api = configurable.search_api.value

    # Determine the structure of search_query
    if isinstance(state.search_query, dict) and "queries" in state.search_query:
        # multiple queries structured in a dictionary
        queries = state.search_query["queries"]
    elif isinstance(state.search_query, dict):
        # single query structured in a dictionary
        queries = [{"query": state.search_query["query"]}]
    else:   # non structured query, ie just a string
        queries = [{"query": state.search_query}]

    all_search_results = []
    all_search_strs = []

    for query in queries:
        search_query = query["query"]
        
        # Search the web
        if search_api == "tavily":
            search_results = tavily_search(state.search_query, include_raw_content=True, max_results=1)
            #search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=int(configurable.max_tokens_per_source), include_raw_content=True)
        elif search_api == "perplexity":
            search_results = perplexity_search(state.search_query, state.research_loop_count)
            #search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=int(configurable.max_tokens_per_source), include_raw_content=False)
        elif search_api == "duckduckgo":
            search_results = duckduckgo_search(state.search_query, max_results=int(configurable.max_search_results), fetch_full_page=configurable.fetch_full_page)
            #search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=int(configurable.max_tokens_per_source), include_raw_content=True)
        elif search_api == "searxng":
             search_results = searxng_search(state.search_query, max_results=int(configurable.max_search_results), fetch_full_page=configurable.fetch_full_page)
            #search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=int(configurable.max_tokens_per_source), include_raw_content=True)
        else:
            raise ValueError(f"Unsupported search API: {configurable.search_api}")
        
        search_results = deduplicate_sources(search_results)
        # search_str = summarize_article_content(search_results, state, config)
        
        # Assuming that the search results are in the  form of a dDict containing a 'results'
        # key with a list of search results
        
        all_search_results.append(search_results)        
        
        result_list = search_results['results']
        for result in result_list:
            #content = summarize_article_content(result.get('raw_content', result['content']), state, config)
            #result['content'] = content
            all_search_strs.append(result.get('raw_content'))
                            
    return {"sources_gathered": [format_sources(all_search_results)], "research_loop_count": state.research_loop_count + 1, "web_research_results": [all_search_strs]}

def summarize_article_content(article_content: str, state: SummaryState, config: RunnableConfig):
    
    # Build the human message
    human_message_content = (
        f"<User Input> \n {state.research_topic} \n <User Input>\n\n"
        f"<Article> \n {article_content} \n <Article>\n\n"
    )
         
    # Run the LLM
    configurable = Configuration.from_runnable_config(config)
    llm = ChatOllama(base_url=configurable.ollama_base_url, model=configurable.local_llm, temperature=0,
                     num_predict=configurable.num_predictions, num_ctx=configurable.num_ctx)
    result = llm.invoke(
        [SystemMessage(content=article_summarizer_instructions),
        HumanMessage(content=human_message_content)]
    )
  
    return result.content

def summarize_sources(state: SummaryState, config: RunnableConfig):
    """ Summarize the gathered sources """

    # Existing summary
    existing_summary = state.running_summary

    # Most recent web research
    most_recent_web_research = state.web_research_results[-1]

    # Build the human message
    if existing_summary:
        human_message_content = (
            f"<User Input> \n {state.research_topic} \n <User Input>\n\n"
            f"<Existing Summary> \n {existing_summary} \n <Existing Summary>\n\n"
            f"<New Search Results> \n {most_recent_web_research} \n <New Search Results>"
        )
    else:
        human_message_content = (
            f"<User Input> \n {state.research_topic} \n <User Input>\n\n"
            f"<Search Results> \n {most_recent_web_research} \n <Search Results>"
        )

    # Run the LLM
    configurable = Configuration.from_runnable_config(config)
    llm = ChatOllama(base_url=configurable.ollama_base_url, model=configurable.local_llm, temperature=0,
                     num_predict=configurable.num_predictions, num_ctx=configurable.num_ctx)
    result = llm.invoke(
        [SystemMessage(content=summarizer_instructions),
        HumanMessage(content=human_message_content)]
    )

    running_summary = result.content

    # TODO: This is a hack to remove the <think> tags w/ Deepseek models
    # It appears very challenging to prompt them out of the responses
    while "<think>" in running_summary and "</think>" in running_summary:
        start = running_summary.find("<think>")
        end = running_summary.find("</think>") + len("</think>")
        running_summary = running_summary[:start] + running_summary[end:]

    return {"running_summary": running_summary}

def reflect_on_summary(state: SummaryState, config: RunnableConfig):
    """ Reflect on the summary and generate a follow-up query """

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    llm_json_mode = ChatOllama(base_url=configurable.ollama_base_url, model=configurable.local_llm, temperature=0, num_predict=configurable.num_predictions, num_ctx=configurable.num_ctx, format="json")
    
    result = llm_json_mode.invoke(
        [SystemMessage(content=reflection_instructions.format(research_topic=state.research_topic)),
        HumanMessage(content=f"Identify a knowledge gap and generate a follow-up web search query based on our existing knowledge: {state.running_summary}")]
    )
    follow_up_query = json.loads(result.content)

    # Get the follow-up query
    query = follow_up_query.get('follow_up_query')

    # JSON mode can fail in some cases
    if not query:

        # Fallback to a placeholder query
        return {"search_query": f"Tell me more about {state.research_topic}"}

    # Update search query with follow-up query
    return {"search_query": follow_up_query['follow_up_query']}

def finalize_summary(state: SummaryState):
    """ Finalize the summary """

    # Format all accumulated sources into a single bulleted list
    all_sources = "\n".join(source for source in state.sources_gathered)
    state.running_summary = f"## Summary\n\n{state.running_summary}\n\n ### Sources:\n{all_sources}"
    return {"running_summary": state.running_summary}

def route_research(state: SummaryState, config: RunnableConfig) -> Literal["finalize_summary", "web_research"]:
    """ Route the research based on the follow-up query """

    configurable = Configuration.from_runnable_config(config)
    if int(state.research_loop_count) <= int(configurable.max_web_research_loops):
        return "web_research"
    else:
        return "finalize_summary"

# Add nodes and edges
builder = StateGraph(SummaryState, input=SummaryStateInput, output=SummaryStateOutput, config_schema=Configuration)
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("summarize_sources", summarize_sources)
builder.add_node("reflect_on_summary", reflect_on_summary)
builder.add_node("finalize_summary", finalize_summary)

# Add edges
builder.add_edge(START, "generate_query")
builder.add_edge("generate_query", "web_research")
builder.add_edge("web_research", "summarize_sources")
builder.add_edge("summarize_sources", "reflect_on_summary")
builder.add_conditional_edges("reflect_on_summary", route_research)
builder.add_edge("finalize_summary", END)

graph = builder.compile()

def run_agent( research_topic : str ):
    """ Run the agent """
    # Load environment variables from a .env file
    load_dotenv()

    # query = input("Enter your question (or type /exit to quit): ")
    # if query.lower() == "/exit":
    #    break
    
    state = SummaryState(research_topic=research_topic, search_query="", running_summary="", research_loop_count=0, sources_gathered=[], web_research_results=[])

    state = graph.invoke(state)

    return state

if __name__ == "__main__":
    print ( run_agent("Advances in Attention Layers in Large Language Models") )
import operator
from dataclasses import dataclass, field
from typing_extensions import TypedDict, Annotated

@dataclass(kw_only=True)
class SummaryState:
    research_topic: str = field(default=None) # Report topic     
    search_query: str = field(default=None) # Search query
    web_research_results: Annotated[list, operator.add] = field(default_factory=list) # Web Query and content 
    summarized_results: Annotated[list, operator.add] = field(default_factory=list) # Distilled Summary of individual sources
    sources_gathered: Annotated[list, operator.add] = field(default_factory=list) # Running summary of sources 
    research_loop_count: int = field(default=0) # Research loop count
    running_summary: str = field(default=None) # Final report


@dataclass(kw_only=True)
class SummaryStateInput:
    research_topic: str = field(default=None) # Report topic     

@dataclass(kw_only=True)
class SummaryStateOutput:
    running_summary: str = field(default=None) # Final report
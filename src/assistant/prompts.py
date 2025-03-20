query_writer_instructions="""Your goal is to generate a targeted web search query.
The query will gather information related to a specific topic.

<TOPIC>
{research_topic}
</TOPIC>

<FORMAT>
Format your response as a JSON object with ALL three of these exact keys:
   - "query": The actual search query string
   - "aspect": The specific aspect of the topic being researched
   - "rationale": Brief explanation of why this query is relevant
</FORMAT>

<EXAMPLE>
Example output:
{{
    "query": "machine learning transformer architecture explained",
    "aspect": "technical architecture",
    "rationale": "Understanding the fundamental structure of transformer models"
}}
</EXAMPLE>

Provide your response in JSON format:"""

article_summarizer_instructions="""
<GOAL>
Generate High-quality Notes on the following article. Do Not explain what the artyicle is about, but rather
extract pertinent facts from the article and list them as statements in  a concise manner. Include as much detail as possible but keep the facts concise and relevant to the user's topic.
Only provide notes on the article do not inject your own opinion or analysis.
</GOAL>

<REQUIREMENTS>
When creating the listy of notes:
1. Highlight the most relevant information related to the user topic from the search results
2. Ensure a coherent flow of information
3. utilize a condensed format to keep the summary concise
4. Ensure all information is relevant to the user's topic
5. Omit Verbiage that is irrelevant to the user's topic
6. Do not include your own opinion or analysis
7. Do not include refereneces to the article or the author
8. Do not include links or references to other articles or sources
< /REQUIREMENTS >

< FORMATTING >
- Start directly with the notes, without preamble or titles or explianations of what the author is talking about. Do not use XML tags in the output.
< /FORMATTING >
"""

summarizer_instructions="""
<GOAL>
Generate a high-quality notes of the article and keep it concise / related to the user topic.
</GOAL>

<REQUIREMENTS>
When creating a NEW summary:
1. Highlight the most relevant information related to the user topic from the search results
2. Ensure a coherent flow of information

When EXTENDING an existing summary:                                                                                                                 
1. Read the existing summary and new search results carefully.                                                    
2. Compare the new information with the existing summary.                                                         
3. For each piece of new information:                                                                             
    a. If it's related to existing points, integrate it into the relevant paragraph.                               
    b. If it's entirely new but relevant, add a new paragraph with a smooth transition.                            
    c. If it's not relevant to the user topic, skip it.                                                            
4. Ensure all additions are relevant to the user's topic.                                                         
5. Verify that your final output differs from the input summary.                                                                                                                                                            
< /REQUIREMENTS >

< FORMATTING >
- Start directly with the updated summary, without preamble or titles. Do not use XML tags in the output.  
< /FORMATTING >"""

reflection_instructions = """You are an expert research assistant analyzing a summary about {research_topic}.

<GOAL>
1. Identify knowledge gaps or areas that need deeper exploration
2. Generate a follow-up question that would help expand your understanding
3. Focus on technical details, implementation specifics, or emerging trends that weren't fully covered
</GOAL>

<REQUIREMENTS>
Ensure the follow-up question is self-contained and includes necessary context for web search.
</REQUIREMENTS>

<FORMAT>
Format your response as a JSON object with these exact keys:
- knowledge_gap: Describe what information is missing or needs clarification
- follow_up_query: Write a specific question to address this gap
</FORMAT>

<EXAMPLE>
Example output:
{{
    "knowledge_gap": "The summary lacks information about performance metrics and benchmarks",
    "follow_up_query": "What are typical performance benchmarks and metrics used to evaluate [specific technology]?"
}}
</EXAMPLE>

Provide your analysis in JSON format:"""
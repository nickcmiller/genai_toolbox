import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from chunk_and_embed.embedding_functions import create_openai_embedding
from chunk_and_embed.embedding_functions import query_chunks_with_embeddings
from text_prompting.model_calls import openai_text_response

from typing import Callable
import string
from string import Template

llm_system_prompt_default = f"""
Use numbered references (e.g. [1]) to cite the sources that are given to you in your answers.
List the references used at the bottom of your answer.
Do not refer to the source material in your text, only in your number citations
Give a detailed answer.
"""

def llm_response_with_query(
    chunks_with_embeddings: list[dict],
    question: str,
    llm_system_prompt: str = llm_system_prompt_default,
    source_template: str = "Title: '{title}',\nText: '{text}'\n",
    template_args: dict = {"title": "title", "text": "text"},
    embedding_function: Callable = create_openai_embedding,
    query_model: str = "text-embedding-3-large",
    threshold: float = 0.4,
    max_query_chunks: int = 3,
    llm_function: Callable = openai_text_response,
    llm_model_choice: str = "4o",
) -> dict:
    query_response = query_chunks_with_embeddings(
        query=question, 
        chunks_with_embeddings=chunks_with_embeddings,
        embedding_function=embedding_function, 
        model_choice=query_model, 
        threshold=threshold,
        max_returned_chunks=max_query_chunks
    )

    if len(query_response) == 0:
        return "Sources are not relevant enough to answer this question"

    # Check that query_response chunks contain 'title' and 'text'
    for chunk in query_response:
        if 'title' not in chunk or 'text' not in chunk:
            raise ValueError("Each chunk in query_response must contain 'title' and 'text' keys")

    sources = ""
    for chunk in query_response:
        format_args = {key: chunk.get(arg, f"No {key} Provided") for key, arg in template_args.items()}
        formatted_source = source_template.format(**format_args)
        sources += f"{20*'-'}\n{formatted_source}\n{20*'-'}" 

    prompt = f"Question: {question}\n\nSources:\n{20*'-'}\n{20*'-'}\n {sources}"
    llm_response = llm_function(
        prompt, 
        system_instructions=llm_system_prompt, 
        model_choice=llm_model_choice,
    )

    return {
        "llm_response": llm_response,
        "query_response": query_response
    }
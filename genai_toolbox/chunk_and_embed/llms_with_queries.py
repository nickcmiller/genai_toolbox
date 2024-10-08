from genai_toolbox.chunk_and_embed.embedding_functions import create_openai_embedding, find_similar_chunks
from genai_toolbox.text_prompting.model_calls import openai_text_response, fallback_text_response

from typing import Callable, List, Dict, Any
import string
from string import Template

llm_system_prompt_default = f"""
Use numbered references (e.g. [1]) to cite the sources that are given to you in your answers.
List the references used at the bottom of your answer.
Do not refer to the source material in your text, only in your number citations
Give a detailed answer.
"""

def llm_response_with_query(
    similar_chunks: List[Dict[str, Any]],
    question: str,
    history_messages: List[Dict[str, str]] = None,
    llm_system_prompt: str = llm_system_prompt_default,
    source_template: str = "Title: '{title}',\nText: '{text}'\n",
    template_args: dict = {"title": "title", "text": "text"},
    llm_model_order: List[Dict[str, str]] = [
        {
            "provider": "openai", 
            "model": "4o-mini"
        },
        {
            "provider": "groq", 
            "model": "llama3.1-70b"
        },
        {
            "provider": "perplexity", 
            "model": "llama3.1-70b"
        },
        {
            "provider": "anthropic", 
            "model": "sonnet"
        }
    ],
) -> str:

    if len(similar_chunks) == 0:
        return "Sources are not relevant enough to answer this question"

    # Check that query_response chunks contain 'title' and 'text'
    for chunk in similar_chunks:
        if 'title' not in chunk or 'text' not in chunk:
            raise ValueError("Each chunk in query_response must contain 'title' and 'text' keys")

    sources = ""
    for chunk in similar_chunks:
        format_args = {key: chunk.get(arg, f"No {key} Provided") for key, arg in template_args.items()}
        formatted_source = source_template.format(**format_args)
        sources += f"{20*'-'}\n{formatted_source}\n{20*'-'}"

    prompt = f"Question: {question}\n\nSources:\n{20*'-'}\n{20*'-'}\n {sources}"
    llm_response = fallback_text_response(
        prompt, 
        history_messages=history_messages,
        system_instructions=llm_system_prompt, 
        model_order=llm_model_order
    )

    return llm_response

def stream_response_with_query(
    similar_chunks: List[Dict[str, Any]],
    question: str,
    history_messages: List[Dict[str, str]] = None,
    llm_system_prompt: str = llm_system_prompt_default,
    source_template: str = "Title: '{title}',\nText: '{text}'\n",
    template_args: dict = {"title": "title", "text": "text"},
    llm_model_order: List[Dict[str, str]] = [
        {
            "provider": "openai", 
            "model": "4o-mini"
        },
        {
            "provider": "groq", 
            "model": "llama3.1-70b"
        },
        {
            "provider": "perplexity", 
            "model": "llama3.1-70b"
        },
        {
            "provider": "anthropic", 
            "model": "sonnet"
        }
    ],
) -> str:
    if len(similar_chunks) == 0:
        return "Sources are not relevant enough to answer this question"

    # Check that query_response chunks contain 'title' and 'text'
    for chunk in similar_chunks:
        if 'title' not in chunk or 'text' not in chunk:
            raise ValueError("Each chunk in query_response must contain 'title' and 'text' keys")

    sources = ""
    for i,chunk in enumerate(similar_chunks):
        format_args = {key: chunk.get(arg, f"No {key} Provided") for key, arg in template_args.items()}
        formatted_source = source_template.format(**format_args)
        sources += f"{20*'-'}\nSource{i+1}: {formatted_source}\n{20*'-'}"

    prompt = f"Question: {question}\n\nSources:\n{20*'-'}\n{20*'-'}\n {sources}"
    llm_response = fallback_text_response(
        prompt, 
        history_messages=history_messages,
        system_instructions=llm_system_prompt, 
        model_order=llm_model_order,
        stream=True
    )
    return llm_response
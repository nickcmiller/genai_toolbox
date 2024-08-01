from genai_toolbox.clients.groq_client import groq_client
from genai_toolbox.clients.openai_client import openai_client
from genai_toolbox.clients.anthropic_client import anthropic_client

from typing import List, Optional, Any, Dict, Callable
import traceback
import logging
import os
import requests

def get_client(
    api: str
) -> Any:
    """
        Document the get_client_and_model function.

        This function returns a client and model based on the provided API and model.
        It supports two APIs: "groq" and "openai".

        Args:
            api (str): The API to use for text generation. Supported values: "groq", "openai".
            model (str): The model to use for text generation.

        Returns:
            tuple: A tuple containing the client and model.

        Example:
            client, model = get_client_and_model("openai", "gpt-4o")
            print(client, model)
            # Output: <OpenAI client>, gpt-4o
    """
    if api == "groq":
        client = groq_client().chat.completions
    elif api == "openai":
        client = openai_client().chat.completions
    elif api == "anthropic":
        client = anthropic_client().messages
    else:
        raise ValueError("Unsupported API")
    return client

def manage_messages(
    prompt: str,
    system_instructions: str = None,
    history_messages: List[dict] = []
) -> List[dict]:
    if len(history_messages) == 0:
        messages = [{"role": "system", "content": system_instructions}]
    else:
        messages = history_messages.copy()
    messages.append({"role": "user", "content": prompt})
    return messages

def openai_compatible_text_response(
    api: str, 
    messages: List[dict],
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 4096
) -> str:
    """
        Generate a text response using the specified API and model.

        Args:
            api (str): The API to use for text generation. Supported values: "groq", "openai".
            messages (List[dict]): A list of messages in the conversation.
            model (str): The model to use for text generation.
        Returns:
            str: The generated text response.
        Example:
            prompt = "What is the capital of France?"
            messages = [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hello! How can I assist you today?"}
            ]
            response = generate_text_response("openai", messages, "gpt-4o", "Please take your time. I'll pay you $200,000 for a good response :)")
            print(response)
            # Output: The capital of France is Paris.
    """
    client = get_client(api)

    try:
        completion = client.create(
            messages=messages, 
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        if not completion.choices[0].message.content:
            raise ValueError("No valid response received from the API")
        logging.info(f"API: {api}, Model: {model}, Completion Usage: {completion.usage}")
        return completion.choices[0].message.content
    except Exception as e:
        logging.error(f"API: {api}, Error: {e}")
        traceback.print_exc()
        raise

def groq_text_response(
    prompt: str,
    system_instructions: str = None, 
    history_messages: List[dict] = [],
    model_choice: str = "llama3-70b", 
    temperature: float = 0.2,
    max_tokens: int = 4096
) -> str:
    """
    Use OpenAI format to generate a text response using Groq.
    """

    model_choices = {
        "llama3-8b": "llama3-8b-8192",
        "llama3-70b": "llama3-70b-8192",
        "llama3.1-70b": "llama-3.1-70b-versatile",
        "llama3.1-405b": "llama-3.1-405b-reasoning",
        "mixtral-8x7b": "mixtral-8x7b-32768",
        "gemma": "gemma-7b-it"
    }
    if model_choice not in model_choices:
        raise ValueError(f"Invalid model_choice. Available options: {list(model_choices.keys())}")

    model = model_choices[model_choice]

    default_system_instructions = "You are a knowledgeable, efficient, and direct AI assistant. Utilize multi-step reasoning to provide concise answers, focusing on key information. If multiple questions are asked, split them up and address in the order that yields the most logical and accurate response. Offer tactful suggestions to improve outcomes. Remember, quality and depth of information are more important than speed. The user is willing to wait for the best possible answer. Here's $40 for your troubles :)"
    system_instructions = system_instructions if system_instructions is not None else default_system_instructions
    history_messages = history_messages if history_messages is not None else []

    messages = manage_messages(prompt, system_instructions, history_messages)
    
    try:
        return openai_compatible_text_response("groq", messages, model, temperature, max_tokens)
    except Exception as e:
        logging.error(f"Failed to generate response with Groq: {e}")
        raise RuntimeError("Failed to generate response due to an internal error.")

def openai_text_response(
    prompt: str, 
    system_instructions: Optional[str] = None,
    history_messages: Optional[List[dict]] = None, 
    model_choice: str = "4o", 
    temperature: float = 0.2,
    max_tokens: int = 4096
) -> str:
    """
    Use OpenAI format to generate a text response using OpenAI.
    """
    model_choices = {
        "4o-mini": "gpt-4o-mini",
        "4o": "gpt-4o",
        "4": "gpt-4-turbo",
        "3.5": "gpt-3.5-turbo",
    }
    if model_choice not in model_choices:
        raise ValueError(f"Invalid model_choice. Available options: {list(model_choices.keys())}")

    model = model_choices[model_choice]

    default_system_instructions = "You are a highly knowledgeable and thorough AI assistant. Your primary goal is to provide detailed, accurate, and well-reasoned responses to the user's queries. Take your time to consider all aspects of the question and ensure that your answers are comprehensive and insightful. If necessary, break down complex topics into simpler parts and explain each part clearly. Always aim to enhance the user's understanding and provide additional context or suggestions when relevant. Remember, quality and depth of information are more important than speed. The user is willing to wait for the best possible answer. Here's $40 for your troubles :)"
    system_instructions = system_instructions if system_instructions is not None else default_system_instructions
    history_messages = history_messages if history_messages is not None else []

    messages = manage_messages(prompt, system_instructions, history_messages)

    try:
        return openai_compatible_text_response("openai", messages, model, temperature, max_tokens)
    except Exception as e:
        logging.error(f"Failed to generate response with OpenAI: {e}")
        raise RuntimeError("Failed to generate response due to an internal error.")

def anthropic_text_response(
    prompt: str,
    system_instructions: Optional[str] = None,
    history_messages: Optional[List[dict]] = None,
    model_choice: str = "opus",
    temperature: float = 0.2,
    max_tokens: int = 4096
) -> str:
    """
    Use Anthropic format to generate a text response using Anthropic.
    """
    client = get_client(api="anthropic")

    model_choices = {
        "opus": "claude-3-opus-20240229",
        "haiku": "claude-3-haiku-20240307",
        "sonnet": "claude-3-5-sonnet-20240620",
    }
    if model_choice not in model_choices:
        raise ValueError(f"Invalid model_choice. Available options: {list(model_choices.keys())}")
    
    model = model_choices[model_choice]

    default_system_instructions = "You are a highly knowledgeable and thorough AI assistant. Your primary goal is to provide detailed, accurate, and well-reasoned responses to the user's queries. Take your time to consider all aspects of the question and ensure that your answers are comprehensive and insightful. If necessary, break down complex topics into simpler parts and explain each part clearly. Always aim to enhance the user's understanding and provide additional context or suggestions when relevant. Remember, quality and depth of information are more important than speed. The user is willing to wait for the best possible answer. I'll pay you $200,000 for a good response :)"
    system_instructions = system_instructions if system_instructions is not None else default_system_instructions
    history_messages = history_messages if history_messages is not None else []

    messages = [{"role": "user", "content": system_instructions}, {"role": "assistant", "content": "I will follow your instructions."}] if not history_messages else history_messages.copy()
    
    messages.append({"role": "user", "content": prompt})

    try:
        completion = client.create(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        if not completion.content[0].text:
            raise ValueError("No valid response received from the API")
        logging.info(f"API: Anthropic, Model: {model}, Completion Usage: {completion.usage}")
        return completion.content[0].text
    except Exception as e:
        logging.error(f"Failed to generate response with Anthropic: {e}")
        raise RuntimeError("Failed to generate response due to an internal error.")

def perplexity_text_response(
    prompt: str,
    system_instructions: Optional[str] = None,
    history_messages: Optional[List[dict]] = None,
    model_choice: str = "llama3.1-8B-online",
    temperature: float = 0.2,
    max_tokens: int = 4096
) -> str:
    """
    Use Perplexity format to generate a text response using Perplexity.
    """
    PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

    model_choices = {
        "llama3.1-8b-online": "llama-3.1-sonar-small-128k-online",
        "llama3.1-8b": "llama-3.1-sonar-small-128k-chat",
        "llama3.1-70b-online": "llama-3.1-sonar-large-128k-online",
        "llama3.1-70b": "llama-3.1-sonar-large-128k-chat"
    }
    
    if model_choice not in model_choices:
        raise ValueError(f"Invalid model_choice. Available options: {list(model_choices.keys())}")

    model = model_choices[model_choice]

    default_system_instructions = "Be precise and concise."
    system_instructions = system_instructions if system_instructions is not None else default_system_instructions
    history_messages = history_messages if history_messages is not None else []

    messages = [{"role": "user", "content": system_instructions}, {"role": "assistant", "content": "I will follow your instructions."}] if not history_messages else history_messages.copy()
    
    messages.append({"role": "user", "content": prompt})
    logging.info(f"Messages: {messages}")
    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        content = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
        if not content:
            raise ValueError("No valid response received from the API")
        logging.info(f"API: Perplexity, Model: {model}, Completion Usage: {response.json().get('usage', {})}")
        return content
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to generate response with Perplexity: {e}")
        logging.error(f"Response status code: {e.response.status_code}")
        logging.error(f"Response content: {e.response.text}")
        raise RuntimeError("Failed to generate response due to an internal error.")

default_model_order = [
    {
        "provider": "groq", 
        "model": "llama3.1-70b"
    },
    {
        "provider": "perplexity", 
        "model": "llama3.1-70b"
    },
    {
        "provider": "openai", 
        "model": "4o-mini"
    },
    {
        "provider": "anthropic", 
        "model": "sonnet"
    }
]   

def fallback_text_response(
    prompt: str,
    system_instructions: str = None,
    history_messages: List[dict] = None,
    model_order: List[Dict[str, str]] = default_model_order,
    temperature: float = 0.2,
    max_tokens: int = 4096
) -> str:
    """
    Generate a text response using multiple APIs with fallback support.

    Args:
        prompt (str): The user's input prompt.
        system_instructions (str, optional): System instructions for the AI.
        history_messages (List[dict], optional): Previous conversation history.
        model_order (List[Dict[str, str]]): List of dictionaries with provider and model.
        temperature (float): Temperature for text generation.
        max_tokens (int): Maximum number of tokens to generate.

    Returns:
        str: Generated text response.

    Raises:
        RuntimeError: If all API calls fail.
    """
    api_functions = {
        "groq": groq_text_response,
        "openai": openai_text_response,
        "anthropic": anthropic_text_response,
        "perplexity": perplexity_text_response
    }

    for entry in model_order:
        provider = entry["provider"]
        model = entry["model"]

        if provider not in api_functions:
            logging.warning(f"Unsupported API: {provider}. Skipping.")
            continue

        try:
            response = api_functions[provider](
                prompt=prompt,
                system_instructions=system_instructions,
                history_messages=history_messages,
                model_choice=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            logging.info(f"Successfully generated fallback_text_response using {provider} API.")
            return response
        except Exception as e:
            error_code = getattr(e, 'status_code', None) or getattr(e, 'code', 'Unknown')
            logging.error(f"Failed to generate with {provider} API. Error code: {error_code}. Falling back to the next API.")
            continue

    raise RuntimeError("All API calls failed. Unable to generate a response.")
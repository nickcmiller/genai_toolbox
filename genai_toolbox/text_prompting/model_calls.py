import sseclient

from typing import List, Optional, Any, Dict, Callable, Generator, Union
import traceback
import logging
import os
import json
import time
import requests


from genai_toolbox.clients.groq_client import groq_client
from genai_toolbox.clients.openai_client import openai_client
from genai_toolbox.clients.anthropic_client import anthropic_client

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

def openai_compatible_response(
    api: str, 
    messages: List[dict],
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    stream: bool = False
) -> Union[str, Generator[str, None, None]]:
    client = get_client(api)
    start_time = time.time()

    try:
        completion = client.create(
            messages=messages, 
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )
        
        if stream:
            return _stream_response(completion, api, model, start_time)
        else:
            return _generate_response(completion, api, model, start_time)
    except Exception as e:
        _handle_error(e, api, model, start_time)

def _stream_response(completion, api, model, start_time):
    def stream_generator():
        total_content = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                total_content += content
                yield content
        _log_response_info(api, model, start_time, len(total_content), streaming=True)
    return stream_generator()

def _generate_response(completion, api, model, start_time):
    if not completion.choices[0].message.content:
        raise ValueError("No valid response received from the API")
    _log_response_info(api, model, start_time, completion.usage)
    return completion.choices[0].message.content

def _log_response_info(api, model, start_time, usage_or_length, streaming=False):
    end_time = time.time()
    response_time = end_time - start_time
    if streaming:
        logging.info(f"API: {api}, Model: {model}, Streaming Response Time: {response_time:.2f}s, Total Content Length: {usage_or_length}")
    else:
        logging.info(f"API: {api}, Model: {model}, Completion Usage: {usage_or_length}, Response Time: {response_time:.2f}s")

def _handle_error(e, api, model, start_time):
    end_time = time.time()
    response_time = end_time - start_time
    error_message = f"API: {api}, Model: {model}, Error: {str(e)}, Response Time: {response_time:.2f}s"
    logging.error(error_message)
    traceback.print_exc()
    raise RuntimeError(error_message)

def openai_text_response(
    prompt: str,
    system_instructions: Optional[str] = None,
    history_messages: Optional[List[dict]] = None,
    model_choice: str = "4o",
    temperature: float = 0.2,
    max_tokens: int = 4096,
    stream: bool = False
) -> Union[str, Generator[str, None, None]]:
    """
    Use OpenAI format to generate a text response using OpenAI.
    Supports both streaming and non-streaming responses.
    """
    if stream:
        print(f"OpenAI Stream Response: \n{prompt},\n{system_instructions}, \n{history_messages}, \n{model_choice}, \n{temperature}, \n{max_tokens}")
    
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
        return openai_compatible_response(  
            api="openai", 
            messages=messages, 
            model=model, 
            temperature=temperature, 
            max_tokens=max_tokens, 
            stream=stream
        )
    except Exception as e:
        error_type = "streaming" if stream else "non-streaming"
        logging.error(f"Failed to generate {error_type} response with OpenAI: {e}")
        raise RuntimeError(f"Failed to generate {error_type} response due to an internal error.")

def groq_text_response(
    prompt: str,
    system_instructions: str = None, 
    history_messages: List[dict] = [],
    model_choice: str = "llama3-70b", 
    temperature: float = 0.2,
    max_tokens: int = 4096,
    stream: bool = False
) -> Union[str, Generator[str, None, None]]:
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
        return openai_compatible_response(
            api="groq", 
            messages=messages, 
            model=model, 
            temperature=temperature, 
            max_tokens=max_tokens, 
            stream=stream
        )
    except Exception as e:
        logging.error(f"Failed to generate response with Groq: {e}")
        raise RuntimeError("Failed to generate response due to an internal error.")

def anthropic_text_response(
    prompt: str,
    system_instructions: Optional[str] = None,
    history_messages: Optional[List[Dict[str, str]]] = None,
    model_choice: str = "sonnet",
    temperature: float = 0.2,
    max_tokens: int = 4096,
    stream: bool = False
) -> Union[str, Generator[str, None, None]]:
    """
        This module provides functions to interact with the Anthropic API for text generation.


        Args:
            prompt (str): The user's input prompt.
            system_instructions (Optional[str]): System instructions for the AI.
            history_messages (Optional[List[Dict[str, str]]]): Previous conversation history.
            model_choice (str): The model to use for generation.
            temperature (float): The temperature for text generation.
            max_tokens (int): The maximum number of tokens to generate.
            stream (bool): Whether to stream the response or not.

        Functions:
                1. get_client(api: str) -> Any:
                    - Returns the appropriate client for the specified API.
                    - Supported APIs: "anthropic".

                2. anthropic_text_response(
                    prompt: str,
                    system_instructions: Optional[str] = None,
                    history_messages: Optional[List[Dict[str, str]]] = None,
                    model_choice: str = "sonnet",
                    temperature: float = 0.2,
                    max_tokens: int = 4096,
                    stream: bool = False
                ) -> Union[str, Generator[str, None, None]]:
                    - Generates a text response using the Anthropic API.
                    - Args:
                        - prompt: The user's input prompt.
                        - system_instructions: Instructions for the AI.
                        - history_messages: Previous conversation history.
                        - model_choice: The model to use for generation.
                        - temperature: The temperature for text generation.
                        - max_tokens: The maximum number of tokens to generate.
                        - stream: Whether to stream the response or not.
                    - Returns:
                        - Generated text response or a generator for streaming.
                    - Raises:
                        - ValueError: If an invalid model_choice is provided.
                        - RuntimeError: If there's an error in generating the response.

                3. _anthropic_get_system_instructions(history_messages: List[Dict[str, str]], default_instructions: str) -> str:
                    - Retrieves system instructions based on conversation history.
                    - Args:
                        - history_messages: Previous conversation history.
                        - default_instructions: Default instructions if no history is present.
                    - Returns:
                        - The appropriate system instructions.

                4. _anthropic_convert_messages(history_messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
                    - Converts history messages into the format required by the Anthropic API.
                    - Args:
                        - history_messages: Previous conversation history.
                    - Returns:
                        - A list of converted messages.

                5. _anthropic_stream_response(client: Any, model: str, messages: List[Dict[str, str]], system_instructions: str, temperature: float, max_tokens: int) -> Generator[str, None, None]:
                    - Streams responses from the Anthropic API.
                    - Args:
                        - client: The Anthropic client.
                        - model: The model to use for generation.
                        - messages: The messages to send to the API.
                        - system_instructions: Instructions for the AI.
                        - temperature: The temperature for text generation.
                        - max_tokens: The maximum number of tokens to generate.
                    - Yields:
                        - Streaming responses from the API.

                6. _anthropic_generate_response(client: Any, model: str, messages: List[Dict[str, str]], system_instructions: str, temperature: float, max_tokens: int) -> str:
                    - Generates a complete response from the Anthropic API.
                    - Args:
                        - client: The Anthropic client.
                        - model: The model to use for generation.
                        - messages: The messages to send to the API.
                        - system_instructions: Instructions for the AI.
                        - temperature: The temperature for text generation.
                        - max_tokens: The maximum number of tokens to generate.
                    - Returns:
                        - The generated response from the API.

        Returns:
            Union[str, Generator[str, None, None]]: Generated text response or a generator for streaming.

        Raises:
            ValueError: If an invalid model_choice is provided.
            RuntimeError: If there's an error in generating the response.
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

    default_system_instructions = "You are a highly knowledgeable and thorough AI assistant. Provide detailed, accurate, and well-reasoned responses to queries."
    
    history_messages = history_messages or []
    system_instructions = system_instructions or _anthropic_get_system_instructions(history_messages, default_system_instructions)

    converted_messages = _anthropic_convert_messages(history_messages)
    converted_messages.append({"role": "user", "content": prompt})

    try:
        if stream:
            return _anthropic_stream_response(client, model, converted_messages, system_instructions, temperature, max_tokens)
        else:
            return _anthropic_generate_response(client, model, converted_messages, system_instructions, temperature, max_tokens)
    except (APIError, APIConnectionError, RateLimitError) as e:
        logging.error(f"Anthropic API error: {str(e)}")
        raise RuntimeError(f"Failed to generate response: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error in Anthropic response generation: {str(e)}")
        raise RuntimeError("An unexpected error occurred while generating the response.")

def _anthropic_get_system_instructions(
    history_messages: List[Dict[str, str]], 
    default: str
) -> str:
    system_messages = [msg['content'] for msg in history_messages if msg['role'] == 'system']
    return " ".join(system_messages) if system_messages else default

def _anthropic_convert_messages(
    messages: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    return [{"role": msg['role'], "content": msg['content']} 
            for msg in messages if msg['role'] in ['user', 'assistant']]

def _anthropic_stream_response(
    client: Callable, 
    model: str, 
    messages: List[Dict[str, str]], 
    system: str, 
    temperature: float, 
    max_tokens: int
) -> Generator[str, None, None]:
    def stream_generator():
        total_content = ""
        start_time = time.time()
        try:
            with client.stream(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
                system=system,
                temperature=temperature
            ) as stream:
                for event in stream:
                    if event.type == "content_block_delta" and event.delta.type == "text_delta":
                        content = event.delta.text
                        total_content += content
                        yield content
            end_time = time.time()
            response_time = end_time - start_time
            logging.info(f"API: Anthropic, Model: {model}, Streaming Response Time: {response_time:.2f}s, Total Content Length: {len(total_content)}")
        except Exception as e:
            logging.error(f"Error during streaming: {str(e)}")
            raise
    return stream_generator()

def _anthropic_generate_response(
    client: Callable, 
    model: str, 
    messages: List[Dict[str, str]], 
    system: str, 
    temperature: float, 
    max_tokens: int
) -> str:
    start_time = time.time()
    completion = client.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        system=system,
        temperature=temperature
    )
    if not completion.content:
        raise ValueError("No valid response received from the API")
    end_time = time.time()
    response_time = end_time - start_time
    logging.info(f"API: Anthropic, Model: {model}, Response Time: {response_time:.2f}s")
    return completion.content

def perplexity_text_response(
    prompt: str,
    system_instructions: Optional[str] = None,
    history_messages: Optional[List[dict]] = None,
    model_choice: str = "llama3.1-8B-online",
    temperature: float = 0.2,
    max_tokens: int = 4096,
    stream: bool = False
) -> Union[str, Generator[str, None, None]]:
    """
    Use Perplexity format to generate a text response using Perplexity.
    Supports both streaming and non-streaming responses.
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
        "max_tokens": max_tokens,
        "stream": stream
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}"
    }

    try:
        if stream:
            return _perplexity_stream_response(url, payload, headers)
        else:
            return _perplexity_generate_response(url, payload, headers)
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to generate response with Perplexity: {e}")
        logging.error(f"Response status code: {e.response.status_code}")
        logging.error(f"Response content: {e.response.text}")
        raise RuntimeError("Failed to generate response due to an internal error.")

def _perplexity_stream_response(
    url: str, 
    payload: dict, 
    headers: dict
) -> Generator[str, None, None]:
    with requests.post(url, json=payload, headers=headers, stream=True) as response:
        response.raise_for_status()
        client = sseclient.SSEClient(response)
        for event in client.events():
            if event.data != '[DONE]':
                try:
                    chunk = json.loads(event.data)
                    content = chunk['choices'][0]['delta'].get('content', '')
                    if content:
                        yield content
                except json.JSONDecodeError:
                    logging.error(f"Failed to decode JSON: {event.data}")

def _perplexity_generate_response(
    url: str, 
    payload: dict, 
    headers: dict
) -> str:
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    content = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
    if not content:
        raise ValueError("No valid response received from the API")
    logging.info(f"API: Perplexity, Model: {payload['model']}, Completion Usage: {response.json().get('usage', {})}")
    return content

default_fallback_model_order = [
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
]   

def fallback_response(
    prompt: str,
    system_instructions: str = None,
    history_messages: List[dict] = None,
    model_order: List[Dict[str, str]] = default_fallback_model_order,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    stream: bool = False
) -> Union[str, Generator[str, None, None]]:
    """
    Generate a text response using multiple APIs with fallback support.
    Supports both streaming and non-streaming responses.

    Args:
        prompt (str): The user's input prompt.
        system_instructions (str, optional): System instructions for the AI.
        history_messages (List[dict], optional): Previous conversation history.
        model_order (List[Dict[str, str]]): List of dictionaries with provider and model.
        temperature (float): Temperature for text generation.
        max_tokens (int): Maximum number of tokens to generate.
        stream (bool): Whether to stream the response or not.

    Returns:
        Union[str, Generator[str, None, None]]: Generated text response or a generator for streaming.

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
                max_tokens=max_tokens,
                stream=stream
            )
            logging.info(f"Successfully generated {'streaming' if stream else ''} response using {provider} API.")
            
            if stream:
                return (chunk for chunk in response)
            else:
                return response

        except Exception as e:
            error_code = getattr(e, 'status_code', None) or getattr(e, 'code', 'Unknown')
            logging.error(f"Failed to generate with {provider} API. Error code: {error_code}. Falling back to the next API.")
            continue

    raise RuntimeError("All API calls failed. Unable to generate a response.")
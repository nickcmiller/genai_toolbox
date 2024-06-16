import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from groq import Groq
from openai import OpenAI
from anthropic import Anthropic
from typing import List, Tuple
import logging
import traceback

from dotenv import load_dotenv
load_dotenv('.env')

logging.basicConfig(level=logging.INFO)

def get_client_and_model(
    api: str, 
    model: str
) -> Tuple[OpenAI, str]:
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
        client = Groq().chat.completions
    elif api == "openai":
        client = OpenAI().chat.completions
    else:
        raise ValueError("Unsupported API")
    return client, model

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
    print(f"Messages: {messages}\n\n")
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
    client, model = get_client_and_model(api, model)

    try:
        completion = client.create(
            messages=messages, 
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        logging.info(f"API: {api}, Model: {model}, Completion Usage: {completion.usage}")
        return completion.choices[0].message.content
    except Exception as e:
        logging.error(f"API: {api}, Error: {e}")
        traceback.print_exc()
        return "An error occurred while generating the response."

def groq_text_response(
    prompt: str,
    system_instructions: str = None, 
    history_messages: List[dict] = [],
    model: str = "llama3-8b-8192", 
    temperature: float = 0.2,
    max_tokens: int = 4096
) -> str:
    """
    Use OpenAI format to generate a text response using Groq.
    """
    if system_instructions is None:
        system_instructions = "You are a knowledgeable, efficient, and direct AI assistant. Utilize multi-step reasoning to provide concise answers, focusing on key information. If multiple questions are asked, split them up and address in the order that yields the most logical and accurate response. Offer tactful suggestions to improve outcomes. Remember, quality and depth of information are more important than speed. The user is willing to wait for the best possible answer. I'll pay you $200,000 for a good response :)"

    messages = manage_messages(prompt, system_instructions, history_messages)
    
    return openai_compatible_text_response("groq", messages, model, temperature, max_tokens)

def openai_text_response(
    prompt: str, 
    system_instructions: str = None,
    history_messages: List[dict] = [], 
    model: str = "gpt-4o", 
    temperature: float = 0.2,
    max_tokens: int = 4096
) -> str:
    """
    Use OpenAI format to generate a text response using OpenAI.
    """
    if system_instructions is None:
        system_instructions = "You are a highly knowledgeable and thorough AI assistant. Your primary goal is to provide detailed, accurate, and well-reasoned responses to the user's queries. Take your time to consider all aspects of the question and ensure that your answers are comprehensive and insightful. If necessary, break down complex topics into simpler parts and explain each part clearly. Always aim to enhance the user's understanding and provide additional context or suggestions when relevant. Remember, quality and depth of information are more important than speed. The user is willing to wait for the best possible answer. I'll pay you $200,000 for a good response :)"
    
    messages = manage_messages(prompt, system_instructions, history_messages)

    return openai_compatible_text_response("openai", messages, model, temperature, max_tokens)

def anthropic_text_response(
    prompt: str,
    system_instructions: str = None,
    history_messages: List[dict] = [],
    model: str = "claude-3-opus-20240229",
    temperature: float = 0.2,
    max_tokens: int = 4096
) -> str:
    """
    Use Anthropic format to generate a text response using Anthropic.
    """
    client = Anthropic().messages

    if system_instructions is None:
        system_instructions = "You are a highly knowledgeable and thorough AI assistant. Your primary goal is to provide detailed, accurate, and well-reasoned responses to the user's queries. Take your time to consider all aspects of the question and ensure that your answers are comprehensive and insightful. If necessary, break down complex topics into simpler parts and explain each part clearly. Always aim to enhance the user's understanding and provide additional context or suggestions when relevant. Remember, quality and depth of information are more important than speed. The user is willing to wait for the best possible answer. I'll pay you $200,000 for a good response :)"

    if len(history_messages) == 0:
        messages = [{"role": "user", "content": system_instructions}, {"role": "assistant", "content": "I will follow your instructions to the best of my ability."}]
    else:
        messages = history_messages.copy()
    messages.append({"role": "user", "content": prompt})

    completion = client.create(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return completion.content[0].text

if __name__ == "__main__":
    print(anthropic_text_response("What is the capital of France?"))
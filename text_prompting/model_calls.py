import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from groq import Groq
from openai import OpenAI
from anthropic import Anthropic

from enum import Enum
from typing import List, Tuple, Optional
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
        "mixtral-8x7b": "mixtral-8x7b-32768",
        "gemma": "gemma-7b-it"
    }
    if model_choice not in model_choices:
        raise ValueError(f"Invalid model_choice. Available options: {list(model_choices.keys())}")

    model = model_choices[model_choice]

    default_system_instructions = "You are a knowledgeable, efficient, and direct AI assistant. Utilize multi-step reasoning to provide concise answers, focusing on key information. If multiple questions are asked, split them up and address in the order that yields the most logical and accurate response. Offer tactful suggestions to improve outcomes. Remember, quality and depth of information are more important than speed. The user is willing to wait for the best possible answer. I'll pay you $200,000 for a good response :)"
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
        "4o": "gpt-4o",
        "4": "gpt-4-turbo",
        "3.5": "gpt-3.5-turbo",
    }
    if model_choice not in model_choices:
        raise ValueError(f"Invalid model_choice. Available options: {list(model_choices.keys())}")

    model = model_choices[model_choice]

    default_system_instructions = "You are a highly knowledgeable and thorough AI assistant. Your primary goal is to provide detailed, accurate, and well-reasoned responses to the user's queries. Take your time to consider all aspects of the question and ensure that your answers are comprehensive and insightful. If necessary, break down complex topics into simpler parts and explain each part clearly. Always aim to enhance the user's understanding and provide additional context or suggestions when relevant. Remember, quality and depth of information are more important than speed. The user is willing to wait for the best possible answer. I'll pay you $200,000 for a good response :)"
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
    client = Anthropic().messages

    model_choices = {
        "opus": "claude-3-opus-20240229",
        "haiku": "claude-3-haiku-20240307",
        "sonnet": "claude-3-sonnet-20240229",
    }
    if model_choice not in model_choices:
        raise ValueError(f"Invalid model_choice. Available options: {list(model_choices.keys())}")
    model = model_choices[model_choice]

    default_system_instructions = "You are a highly knowledgeable and thorough AI assistant. Your primary goal is to provide detailed, accurate, and well-reasoned responses to the user's queries. Take your time to consider all aspects of the question and ensure that your answers are comprehensive and insightful. If necessary, break down complex topics into simpler parts and explain each part clearly. Always aim to enhance the user's understanding and provide additional context or suggestions when relevant. Remember, quality and depth of information are more important than speed. The user is willing to wait for the best possible answer. I'll pay you $200,000 for a good response :)"
    system_instructions = system_instructions if system_instructions is not None else default_system_instructions
    history_messages = history_messages if history_messages is not None else []

    messages = [{"role": "user", "content": system_instructions}, {"role": "assistant", "content": "I will follow your instructions to the best of my ability."}] if not history_messages else history_messages.copy()
    messages.append({"role": "user", "content": prompt})

    try:
        completion = client.create(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return completion.content[0].text
    except Exception as e:
        logging.error(f"Failed to generate response with Anthropic: {e}")
        raise RuntimeError("Failed to generate response due to an internal error.")

class Provider(Enum):
    OPENAI = ("openai", "4o", openai_text_response)
    ANTHROPIC = ("anthropic", "opus", anthropic_text_response)
    GROQ = ("groq", "llama3-70b", groq_text_response)

    def __init__(self, provider_name, default_model, function):
        self.provider_name = provider_name
        self.default_model = default_model
        self.function = function

    def get_info(self):
        return {
            "default_model": self.default_model,
            "function": self.function
        }

def prompt_string_list(
    string_list: List[str],
    instructions: str,
    provider: Provider = Provider.OPENAI,
    model_choice: Optional[str] = None,
    system_instructions: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 4096
) -> List[str]:
    modified_list = []
    provider_info = provider.get_info()

    model_choice = model_choice if model_choice is not None else provider_info["default_model"]
    logging.info(f"Prompting {provider.provider_name} with model {model_choice} for {len(string_list)} strings")

    for index, s in enumerate(string_list, start=1):
        try:
            logging.info(f"Prompting string {index} of {len(string_list)}")
            prompt_string = instructions.format(s)
            response = provider_info["function"](
                prompt=prompt_string,
                system_instructions=system_instructions,
                temperature=temperature,
                max_tokens=max_tokens,
                model_choice=model_choice
            )
            modified_list.append(response)
        except Exception as e:
            logging.error(f"Error prompting {provider.provider_name} with model {model_choice}: {e}")
            traceback.print_exc()
            raise

    return modified_list


if __name__ == "__main__":
    country_list = ["United States", "United Kingdom"]
    instructions = "What is the capital of {}?"
    response = prompt_string_list(country_list, instructions, system_instructions="You are a helpful assistant that can answer questions about the capital of countries.", provider=Provider.OPENAI, model_choice="4o", max_tokens=100)
    print(response)
import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from text_prompting.model_calls import openai_text_response, anthropic_text_response, groq_text_response

from enum import Enum
from typing import List, Optional
import logging
import traceback

from dotenv import load_dotenv
load_dotenv(os.path.join(root_dir, '.env'))

logging.basicConfig(level=logging.INFO)

class Provider(Enum):
    OPENAI = ("openai", "4o", openai_text_response)
    ANTHROPIC = ("anthropic", "opus", anthropic_text_response)
    GROQ = ("groq", "llama3-70b", groq_text_response)

    def __init__(self, provider_name, default_model, function):
        self.provider_name = provider_name
        self.default_model = default_model
        self.function = function

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

    model_choice = model_choice if model_choice is not None else provider.default_model
    logging.info(f"Prompting {provider.provider_name} with model {model_choice} for {len(string_list)} strings")

    for index, s in enumerate(string_list, start=1):
        try:
            logging.info(f"Prompting string {index} of {len(string_list)}")
            prompt_string = instructions.format(s)
            response = provider.function(
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
    country_list = ["United States", "Ukraine"]
    instructions = "What is the capital of {}?"
    response = prompt_string_list(country_list, instructions, system_instructions="You are a helpful assistant that can answer questions about the capital of countries.", provider=Provider.GROQ)
    print(response)
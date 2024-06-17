import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from text_prompting.model_calls import openai_text_response, anthropic_text_response, groq_text_response
from helper_functions.string_helpers import concatenate_list_text_to_list_text

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

    for index, input_string in enumerate(string_list, start=1):
        try:
            logging.info(f"Prompting string {index} of {len(string_list)}")
            prompt_string = instructions.format(input_string)
            response = provider.function(
                prompt=prompt_string,
                system_instructions=system_instructions if system_instructions is not None else "",
                temperature=temperature,
                max_tokens=max_tokens,
                model_choice=model_choice
            )
            modified_list.append(response)
        except Exception as e:
            logging.error(f"Error prompting {provider.provider_name} with model {model_choice}: {e}")
            traceback.print_exc()

    return modified_list

def revise_with_prompt_list(
    string_list: List[str],
    prompt_list: List[dict],
    delimiter: str = f"\n{'-'*10}\n"
) -> dict:


    def execute_prompt_dict(
        modified_strings: List[str],
        original_strings: List[str], 
        prompt: dict,
        delimiter: str
    ) -> List[str]:
        """
            Executes a prompt on a list of strings.

            Args:
                modified_strings (List[str]): The list of strings to be modified.
                original_strings (List[str]): The list of strings to be used as the original input.
                prompt (dict): The prompt to be executed.
                    - provider (Provider): The provider to be used to execute the prompt.
                    - instructions (str): The instructions to be used to execute the prompt.
                    - model_choice (str): The model to be used to execute the prompt.
                delimiter (str): The delimiter to be used to concatenate the strings.

            Returns:
                List[str]: The list of modified strings.
        """
        provider = prompt["provider"]
        instructions = prompt["instructions"] 
        model_choice = prompt.get("model_choice")
        
        modified_strings = concatenate_list_text_to_list_text(
            modified_strings, 
            original_strings, 
            delimiter=delimiter
        )
        return prompt_string_list(
            string_list=modified_strings,
            provider=provider,
            instructions=instructions,
            model_choice=model_choice
        )

    revision_dict = {"Original": string_list}
    modified_strings = [""] * len(string_list)
    
    for count, prompt in enumerate(prompt_list, start=1):
        try:
            logging.info(f"\n{'#' * 10}\nExecuting prompt {count}\n{'#' * 10}\n")
            modified_strings = execute_prompt_dict(
                modified_strings=modified_strings, 
                original_strings=revision_dict['Original'], 
                prompt=prompt, 
                delimiter=delimiter
            )
            revision_dict[f"Revision {count}"] = modified_strings
        except Exception as e:
            logging.error(f"Failed to generate summary for prompt {count}: {e}")
            logging.error(traceback.format_exc()) 
            continue
        
    return {
        "modified_string_list": modified_strings, 
        "revision_dict": revision_dict
    }


if __name__ == "__main__":
    list_text = ["Pacers", "Bulls"]
    prompt_list = [
        {"provider": Provider.GROQ, "instructions": "Who is the coach of {}?", "model_choice": "llama3-70b", "system_instructions": "You are a helpful assistant that can answer questions about the coach of NBA teams."},
        {"provider": Provider.GROQ, "instructions": "What is the experience of the coach? \nPrior info: {}", "model_choice": "llama3-70b", "system_instructions": "You are a helpful assistant that can answer questions about the experience of NBA coaches."},
        {"provider": Provider.GROQ, "instructions": "Who is the GM the coach reports to? \nPrior info: {}", "model_choice": "llama3-70b", "system_instructions": "You are a helpful assistant that can answer questions about the GM of NBA teams."}
    ]
    response = revise_with_prompt_list(list_text, prompt_list)
    for revision in response["revision_dict"]:
        print(f"\n{'#'*10}\n{revision}\n{'#'*10}\n")
        for r in response["revision_dict"][revision]:
            print(r)
            print(f"\n{'-'*10}\n")
   
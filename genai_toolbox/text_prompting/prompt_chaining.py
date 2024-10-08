
from enum import Enum
from typing import List, Dict, Optional
import os
import logging
import traceback

from genai_toolbox.text_prompting.model_calls import openai_text_response, anthropic_text_response, groq_text_response, fallback_text_response
from genai_toolbox.helper_functions.string_helpers import concatenate_list_text_to_list_text

logging.basicConfig(level=logging.INFO)

default_temperature = 0.2
default_max_tokens = 4096

def prompt_string_list(
    string_list: List[str],
    instructions: str,
    model_order: List[Dict],
    system_instructions: Optional[str] = None,
    temperature: float = default_temperature,
    max_tokens: int = default_max_tokens
) -> List[str]:
    """
        Prompts a list of strings with given instructions using a specified provider and model.

        This function takes a list of strings and applies a set of instructions to each string
        using a specified AI provider (e.g., OpenAI, Anthropic, Groq). It processes each string
        individually and returns a list of modified strings based on the applied instructions.

        Args:
            string_list (List[str]): The list of input strings to be processed.
            instructions (str): The instructions to be applied to each string.
            model_order (List[Dict]): The order of the models to be used.
            system_instructions (Optional[str], optional): Additional system-level instructions for the AI model.
            temperature (float, optional): The randomness of the model's output. Defaults to 0.2.
            max_tokens (int, optional): The maximum number of tokens in the response. Defaults to 4096.

        Returns:
            List[str]: A list of processed strings after applying the instructions.

        Raises:
            Exception: Logs any errors that occur during the processing of individual strings.

        Example:
            input_strings = [
                "The quick brown fox jumps over the lazy dog.",
                "To be or not to be, that is the question."
            ]
            instructions = "Translate the following text to French: {}"
            
            translated_strings = prompt_string_list(
                string_list=input_strings,
                instructions=instructions,
                model_order=[
                    {
                        "provider": "openai", 
                        "model": "4o-mini"
                    }
                ],
                temperature=0.3
            )
            
            >>> print(translated_strings)
            [
                "Le rapide renard brun saute par-dessus le chien paresseux.",
                "Être ou ne pas être, telle est la question."
            ]
    """
    modified_list = []

    for index, input_string in enumerate(string_list, start=1):
        try:
            logging.info(f"Prompting string {index} of {len(string_list)}")
            prompt_string = instructions.format(input_string)
            response = fallback_text_response(
                prompt=prompt_string,
                system_instructions=system_instructions,
                model_order=model_order
            )
            modified_list.append(response)
        except Exception as e:
            logging.error(f"Error prompting string {index} of {len(string_list)}: {e}")
            traceback.print_exc()
    return modified_list

def execute_prompt_dict(
    modified_strings: List[str],
    original_strings: List[str], 
    prompt_dict: dict,
    concatenation_delimiter: str = f"\n{'-'*10}\n",
    temperature: float = default_temperature,
    max_tokens: int = default_max_tokens
) -> List[str]:
    """
        Execute a single prompt dictionary on a list of modified and original strings.

        This function processes the given strings using the specified prompt configuration,
        concatenates the modified and original strings, and then applies the prompt to the result.

        Args:
            modified_strings (List[str]): A list of previously modified strings.
            original_strings (List[str]): A list of original input strings.
            prompt_dict (dict): A dictionary containing prompt configuration:
                - provider (str): The name of the AI provider (e.g., "openai").
                - instructions (str): The instructions for the prompt.
                - model_order (List[Dict]): The order of the models to be used.
            concatenation_delimiter (str): The delimiter used to separate concatenated strings.

        Returns:
            List[str]: A list of strings after applying the prompt.

        Example:
            modified_strings = ["Summary of chapter 1", "Summary of chapter 2"]
            original_strings = ["Full text of chapter 1", "Full text of chapter 2"]
            prompt = {
                "instructions": "Enhance the summary with more details from the original text.",
                "model_order": [
                    {
                        "provider": "openai", 
                        "model": "4o-mini"
                    }
                ]
            }
            concatenation_delimiter = "\n---\n"
            
            >>> print(execute_prompt_dict(modified_strings, original_strings, prompt, delimiter))
            [
                "Enhanced summary of chapter 1 with more details", 
                "Enhanced summary of chapter 2 with more details"
            ]

        Raises:
            ValueError: If the prompt dictionary is missing required keys.
            Exception: Any exception raised during the execution of the prompt.
    """
    instructions = prompt_dict["instructions"] 
    model_order = prompt_dict["model_order"]
    system_instructions = prompt_dict.get("system_instructions", None)
    
    modified_strings = concatenate_list_text_to_list_text(
        modified_strings, 
        original_strings, 
        delimiter=concatenation_delimiter
    )

    return prompt_string_list(
        string_list=modified_strings,
        instructions=instructions,
        model_order=model_order,
        system_instructions=system_instructions,
        temperature=temperature,
        max_tokens=max_tokens
    )

def revise_list_with_prompt_list(
    string_list: List[str],
    prompt_list: List[dict],
    concatenation_delimiter: str = f"\n{'-'*10}\n",
    temperature: float = default_temperature,
    max_tokens: int = default_max_tokens
) -> dict:
    """
        Revise a list of strings using a series of prompts.

        This function takes a list of strings and applies a series of prompts to revise them.
        It keeps track of each revision and returns both the final modified strings and a
        dictionary containing all revisions.

        Args:
            string_list (List[str]): The initial list of strings to be revised.
            prompt_list (List[dict]): A list of prompt dictionaries, each containing
                instructions for a revision step.
            concatenation_delimiter (str, optional): The delimiter used to separate concatenated strings.
                Defaults to a line of 10 dashes.

        Returns:
            dict: A dictionary containing:
                - 'modified_string_list': The final revised list of strings.
                - 'revision_dict': A dictionary with all revision steps, including the original.

        Raises:
            Exception: Any exception raised during the execution of a prompt is caught,
                logged, and the function continues with the next prompt.

        Example:
            string_list = ["Initial text 1", "Initial text 2"]
            prompt_list = [
                {"provider": "openai", "instructions": "Summarize the text"},
                {"provider": "openai", "instructions": "Add more details"}
            ]
            result = revise_with_prompt_list(string_list, prompt_list)
            >>> print(result['modified_string_list'])
            ["Summarized text 1", "Summarized text 2"]
            >>> print(result['revision_dict'])
            {
                "Original": ["Initial text 1", "Initial text 2"],
                "Revision 1": ["Summarized text 1", "Summarized text 2"],
                "Revision 2": ["Enhanced summary 1", "Enhanced summary 2"]
            }
    """

    revision_dict = {"Original": string_list}
    modified_strings = [""] * len(string_list)
    
    for count, prompt_dict in enumerate(prompt_list, start=1):
        try:
            logging.info(f"\n{'#' * 10}\nExecuting prompt {count}\n{'#' * 10}\n")
            modified_strings = execute_prompt_dict(
                modified_strings=modified_strings, 
                original_strings=revision_dict['Original'], 
                prompt_dict=prompt_dict, 
                concatenation_delimiter=concatenation_delimiter,
                temperature=temperature,
                max_tokens=max_tokens
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

def revise_string_with_prompt_list(
    string: str,
    prompt_list: List[dict],
    concatenation_delimiter: str = f"\n{'-'*10}\n",
    temperature: float = default_temperature,
    max_tokens: int = default_max_tokens
) -> dict:
    
    response = revise_list_with_prompt_list(
        string_list=[string],
        prompt_list=prompt_list,
        concatenation_delimiter=concatenation_delimiter,
        temperature=temperature,
        max_tokens=max_tokens
    )
    revision_dict = response["revision_dict"]
    new_revision_dict = {}
    
    for k, v in revision_dict.items():
        new_revision_dict[k] = v[0]

    return {
        "modified_string": response["modified_string_list"][0],
        "revision_dict": new_revision_dict
    }


if __name__ == "__main__":
    list_text = ["Pacers", "Bulls"]
    prompt_list = [
        {
            "instructions": "Who is the coach of {}?", 
            "model_order": [
                {
                    "provider": "groq", 
                    "model": "llama3-70b"
                },
                {
                    "provider": "openai", 
                    "model": "4o-mini"
                }
            ], 
            "system_instructions": "You are a helpful assistant that can answer questions about the coach of NBA teams."
        },
        {
            "instructions": "What is the experience of the coach? \nPrior info: {}", 
            "model_order": [
                {
                    "provider": "groq", 
                    "model": "llama3-70b"
                },
                {
                    "provider": "openai", 
                    "model": "4o-mini"
                }
            ], 
            "system_instructions": "You are a helpful assistant that can answer questions about the experience of NBA coaches."
        },
        {
            "instructions": "Who is the GM the coach reports to? \nPrior info: {}", 
            "model_order": [
                {
                    "provider": "groq", 
                    "model": "llama3-70b"
                },
                {
                    "provider": "openai", 
                    "model": "4o-mini"
                }
            ], 
            "system_instructions": "You are a helpful assistant that can answer questions about the GM of NBA teams."
        }
    ]
    response = revise_with_prompt_list(list_text, prompt_list)
    for revision in response["revision_dict"]:
        print(f"\n{'#'*10}\n{revision}\n{'#'*10}\n")
        for r in response["revision_dict"][revision]:
            print(r)
            print(f"\n{'-'*10}\n")
   
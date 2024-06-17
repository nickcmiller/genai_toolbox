from typing import List
import logging

logging.basicConfig(level=logging.INFO)

def concatenate_list_text_to_list_text(
    first_list: List[str],
    second_list: List[str],
    delimiter: str = f"\n{'-'*10}\n",
) -> List[str]:
    """
    Concatenates each element of the second list to the corresponding element of the first list,
    separated by the specified delimiter. Returns a new list with the concatenated text.

    Args:
        first_list (List[str]): The first list of strings.
        second_list (List[str]): The second list of strings.
        delimiter (str): The delimiter to separate the elements of the first and second lists.

    Returns:
        List[str]: A new list with the modified text.

    Raises:
        ValueError: If the input lists are not of the same length.
    """
    if not all(isinstance(i, str) for i in first_list) or not all(isinstance(i, str) for i in second_list):
        logging.error("Both input lists should contain only strings.")
        return None

    if len(first_list) != len(second_list):
        logging.error("Input lists must be of the same length.")
        return None
    
    concatenated_list = []
    for i in range(len(first_list)):
        combined_text = f"{first_list[i]}{delimiter}{second_list[i]}"
        concatenated_list.append(combined_text)
    
    return concatenated_list
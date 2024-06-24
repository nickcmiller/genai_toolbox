import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from clients.openai_client import openai_client
from typing import Dict, Callable

def create_openai_embedding(
    model_choice: str,
    text: str, 
    client = openai_client()
) -> Dict:
    response = client.embeddings.create(
        input=text, 
        model=model_choice
    )
    return response.data[0].embedding

def create_embedding_for_dict(
    embedding_function: Callable,
    chunk_dict: dict,
    key_to_embed: str = "text",
    model_choice: str = "text-embedding-3-small"
) -> dict:
    """
        Creates an embedding for the text in the given dictionary using the specified model and retains all other key-value pairs.

        Args:
        chunk_dict (dict): A dictionary containing the text to embed under the key 'text' and possibly other data.
        embedding_function (Callable): The embedding function used to create embeddings.
        model_choice (str): The model identifier to use for embedding generation.

        Returns:
        dict: A dictionary containing the original text, its corresponding embedding, and all other key-value pairs from the input dictionary.
    """
    if 'text' not in chunk_dict:
        raise KeyError("The 'text' key is missing from the chunk_dict.")
    
    if not chunk_dict[key_to_embed]:
        raise ValueError(f"The '{key_to_embed}' value in chunk_dict is empty.")

    if not isinstance(embedding_function, Callable):
        raise ValueError("The 'embedding_function' argument must be a callable object.")

    text = chunk_dict[key_to_embed]
    embedding = embedding_function(
        model_choice=model_choice,
        text=text
    )
    result_dict = {**chunk_dict, "embedding": embedding}
    return result_dict

def embed_dict_list(
    embedding_function: Callable,
    chunk_dicts: list[dict],
    key_to_embed: str = "text",
    model_choice: str = "text-embedding-3-small"
) -> list[dict]:
    """
        Creates embeddings for a list of dictionaries containing text.

        This function applies the specified embedding function to each dictionary in the input list,
        embedding the text found under the specified key in each dictionary.

        Args:
        embedding_function (Callable): The function used to create embeddings.
        chunk_dicts (list[dict]): A list of dictionaries, each containing text to be embedded.
        key_to_embed (str, optional): The key in each dictionary that contains the text to embed. Defaults to "text".
        model_choice (str, optional): The model identifier to use for embedding generation. Defaults to "text-embedding-3-small".

        Returns:
        list[dict]: A list of dictionaries, each containing the original data plus the generated embedding.

        Raises:
        ValueError: If the embedding_function is not callable, if the key_to_embed is missing from any dictionary,
                    or if the value for key_to_embed is empty in any dictionary.
    """
    return [create_embedding_for_dict(
        embedding_function=embedding_function, 
        chunk_dict=chunk_dict, 
        key_to_embed=key_to_embed,
        model_choice=model_choice
    ) for chunk_dict in chunk_dicts]

def cosine_similarity(
    vec1: list[float], 
    vec2: list[float]
):
    """
        Calculates the cosine similarity between two vectors.

        This function computes the cosine similarity between two vectors, which is a measure of
        the cosine of the angle between them. It is often used to determine how similar two vectors are,
        regardless of their magnitude.

        Args:
        vec1 (list[float]): The first vector, represented as a list of floats.
        vec2 (list[float]): The second vector, represented as a list of floats.

        Returns:
        float: The cosine similarity between vec1 and vec2. The value ranges from -1 to 1,
            where 1 indicates identical vectors, 0 indicates orthogonal vectors,
            and -1 indicates opposite vectors.

        Raises:
        ValueError: If the input vectors have different lengths.

        Note:
        This function uses numpy for efficient vector operations.
    """
    import numpy as np
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))




if __name__ == "__main__":
    print(create_openai_embedding(
        "text-embedding-3-large",
        "Hello, world!", 
    ))
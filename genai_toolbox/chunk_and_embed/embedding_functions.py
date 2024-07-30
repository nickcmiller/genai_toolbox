from genai_toolbox.clients.openai_client import openai_client

import tiktoken

from typing import Dict, Callable
import concurrent.futures
import logging

# Similarity Metrics
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

# Embedding Clients
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

# Embedding Functions
def num_tokens_from_string(
    string: str, 
    encoding_name: str = "o200k_base"
) -> int:
    """
        num_tokens_from_string

        This function calculates the number of tokens in a given string based on a specified encoding.

        Parameters:
        - string (str): The input string for which the number of tokens is to be calculated.
        - encoding_name (str, optional): The name of the encoding to be used for tokenization. 
                                          Defaults to "o200k_base". This encoding is typically used 
                                          for OpenAI models and may vary based on the model's requirements.

        Returns:
        - int: The total number of tokens in the input string. The token count is determined by 
               encoding the string and counting the resulting tokens.

        Raises:
        - ValueError: If the encoding_name is not recognized or if there is an issue with the encoding process.

        Example:
        >>> token_count = num_tokens_from_string("Hello, world!")
        >>> print(token_count)
        4  # Example output, actual token count may vary based on encoding

        Note:
        This function utilizes the `tiktoken` library to perform the encoding and token counting. 
        It is important to choose the correct encoding that matches the model being used for 
        generating embeddings or processing text.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def create_embedding_for_dict(
    embedding_function: Callable,
    chunk_dict: dict,
    key_to_embed: str = "text",
    model_choice: str = "text-embedding-3-small"
) -> dict:
    """
        create_embedding_for_dict

        This function generates an embedding for a given dictionary using a specified embedding function.

        Parameters:
        - embedding_function (Callable): A callable function that takes a model choice and text as input 
                                        and returns the corresponding embedding. This function is expected 
                                        to handle the embedding process, typically by calling an external 
                                        API or library.
        - chunk_dict (dict): A dictionary containing the data for which the embedding is to be created. 
                            It must include the text to be embedded under the specified key.
        - key_to_embed (str, optional): The key in the dictionary that contains the text to be embedded. 
                                        Defaults to "text".
        - model_choice (str, optional): The identifier for the embedding model to be used. Defaults to 
                                        "text-embedding-3-small".

        Returns:
        - dict: A new dictionary that includes all the original key-value pairs from chunk_dict, 
                along with a new key "embedding" that contains the generated embedding.

        Raises:
        - KeyError: If the specified key_to_embed is not found in chunk_dict.
        - ValueError: If the embedding_function is not callable.

        Logging:
        - If the text to be embedded is empty or consists only of whitespace, a warning is logged 
        indicating that the chunk_dict is being skipped.

        Example:
        >>> chunk = {"text": "This is a sample text."}
        >>> embedding_result = create_embedding_for_dict(create_openai_embedding, chunk)
        >>> print(embedding_result)
        {
            "text": "This is a sample text.",
            "embedding": [0.1, 0.2, 0.3, ...]  # Example embedding vector
        }
    """
    if key_to_embed not in chunk_dict:
        raise KeyError(f"The '{key_to_embed}' key is missing from the chunk_dict.")
    
    text = chunk_dict[key_to_embed]
    if not text or not text.strip():
        # Log the chunk_dict that's not being processed due to empty text
        logging.warning(f"Skipping chunk_dict due to empty text: {chunk_dict}")
        return None

    if not isinstance(embedding_function, Callable):
        raise ValueError("The 'embedding_function' argument must be a callable object.")

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
    model_choice: str = "text-embedding-3-small",
    max_workers: int = 25
) -> list[dict]:
    """
        embed_dict_list function

        This function takes a list of dictionaries and generates embeddings for each dictionary 
        using the specified embedding function. It utilizes a ThreadPoolExecutor to perform 
        the embedding process concurrently, allowing for efficient processing of multiple 
        dictionaries at once.

        Parameters:
            embedding_function (Callable): A callable function that generates embeddings. 
                                            This function should accept a model_choice and text 
                                            as parameters and return the corresponding embedding.
            chunk_dicts (list[dict]): A list of dictionaries, each containing the data for which 
                                    embeddings need to be generated. Each dictionary should 
                                    include a key specified by the key_to_embed parameter.
            key_to_embed (str, optional): The key in each dictionary whose value will be used 
                                        for generating the embedding. Defaults to "text".
            model_choice (str, optional): The identifier for the embedding model to be used. 
                                        Defaults to "text-embedding-3-small".
            max_workers (int, optional): The maximum number of threads to use for concurrent 
                                        processing. Defaults to 25.

        Returns:
            list[dict]: A list of dictionaries, each containing the original key-value pairs 
                        from the input dictionaries along with a new key "embedding" that 
                        contains the generated embedding.

        Raises:
            ValueError: If the embedding_function is not callable.

        Logging:
            - If any chunk_dict is not processed due to an error, a warning is logged.

        Example:
            >>> chunks = [{"text": "This is a sample text."}, {"text": "Another sample text."}]
            >>> embeddings = embed_dict_list(create_openai_embedding, chunks)
            >>> print(embeddings)
            [
                {"text": "This is a sample text.", "embedding": [0.1, 0.2, 0.3, ...]},
                {"text": "Another sample text.", "embedding": [0.4, 0.5, 0.6, ...]}
            ]
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                create_embedding_for_dict,
                embedding_function=embedding_function,
                chunk_dict=chunk_dict,
                key_to_embed=key_to_embed,
                model_choice=model_choice
            )
            for chunk_dict in chunk_dicts
        ]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    return [result for result in results if result is not None]

def add_similarity_to_next_dict_item(
    chunk_dicts: list[dict],
    similarity_metric: Callable = cosine_similarity
) -> list[dict]:
    """
        Adds a 'similarity_to_next_item' key to each dictionary in the list,
        calculating the cosine similarity between the current item's embedding
        and the next item's embedding. The last item's similarity is always 0.

        Args:
            chunk_dicts (list[dict]): List of dictionaries containing 'embedding' key.

        Returns:
            list[dict]: The input list with added 'similarity_to_next_item' key for each dict.

        Example:
            Input:
            [
                {..., "embedding": [0.1, 0.2, 0.3]},
                {..., "embedding": [0.4, 0.5, 0.6]},
            ]
            Output:
            [
                {..., "embedding": [0.1, 0.2, 0.3], "similarity_to_next_item": 0.9},
                {..., "embedding": [0.4, 0.5, 0.6], "similarity_to_next_item": 0.9},
            ]
    """
    for i in range(len(chunk_dicts) - 1):
        current_embedding = chunk_dicts[i]['embedding']
        next_embedding = chunk_dicts[i + 1]['embedding']
        similarity = cosine_similarity(current_embedding, next_embedding)
        chunk_dicts[i]['similarity_to_next_item'] = similarity

    # similarity_to_next_item for the last item is always 0
    chunk_dicts[-1]['similarity_to_next_item'] = 0

    return chunk_dicts

# Query embeddings
def find_similar_chunks(
    query: str,
    chunks_with_embeddings: list[dict], 
    embedding_function: Callable = create_openai_embedding,
    model_choice: str = "text-embedding-3-large",
    threshold: float = 0.4,
    max_returned_chunks: int = 10,
) -> list[dict]:
    query_embedding = embedding_function(text=query, model_choice=model_choice)

    similar_chunks = []
    for chunk in chunks_with_embeddings:
        similarity = cosine_similarity(query_embedding, chunk['embedding'])
        if similarity > threshold:
            chunk['similarity'] = similarity
            similar_chunks.append(chunk)
        
    similar_chunks.sort(key=lambda x: x['similarity'], reverse=True)
    top_chunks = similar_chunks[0:max_returned_chunks]
    no_embedding_key_chunks = [{k: v for k, v in chunk.items() if k != 'embedding'} for chunk in top_chunks]
    
    return no_embedding_key_chunks

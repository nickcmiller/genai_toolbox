from genai_toolbox.clients.openai_client import openai_client

import tiktoken

from typing import Dict, Callable, Optional, List
import concurrent.futures
import logging
import time
import asyncio

class RateLimiter:
    """
        RateLimiter

        A class to manage the rate limiting of API calls or other operations that require 
        controlling the frequency of execution. This is particularly useful in scenarios 
        where you need to adhere to a specific rate limit imposed by an external service.

        Attributes:
            rate_limit (int): The maximum number of operations allowed within the specified time unit.
            time_unit (int): The time period (in seconds) over which the rate limit is applied.
            tokens (float): The current number of available tokens, which represent the number of operations 
                            that can be performed.
            last_refill (float): The timestamp of the last time the tokens were refilled.

        Methods:
            wait_for_token():
                Asynchronously waits for a token to become available, ensuring that the rate limit is not exceeded.
                If no tokens are available, it will pause execution until a token can be obtained.
    """
    def __init__(self, rate_limit, time_unit=60):
        self.rate_limit = rate_limit
        self.time_unit = time_unit
        self.tokens = rate_limit
        self.last_refill = time.time()

    async def wait_for_token(self):
        current_time = time.time()
        time_passed = current_time - self.last_refill
        self.tokens += time_passed * (self.rate_limit / self.time_unit)
        self.tokens = min(self.tokens, self.rate_limit)
        self.last_refill = current_time

        if self.tokens < 1:
            await asyncio.sleep(1 / (self.rate_limit / self.time_unit))
            return await self.wait_for_token()

        self.tokens -= 1
        return

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
    """
        create_openai_embedding

        This function creates an embedding for a given text using the OpenAI API.
    """
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
    model_choice: str = "text-embedding-3-large"
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

async def create_embedding_for_dict_async(
    embedding_function: Callable,
    chunk_dict: dict,
    key_to_embed: str = "text",
    model_choice: str = "text-embedding-3-large",
    rate_limiter: RateLimiter = None,
    metadata_keys: Optional[List[str]] = None
) -> dict:
    """
        Asynchronously creates an embedding for a given dictionary using the specified embedding function.

        This function takes a dictionary (chunk_dict) and generates an embedding for the text specified by 
        the key_to_embed. It also handles rate limiting to ensure that the number of API calls does not exceed 
        the specified limit.

        Args:
            embedding_function (Callable): A callable function that generates embeddings. This function should 
                                        accept the parameters model_choice and text.
            chunk_dict (dict): A dictionary containing the data from which the embedding will be created. 
                            It must include the key specified by key_to_embed.
            key_to_embed (str, optional): The key in chunk_dict whose value will be embedded. Defaults to "text".
            model_choice (str, optional): The identifier for the embedding model to be used. Defaults to 
                                        "text-embedding-3-large".
            rate_limiter (RateLimiter, optional): An instance of RateLimiter to manage API call frequency. 
                                                If None, no rate limiting will be applied.
            metadata_keys (list[str], optional): A list of keys in chunk_dict to be appended to the text.
                                        Defaults to None.

        Returns:
            dict: A new dictionary that includes all the original key-value pairs from chunk_dict, 
                along with a new key "embedding" that contains the generated embedding.

        Raises:
            KeyError: If the specified key_to_embed is not found in chunk_dict.
            ValueError: If the embedding_function is not callable.

        Logging:
            If the text to be embedded is empty or consists only of whitespace, a warning is logged 
            indicating that the chunk_dict is being skipped.

        Example:
            >>> chunk = {"text": "This is a sample text."}
            >>> embedding_result = await create_embedding_for_dict_async(create_openai_embedding, chunk)
            >>> print(embedding_result)
            {
                "text": "This is a sample text.",
                "embedding": [0.1, 0.2, 0.3, ...]  # Example embedding vector
            }
    """
    if rate_limiter:
        await rate_limiter.wait_for_token()

    if key_to_embed not in chunk_dict:
        raise KeyError(f"The '{key_to_embed}' key is missing from the chunk_dict.")
    
    text = chunk_dict[key_to_embed]
    if not text or not text.strip():
        logging.warning(f"Skipping chunk_dict due to empty text: {chunk_dict}")
        return None

    if metadata_keys:
        metadata_lines = "\n".join(f"{key}: {chunk_dict.get(key, '')}" for key in metadata_keys if key in chunk_dict)
        if metadata_lines:
            text += "\n" + metadata_lines

    if not isinstance(embedding_function, Callable):
        raise ValueError("The 'embedding_function' argument must be a callable object.")

    embedding = await asyncio.to_thread(
        embedding_function,
        model_choice=model_choice,
        text=text
    )

    result_dict = {**chunk_dict, "embedding": embedding}
    return result_dict

async def embed_dict_list_async(
    embedding_function: Callable,
    chunk_dicts: list[dict],
    key_to_embed: str = "text",
    model_choice: str = "text-embedding-3-large",
    max_workers: int = 25,
    rate_limit: int = 5000,
    metadata_keys: Optional[List[str]] = None
) -> list[dict]:
    """
        Asynchronously embeds a list of dictionaries using the specified embedding function.

        This function takes a list of dictionaries and applies an embedding function to each dictionary's 
        specified key, generating embeddings for the text contained in that key. It utilizes a rate limiter 
        to ensure that the number of API calls does not exceed the specified limit.

        Args:
            embedding_function (Callable): A callable function that generates embeddings for the given text.
            chunk_dicts (list[dict]): A list of dictionaries, each containing the text to be embedded.
            key_to_embed (str, optional): The key in each dictionary that contains the text to be embedded. 
                                        Defaults to "text".
            model_choice (str, optional): The model to be used for generating embeddings. Defaults to 
                                        "text-embedding-3-large".
            max_workers (int, optional): The maximum number of concurrent workers for processing. Defaults to 25.
            rate_limit (int, optional): The maximum number of API calls allowed within a specified time unit. 
                                        Defaults to 5000.
            metadata_keys (list[str], optional): A list of keys in chunk_dict to be appended to the text.
                                        Defaults to None.

        Returns:
            list[dict]: A list of dictionaries, each containing the original data along with a new key "embedding" 
                        that contains the generated embedding.

        Raises:
            KeyError: If the specified key_to_embed is not found in any of the chunk_dicts.
            ValueError: If the embedding_function is not callable.

        Logging:
            If the text to be embedded is empty or consists only of whitespace, a warning is logged 
            indicating that the chunk_dict is being skipped.

        Example:
            >>> chunks = [{"text": "This is a sample text."}, {"text": "Another sample."}]
            >>> embeddings = await embed_dict_list_async(create_openai_embedding, chunks)
            >>> print(embeddings)
            [
                {"text": "This is a sample text.", "embedding": [0.1, 0.2, 0.3, ...]},
                {"text": "Another sample.", "embedding": [0.4, 0.5, 0.6, ...]}
            ]
    """
    rate_limiter = RateLimiter(rate_limit)
    
    async def process_chunk(chunk_dict):
        return await create_embedding_for_dict_async(
            embedding_function=embedding_function,
            chunk_dict=chunk_dict,
            key_to_embed=key_to_embed,
            model_choice=model_choice,
            rate_limiter=rate_limiter,
            metadata_keys=metadata_keys
        )

    tasks = [process_chunk(chunk_dict) for chunk_dict in chunk_dicts]
    results = await asyncio.gather(*tasks)
    
    return [result for result in results if result is not None]

def embed_dict_list(
    embedding_function: Callable,
    chunk_dicts: list[dict],
    key_to_embed: str = "text",
    model_choice: str = "text-embedding-3-large",
    max_workers: int = 25,
    rate_limit: int = 5000,
    metadata_keys: Optional[List[str]] = None
) -> list[dict]:
    return asyncio.run(embed_dict_list_async(
        embedding_function=embedding_function,
        chunk_dicts=chunk_dicts,
        key_to_embed=key_to_embed,
        model_choice=model_choice,
        max_workers=max_workers,
        rate_limit=rate_limit,
        metadata_keys=metadata_keys
    ))

# def embed_dict_list(
#     embedding_function: Callable,
#     chunk_dicts: list[dict],
#     key_to_embed: str = "text",
#     model_choice: str = "text-embedding-3-large",
#     max_workers: int = 25
# ) -> list[dict]:
#     with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = [
#             executor.submit(
#                 create_embedding_for_dict,
#                 embedding_function=embedding_function,
#                 chunk_dict=chunk_dict,
#                 key_to_embed=key_to_embed,
#                 model_choice=model_choice
#             )
#             for chunk_dict in chunk_dicts
#         ]
#         results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
#     return [result for result in results if result is not None]


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
    similarity_threshold: float = 0.30,
    filter_limit: int = 15,
    max_similarity_delta: float = 0.075,
) -> list[dict]:
    """
        Find similar chunks based on a query string by calculating the cosine similarity 
        between the query's embedding and the embeddings of the provided chunks.

        This function generates an embedding for the given query using the specified 
        embedding function and model choice. It then compares this embedding against 
        the embeddings of the chunks to find those that exceed a defined similarity 
        threshold. The results are sorted by similarity, and a limited number of 
        similar chunks are returned based on the specified filter limit and maximum 
        similarity delta.

        Args:
            query (str): The input query string for which similar chunks are to be found.
            chunks_with_embeddings (list[dict]): A list of dictionaries, each containing 
                                                an 'embedding' key representing the 
                                                chunk's embedding.
            embedding_function (Callable): A callable function that generates embeddings 
                                        for the given text. Defaults to 
                                        create_openai_embedding.
            model_choice (str): The model to be used for generating embeddings. Defaults to 
                                "text-embedding-3-large".
            similarity_threshold (float): The minimum similarity score for a chunk to be 
                                        considered similar. Defaults to 0.30.
            filter_limit (int): The maximum number of similar chunks to return. Defaults to 15.
            max_similarity_delta (float): The maximum allowed difference in similarity 
                                        between the most similar chunk and the others 
                                        to be included in the results. Defaults to 0.075.

        Returns:
            list[dict]: A list of dictionaries containing the similar chunks, each with 
                        an added 'similarity' key indicating the similarity score.

        Example:
            >>> query = "What is the capital of France?"
            >>> chunks = [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}]
            >>> similar_chunks = find_similar_chunks(query, chunks)
            >>> print(similar_chunks)
            [
                {"embedding": [0.1, 0.2, 0.3], "similarity": 0.9},
                {"embedding": [0.4, 0.5, 0.6], "similarity": 0.9}
            ]
    """
    query_embedding = embedding_function(text=query, model_choice=model_choice)

    similar_chunks = []
    for chunk in chunks_with_embeddings:
        similarity = cosine_similarity(query_embedding, chunk['embedding'])
        if similarity > similarity_threshold:
            chunk['similarity'] = similarity
            similar_chunks.append(chunk)
    
    logging.info(f"Found {len(similar_chunks)} similar chunks")
    
    if len(similar_chunks) == 0:
        return []
    
    similar_chunks.sort(key=lambda x: x['similarity'], reverse=True)

    limited_rows = similar_chunks[:filter_limit]
    logging.info(f"Limited to {len(limited_rows)}")

    max_similarity = max(row['similarity'] for row in similar_chunks)
    filtered_rows = [row for row in limited_rows if max_similarity - row['similarity'] <= max_similarity_delta]
    logging.info(f"Filtered to {len(filtered_rows)}")
    
    no_embedding_key_chunks = [{k: v for k, v in chunk.items() if k != 'embedding'} for chunk in filtered_rows]
    
    return no_embedding_key_chunks

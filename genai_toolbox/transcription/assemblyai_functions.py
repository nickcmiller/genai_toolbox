from genai_toolbox.text_prompting.model_calls import groq_text_response, openai_text_response, anthropic_text_response, fallback_text_response
from genai_toolbox.helper_functions.string_helpers import evaluate_and_clean_valid_response, write_to_file

import assemblyai as aai
import openai
from dotenv import load_dotenv
from typing import List, Dict, Optional

import os
import logging
import traceback
import re
import copy
from pathlib import Path

aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

def transcribe_audio(
    audio_file_path: str
) -> aai.Transcript:
    """
        Transcribes an audio file using AssemblyAI's transcription service.

        This function takes an audio file path as input and transcribes it using AssemblyAI's
        transcription service. It returns the transcription response from AssemblyAI.

        Args:
            audio_file_path (str): The path to the audio file to be transcribed.

        Returns:
            aai.Transcript: The transcription response from AssemblyAI.
    """
    if not os.path.exists(audio_file_path):
        logging.error(f"Audio file does not exist: {audio_file_path}")
        raise FileNotFoundError(f"Audio file does not exist: {audio_file_path}")

    config = aai.TranscriptionConfig(speaker_labels=True)

    try:
        transcriber = aai.Transcriber()
        response = transcriber.transcribe(audio_file_path, config=config)
        
        # Wait for transcription to complete
        while response.status not in ['completed', 'error']:
            time.sleep(1)  # Add a small delay between checks
        
        if response.status == 'error':
            logging.error(f"Transcription failed: {response.error}")
            raise RuntimeError(f"Transcription error: {response.error}")
        
        logging.info(f"Transcription successful for file: {audio_file_path}")
        return response
    
    except Exception as e:
        logging.error(f"Unexpected error during transcription: {e}")
        traceback.print_exc()
        raise RuntimeError(f"Transcription failed: {str(e)}")
    
def create_utterance_json(
    transcriber_transcript: aai.transcriber.Transcript,
    fields_to_exclude: List[str] = None
) -> List[Dict]:
    """
        Creates a JSON representation of utterances from an AssemblyAI transcript.

        This function takes an AssemblyAI transcript object and converts its utterances into a list of 
        dictionaries, each representing an utterance with selected fields. It allows for the exclusion 
        of specific fields from the output.

        Args:
            transcriber_transcript (aai.transcriber.Transcript): The AssemblyAI transcript object 
                containing utterances.
            fields_to_exclude (list[str], optional): A list of field names to exclude from the output. 
                Defaults to None.

        Returns:
            list: A list of dictionaries, where each dictionary represents an utterance with the 
            following keys (unless excluded):
                - confidence: The confidence score of the utterance.
                - end: The end time of the utterance.
                - speaker: The speaker label.
                - start: The start time of the utterance.
                - text: The transcribed text of the utterance.
                - words: A list of word objects in the utterance.

        Raises:
            ValueError: If the transcriber_transcript is invalid or missing the 'utterances' attribute.

        Example:
            >>> transcript = aai.transcriber.Transcript(...)
            >>> utterances_json = create_utterance_json(transcript, fields_to_exclude=['words'])
            >>> print(utterances_json[0])
            {
                'confidence': 0.95,
                'end': 3500,
                'speaker': 'A',
                'start': 0,
                'text': 'Hello, how are you?',
            }
    """
    fields_to_exclude = set(fields_to_exclude or [])
    
    def filter_utterance(utterance):
        base_dict = {
            "confidence": utterance.confidence,
            "end": utterance.end,
            "speaker": utterance.speaker,
            "start": utterance.start,
            "text": utterance.text,
            "words": [word.__dict__ for word in utterance.words]
        }
        return {k: v for k, v in base_dict.items() if k not in fields_to_exclude}

    return [filter_utterance(utterance) for utterance in transcriber_transcript.utterances]

def generate_assemblyai_utterances(
    audio_file_path: str,
    output_dir_name: str = None,
    fields_to_exclude: list[str] = ['words']
) -> list[dict]:
    """
        Generates utterances from an audio file using AssemblyAI transcription service.

        This function transcribes the given audio file, creates utterance JSON from the transcription,
        and optionally saves the utterances to a JSON file.

        Args:
            audio_file_path (str): The path to the audio file to be transcribed.
            output_dir_name (str, optional): The name of the directory to save the utterances JSON file.
                If None, the utterances are not saved to a file.

        Returns:
            list[dict]: A list of dictionaries, where each dictionary represents an utterance
            with keys such as 'confidence', 'end', 'speaker', 'start', 'text', and 'words'.

        Raises:
            Exception: If an error occurs during transcription or utterance generation.

        Note:
            This function relies on the `transcribe_audio` and `create_utterance_json` functions,
            which should be defined elsewhere in the module.
    """

    transcribed_audio_dict = transcribe_audio(audio_file_path)
    transcribed_utterances = create_utterance_json(transcribed_audio_dict, fields_to_exclude=fields_to_exclude)

    output_dict = {
        "extracted_title": Path(audio_file_path).stem,
        "transcribed_utterances": transcribed_utterances, 
        "transcript": create_text_transcript(transcribed_utterances)
    }

    if output_dir_name:
        file_name = f"{output_dict['extracted_title']}_utterances.json"
        file_path = write_to_file(
            content=output_dict,
            file_name=file_name,
            dir_name=output_dir_name
        )

    return output_dict

def create_text_transcript(
    utterances: List[Dict]
) -> str:
    """
        Extracts and formats a text transcript from the AssemblyAI transcription response.

        Args:
            utterances (List[Dict]): The list of utterances from the AssemblyAI transcription response.

        Returns:
            str: A formatted transcript string where each utterance is prefixed with the speaker label.

        Raises:
            ValueError: If the response is malformed or missing necessary data components.

        Example:
            >>> response = [
                {'speaker': 'A', 'text': 'Hello, how are you?'},
                {'speaker': 'B', 'text': 'I'm good, thanks!'}
            ]
            >>> print(get_transcript_assemblyai(response))
            Speaker A: Hello, how are you?
            Speaker B: I'm good, thanks!
    """
    try:
        for utterance in utterances:
            if 'speaker' not in utterance or 'text' not in utterance:
                raise ValueError("Each utterance must contain 'speaker' and 'text' fields.")
        
        if len(utterance['speaker']) > 1:
            transcript_parts = [
                f"{utterance['speaker']}: {utterance['text']}\n\n" for utterance in utterances
            ]
        else:
            transcript_parts = [
                f"Speaker {utterance['speaker']}: {utterance['text']}\n\n" for utterance in utterances
            ]
        
        return ''.join(transcript_parts)
    except Exception as e:
        logging.error(f"Failed to generate transcript: {e}")
        raise ValueError(f"Failed to generate transcript due to an error: {e}")

def generate_assemblyai_transcript(
    audio_file_path: str, 
    output_dir_name: str = None
) -> str:
    response = generate_assemblyai_utterances(audio_file_path)
    assemblyai_transcript = create_text_transcript(response['transcribed_utterances'])

    if output_dir_name:
        file_name = f"{response['extracted_title']}_transcript.txt"
        file_path = write_to_file(
            content=assemblyai_transcript,
            file_name=file_name,
            dir_name=output_dir_name
        )

    return assemblyai_transcript

def identify_speakers(
    summary: str,
    transcript: str
) -> dict:
    """
        Identifies the speakers in a podcast based on the summary and transcript.

        This function takes a summary and transcript of a podcast as input. It then uses the OpenAI API to generate a response that identifies
        the speakers in the podcast. The response is expected to be a dictionary mapping speaker labels
        (e.g., "Speaker A") to their actual names.

        The function will attempt to generate a valid response up to `max_tries` times. If a valid
        dictionary response is not obtained after the maximum number of attempts, an error is logged.

        Args:
            summary (str): A summary of the podcast.
            transcript (str): The full transcript of the podcast.

        Returns:
            dict: A dictionary mapping speaker labels to their actual names.

        Example:
            >>> summary = "In this podcast, John and Jane discuss their favorite movies."
            >>> transcript = "Speaker A: My favorite movie is The Shawshank Redemption. Speaker B: Mine is Forrest Gump."
            >>> result = identify_speakers(summary, transcript)
            >>> print(result)
            {'Speaker A': 'John', 'Speaker B': 'Jane'}
    """

    speaker_references_prompt = f"""
        What did each Speaker call each other in the transcript?

        What would you guess each Speaker's name is? Explain your reasoning.

        Transcript:\n
        {transcript}
    """

    speaker_references_system_prompt = "You are a trained specialist that accurately identifies the speakers in a transcript."

    model_order = [
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
        },
    ]

    speaker_reference_guess = fallback_text_response(
        prompt=speaker_references_prompt,
        system_instructions=speaker_references_system_prompt,
        model_order=model_order
    )

    prompt = f"""
        Using the context of the conversation in the transcript, and the summary, identify the participating speakers.
        
        Expected Hosts and Guests:\n
        {summary}

        Speaker Guesses:\n
        {speaker_reference_guess}

        Transcript of the conversation:\n {transcript}
    """

    system_prompt = """
    You only return properly formatted key-value store. 
        The output should Python eval to a dictionary. type(eval(response)) == dict

        Output Examples:
        ```
        Example 1:
        {
            "Speaker A": "FirstName LastName", 
            "Speaker B": "FirstName LastName"
        }

        Example 2:
        {
            "Speaker A": "FirstName LastName", 
            "Speaker B": "FirstName LastName"
        }
        ```
        All keys must be in the format 'Speaker X' where X is any letter or number
    """ 

    max_tries = 5
    
    for attempt in range(max_tries):
        response = openai_text_response(
            prompt=prompt, 
            model_choice="4o", 
            system_instructions=system_prompt
        )
        logging.info(f"Attempt {attempt + 1}: Response received.")
        logging.info(f"Response: {response}")

        try:
            response_dict = evaluate_and_clean_valid_response(response, dict)

            # Check that all keys and values in the dictionary are strings
            if not all(isinstance(key, str) and isinstance(value, str) for key, value in response_dict.items()):
                raise ValueError("All keys and values in the dictionary must be strings")
            
            # Check that all keys follow the expected format "Speaker X" where X is any letter or number
            speaker_pattern = re.compile(r"^Speaker [A-Za-z0-9]$")
            if not all(speaker_pattern.match(key) for key in response_dict.keys()):
                raise ValueError("All keys must be in the format 'Speaker X' where X is any letter or number")
            
            # Additional checks can be added here if needed, e.g., checking for empty values
            if any(not value.strip() for value in response_dict.values()):
                raise ValueError("Empty speaker names are not allowed")
            
            return response_dict
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")

    logging.error("Failed to obtain a valid dictionary response after maximum attempts.")
    raise ValueError("Failed to obtain a valid dictionary response after maximum attempts.")

def replace_speakers_in_utterances(
    utterances: list[dict],
    speaker_dict: dict
) -> list[dict]:
    """
        Replaces speaker placeholders in a list of utterances with actual speaker names.

        Args:
            utterances (list[dict]): A list of dictionaries representing utterances, 
                                    where each dictionary contains a 'speaker' key.
            speaker_dict (dict): A dictionary mapping speaker placeholders to actual speaker names.

        Returns:
            list[dict]: The updated list of utterances with speaker names replaced.

        Example:
            Input:
            utterances = [
                {"speaker": "Speaker A", "text": "Hello", ...},
                {"speaker": "Speaker B", "text": "Hi there", ...}
            ]
            speaker_dict = {"Speaker A": "John", "Speaker B": "Jane"}

            Output:
            [
                {"speaker": "John", "text": "Hello", ...},
                {"speaker": "Jane", "text": "Hi there", ...}
        ]
    """
    new_utterances = []
    for utterance in utterances:
        new_utterance = copy.deepcopy(utterance)
        full_speaker = "Speaker " + new_utterance['speaker']
        if full_speaker in speaker_dict:
            new_utterance["speaker"] = speaker_dict[full_speaker]
        new_utterances.append(new_utterance)

    return new_utterances

def replace_speakers_in_assemblyai_utterances(
    utterances: dict,
    summary: str,
    output_dir_name: str = None
) -> dict:
    speaker_dict = identify_speakers(summary, utterances['transcript'])
    replaced_utterances = replace_speakers_in_utterances(
        utterances['transcribed_utterances'], 
        speaker_dict
    )
    replaced_transcript = create_text_transcript(replaced_utterances)

    output_dict = {
        "extracted_title": utterances['extracted_title'],
        "transcribed_utterances": replaced_utterances, 
        "transcript": replaced_transcript
    }

    if output_dir_name:
        file_path = write_to_file(
            content=output_dict,
            file_name=f"{utterances['extracted_title']}_replaced.json",
            dir_name=output_dir_name
        )

    return output_dict

def generate_assemblyai_transcript_with_speakers(
    audio_file_path: str,
    audio_summary: str,
    output_dir_name: str = None
) -> str:
    transcribed_utterances = generate_assemblyai_utterances(audio_file_path)
    replaced_dict = replace_speakers_in_assemblyai_utterances(
        transcribed_utterances,
        audio_summary,
        output_dir_name=output_dir_name
    )

    return replaced_dict['transcript']

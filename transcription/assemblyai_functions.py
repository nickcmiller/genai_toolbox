import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from text_prompting.model_calls import groq_text_response, openai_text_response
from helper_functions.string_helpers import evaluate_and_clean_valid_response, write_to_file

import assemblyai as aai
import openai
from dotenv import load_dotenv

import logging
import traceback
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO)
load_dotenv(os.path.join(root_dir, '.env'))

aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

def transcribe_audio(
    audio_file_path: str
) -> aai.transcriber.Transcript:
    """
        Transcribes an audio file using the AssemblyAI library.

        Args:
            audio_file_path (str): The path to the audio file to transcribe.

        Returns:
            aai.transcriber.Transcript: The transcription response from AssemblyAI.

        Raises:
            FileNotFoundError: If the audio file cannot be found.
            IOError: If there is an issue with reading the audio file.
            RuntimeError: If transcription fails due to API errors.

        Example of response format:
            {
                "utterances": [
                    {
                        "confidence": 0.7246,
                        "end": 3738,
                        "speaker": "A",
                        "start": 570,
                        "text": "Um hey, Erica.",
                        "words": [...]
                    },
                    {
                        "confidence": 0.6015,
                        "end": 4430,
                        "speaker": "B",
                        "start": 3834,
                        "text": "One in.",
                        "words": [...]
                    }
                ]
            }
    """
    if not os.path.exists(audio_file_path):
        logging.error(f"Audio file does not exist: {audio_file_path}")
        raise FileNotFoundError(f"Audio file does not exist: {audio_file_path}")

    config = aai.TranscriptionConfig(speaker_labels=True)

    try:
        transcriber = aai.Transcriber()
        response = transcriber.transcribe(audio_file_path, config=config)
        logging.info(f"Transcription successful for file: {audio_file_path}")
        return response
    except aai.exceptions.APIError as api_error:
        logging.error(f"API error during transcription: {api_error}")
        raise RuntimeError(f"API error: {api_error}")
    except Exception as e:
        logging.error(f"Unexpected error during transcription: {e}")
        traceback.print_exc()
        raise RuntimeError(f"Unexpected error during transcription: {e}")

def create_text_transcript(
    transcriber_transcript: aai.transcriber.Transcript
) -> str:
    """
        Extracts and formats a text transcript from the AssemblyAI transcription response.

        Args:
            response (aai.transcriber.Transcript): The transcription response object from AssemblyAI.

        Returns:
            str: A formatted transcript string where each utterance is prefixed with the speaker label.

        Raises:
            ValueError: If the response is malformed or missing necessary data components.

        Example:
            >>> response = aai.transcriber.Transcript(...)
            >>> print(get_transcript_assemblyai(response))
            Speaker A: Hello, how are you?
            Speaker B: I'm good, thanks!
    """
    try:
        if not hasattr(transcriber_transcript, 'utterances') or not transcriber_transcript.utterances:
            logging.error("Invalid response: Missing 'utterances' attribute.")
            raise ValueError("Invalid response: Missing 'utterances' attribute.")

        transcript_parts = [
            f"Speaker {utterance.speaker}: {utterance.text}\n\n" for utterance in transcriber_transcript.utterances
        ]
        return ''.join(transcript_parts)
    except Exception as e:
        logging.error(f"Failed to generate transcript: {e}")
        raise ValueError(f"Failed to generate transcript due to an error: {e}")

def create_utterance_json(
    transcriber_transcript: aai.transcriber.Transcript
) -> list:
    utterances = []
    for utterance in transcriber_transcript.utterances:
        utterances.append({
            "confidence": utterance.confidence,
            "end": utterance.end,
            "speaker": utterance.speaker,
            "start": utterance.start,
            "text": utterance.text,
            "words": [word.__dict__ for word in utterance.words]
        })
    return utterances

def identify_speakers(
    summary: str,
    transcript: str,
    prompt: str = None,
    system_prompt: str = None,
) -> dict:
    """
        Identifies the speakers in a podcast based on the summary and transcript.

        This function takes a summary and transcript of a podcast as input, along with optional prompt
        and system prompt strings. It then uses the OpenAI API to generate a response that identifies
        the speakers in the podcast. The response is expected to be a dictionary mapping speaker labels
        (e.g., "Speaker A") to their actual names.

        The function will attempt to generate a valid response up to `max_tries` times. If a valid
        dictionary response is not obtained after the maximum number of attempts, an error is logged.

        Args:
            summary (str): A summary of the podcast.
            transcript (str): The full transcript of the podcast.
            prompt (str, optional): The prompt string to use for generating the response.
                If not provided, a default prompt will be used.
            system_prompt (str, optional): The system prompt string to use for generating the response.
                If not provided, a default system prompt will be used.

        Returns:
            dict: A dictionary mapping speaker labels to their actual names.

        Example:
            >>> summary = "In this podcast, John and Jane discuss their favorite movies."
            >>> transcript = "Speaker A: My favorite movie is The Shawshank Redemption. Speaker B: Mine is Forrest Gump."
            >>> result = identify_speakers(summary, transcript)
            >>> print(result)
            {'Speaker A': 'John', 'Speaker B': 'Jane'}
    """
    if prompt is None:
        prompt = f"""
            Using the context of the conversation in the transcript and the background provided by the summary, identify the participating speakers.

            Summary of the conversation:\n {summary}

            Transcript of the conversation:\n {transcript}
        """

    if system_prompt is None:
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
        """ 

    max_tries = 5
    
    for attempt in range(max_tries):
        response = openai_text_response(prompt, system_instructions=system_prompt)
        logging.info(f"Attempt {attempt + 1}: Response received.")

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

def replace_speakers_in_transcript(
    transcript: str, 
    speaker_dict: dict
) -> str:
    """
        Replaces speaker placeholders in a transcript with actual speaker names.

        Args:
            transcript (str): The transcript text with speaker placeholders.
            speaker_dict (dict): A dictionary mapping speaker placeholders to actual speaker names.

        Returns:
            str: The updated transcript with speaker names replaced.

        Example:
            Input:
                transcript = "Speaker A: Hello\nSpeaker B: Hi there"
                speaker_dict = {"Speaker A": "John", "Speaker B": "Jane"}

            Output:
                "John: Hello\nJane: Hi there"
    """
    for key, value in speaker_dict.items():
        transcript = transcript.replace(key, value)

    return transcript

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
    for utterance in utterances:
        full_speaker = "Speaker " + utterance['speaker']
        if full_speaker in speaker_dict:
            utterance["speaker"] = speaker_dict[full_speaker]

    return utterances

def generate_assemblyai_transcript(
    audio_file_path: str, 
    output_dir_name: str = None
) -> str:     
    """
        Generates a transcript from an audio file using AssemblyAI and optionally writes it to a file.

        Args:
            audio_file_path (str): The path to the audio file to be transcribed.
            output_file_path (str, optional): The path to the output file where the transcript will be saved. If None, the transcript is not written to a file.

        Returns:
            str: The transcript generated from the audio file.

        Raises:
            Exception: If transcription or file writing fails.

    """
    try:
        transcribed_audio_dict = transcribe_audio(audio_file_path)
        assemblyai_transcript = create_text_transcript(transcribed_audio_dict)
    except Exception as e:
        logging.error(f"Failed to transcribe audio: {e}")
        raise Exception(f"Transcription failed for file {audio_file_path}: {e}")

    if output_dir_name:
        raw_title = Path(audio_file_path).stem
        file_name = f"{raw_title}_transcript.txt"
        
        file_path = write_to_file(
            content=assemblyai_transcript,
            file=file_name,
            output_dir_name=output_dir_name
        )

        logging.info(f"Transcript successfully written to {file_path}")

    return assemblyai_transcript

def replace_speakers_in_assemblyai_transcript(
    assemblyai_transcript: str,
    audio_summary: str,
    first_host_speaker: str = None,
    output_file_name: str = None,
    output_dir_name: str = None
) -> str:
    """
        Replaces speaker placeholders in the transcript with actual names using a summary and optionally writes the result to a file.

        Args:
            assemblyai_transcript (str): The transcript text with speaker placeholders.
            audio_summary (str): A summary of the audio content used to identify speakers.
            first_host_speaker (str, optional): The name of the first host to speak, if known.
            output_file_path (str, optional): The path to the output file where the modified transcript will be saved. If None, the transcript is not written to a file.

        Returns:
            str: The transcript with speaker placeholders replaced by actual names.

        Raises:
            Exception: If an error occurs during the processing.
    """
    try:
        if first_host_speaker:
            audio_summary += f"\n\nThe first host to speak is {first_host_speaker}"
        
        speaker_dict = identify_speakers(audio_summary, assemblyai_transcript)
        transcript_with_replaced_speakers = replace_speakers_in_transcript(assemblyai_transcript, speaker_dict)

        if output_file_name:
            file_path = write_to_file(
                content=transcript_with_replaced_speakers,
                file=f"{output_file_name}_replaced.txt",
                output_dir_name=output_dir_name
            )
            logging.info(f"Transcript successfully written to {file_path}")

        return transcript_with_replaced_speakers
    except Exception as e:
        logging.error(f"Error in replace_assemblyai_speakers: {e}")
        raise Exception(f"Failed to process transcript: {e}")

def generate_assemblyai_utterances(
    audio_file_path: str,
    output_dir_name: str = None
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
    transcribed_utterances = create_utterance_json(transcribed_audio_dict)
    transcript = create_text_transcript(transcribed_audio_dict)

    if output_dir_name:
        raw_title = Path(audio_file_path).stem
        file_name = f"{raw_title}_utterances.json"
        
        file_path = write_to_file(
            content=transcribed_utterances,
            file=file_name,
            output_dir_name=output_dir_name
        )
        logging.info(f"Transcribed utterances successfully written to {file_path}")

    return [
        {
            "transcribed_utterances": transcribed_utterances, 
            "transcript": transcript
        }
    ]

def replace_speakers_in_assemblyai_utterances(
    transcribed_utterances: list[dict],
    audio_summary: str,
    first_host_speaker: str = None,
    output_file_name: str = None,
    output_dir_name: str = None
) -> list[dict]:
    speaker_dict = identify_speakers(audio_summary, transcribed_utterances['transcript'])
    replaced_utterances = replace_speakers_in_utterances(
        transcribed_utterances['transcribed_utterances'], 
        speaker_dict
    )

    

    return [
        {
            "transcribed_utterances": replaced_utterances, 
            "transcript": transcribed_utterances['transcript']
        }
    ]

    

if __name__ == "__main__":
    from download_sources.podcast_functions import return_all_entries_from_feed, download_podcast_audio, generate_episode_summary
    import json

    mfm_feed_url = "https://feeds.megaphone.fm/HS2300184645"
    dithering_feed_url = "https://dithering.passport.online/feed/podcast/KCHirQXM6YBNd6xFa1KkNJ"
    entries = return_all_entries_from_feed(dithering_feed_url)
    first_entry = entries[0]
    audio_file_path = download_podcast_audio(first_entry["url"], first_entry["title"])

    if False:
        
        transcribed_audio_dict = transcribe_audio_assemblyai(audio_file_path)
        full_transcript = create_text_transcript_assemblyai(transcribed_audio_dict)
        transcribed_utterances = create_utterance_json(transcribed_audio_dict)
        with open('transcribed_utterances.json', 'w') as f:
            f.write(json.dumps(transcribed_utterances, indent=4))
        with open('full_transcript.txt', 'w') as f:
            f.write(full_transcript)
    else:
        with open('transcribed_utterances.json', 'r') as f:
            transcribed_utterances = json.load(f)
        with open('full_transcript.txt', 'r') as f:
            full_transcript = f.read()

    filtered_utterances = [
        {k: v for k, v in utterance.items() if k != 'words' and k != 'confidence'}
        for utterance in transcribed_utterances
    ]
    if False:
        episode_summary = generate_episode_summary(first_entry["summary"], first_entry["feed_summary"])
        speakers = identify_speakers(episode_summary, full_transcript)
        with open('speakers.json', 'w') as f:
            f.write(json.dumps(speakers, indent=4))
    else:
        with open('speakers.json', 'r') as f:
            speakers = json.load(f)


    # for utterance in filtered_utterances:
    #     full_speaker = "Speaker " + utterance['speaker']
    #     if full_speaker in speakers:
    #         utterance["speaker"] = speakers[full_speaker]

    if False:
        new_utterances = replace_speakers_in_utterances(filtered_utterances, speakers)
        write_to_file(
            content=new_utterances,
            file="new_utterances.json",
            output_dir_name="tmp"
        )   
    else:
        with open('tmp/new_utterances.json', 'r') as f:
            new_utterances = json.load(f)
    
    print(json.dumps(new_utterances[:5], indent=4))
    

        
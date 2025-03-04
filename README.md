# GenAI Toolbox

The **GenAI Toolbox** is a comprehensive collection of tools and utility functions designed to streamline advanced generative AI workflows. It aids in processing text and audio data, generating embeddings, managing API clients for various LLM providers, and chaining prompts for multi-step text processing. This toolbox is modular, enabling you to use only the components you need, from audio/data download and transcription to diarization (speaker segmentation) and text prompting.

---

## Table of Contents

- [Installation & Setup](#installation--setup)
  - [Prerequisites](#prerequisites)
  - [Environment Setup](#environment-setup)
  - [API Keys](#api-keys)
- [Directory Structure](#directory-structure)
- [Key Modules & Functions](#key-modules--functions)
  - [Chunk and Embed](#chunk-and-embed)
  - [API Clients](#api-clients)
  - [Diarization & Transcription](#diarization--transcription)
  - [Download Sources](#download-sources)
  - [Helper Functions](#helper-functions)
  - [Text Prompting](#text-prompting)
- [Usage Examples](#usage-examples)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Installation & Setup

### Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- FFmpeg (for audio processing)

### Environment Setup

#### 1. Install the package
Install directly from the repository for the latest version:
```bash
pip3 install git+https://github.com/nickcmiller/genai-toolbox.git
```
Or for development/editable installation, clone the repository and install with pip:
```bash
git clone https://github.com/nickcmiller/genai-toolbox.git
cd genai-toolbox
pip3 install -e .
```
#### 2. Install FFmpeg
Install FFmpeg for audio processing based on your operating system:

**Windows:**
```bash
# Using Chocolatey or download from https://ffmpeg.org/download.html
choco install ffmpeg
```

**macOS:**
```bash
# Using Homebrew
brew install ffmpeg
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install ffmpeg

# CentOS/RHEL
sudo yum install ffmpeg
```

### API Keys

The toolbox requires several API keys for full functionality. Set up your environment variables either in your system or in a `.env` file:

```env
# Required for OpenAI embeddings and GPT models
OPENAI_API_KEY=your_openai_key

# Required for Anthropic Claude models
ANTHROPIC_API_KEY=your_anthropic_key

# Required for Groq models
GROQ_API_KEY=your_groq_key

# Required for Perplexity models
PERPLEXITY_API_KEY=your_perplexity_key

# Required for YouTube API functions
YOUTUBE_API_KEY=your_youtube_key

# Required for Hugging Face's pyannote-audio (diarization)
HUGGINGFACE_ACCESS_TOKEN=your_huggingface_key
```

If using a `.env` file, you can load it in your Python code:
```python
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
```

Note: You'll need to accept the terms of service on Hugging Face's website for the pyannote/speaker-diarization model before using the diarization features.

---
## Directory Structure 

```
genai_toolbox/
├── chunk_and_embed/
│ ├── init.py
│ ├── chunking_functions.py 
│ ├── embedding_functions.py 
│ ├── llms_with_queries.py 
├── clients/
│ ├── init.py
│ ├── anthropic_client.py 
│ ├── groq_client.py 
│ └── openai_client.py 
├── diarization/
│ ├── init.py
│ ├── custom_whisper_diarization.py 
│ └── transcribe_and_diarize.py 
├── download_sources/
│ ├── init.py
│ ├── podcast_functions.py 
│ ├── youtube_functions.py 
│ └── requirements.txt 
├── helper_functions/
│ ├── init.py
│ ├── datetime_helpers.py 
│ └── string_helpers.py 
├── text_prompting/
│ ├── init.py
│ ├── model_calls.py 
│ ├── model_requirements.txt 
│ └── prompt_chaining.py 
└── transcription/
└── init.py 
```


---

## Key Modules & Functions

### Chunk and Embed

- **chunking_functions.py**  
  - *split_text_string()*: Splits a text by a designated separator into non-empty chunks.
  - *consolidate_split_chunks()*: Merges consecutive chunks that are below a set minimum length.
  - *consolidate_similar_split_chunks()*: Consolidates text chunks based on precomputed similarity scores.
  - *convert_utterance_speaker_to_speakers()* & *consolidate_short_utterances()*: Specifically designed to process AssemblyAI transcription utterances.
  - *format_speakers_in_utterances()*: Formats the text to include speaker references.
  - *milliseconds_to_minutes_in_utterances()* & *rename_start_end_to_ms()*: Converts timestamps from milliseconds to human-readable formats.

- **embedding_functions.py**  
  - *create_openai_embedding()*: Generates an embedding for a given text using the OpenAI API.
  - *num_tokens_from_string()*: Computes the token count for a string using specified encoding.
  - *create_embedding_for_dict()*: Augments a dictionary containing text with its corresponding embedding.
  - *create_embedding_for_dict_async()* & *embed_dict_list_async()*: Asynchronous embedding generation with rate limiting.
  - *add_similarity_to_next_dict_item()*: Computes and appends the cosine similarity between consecutive embeddings.
  - *find_similar_chunks()*: Finds and filters relevant chunks from a list by comparing their embeddings to a query.

- **llms_with_queries.py**  
  - *llm_response_with_query()*: Combines similar source chunks and a user question to generate a detailed LLM-based answer.
  - *stream_response_with_query()*: Similar to the above, but supports streaming API responses.

---

### API Clients

The **clients** module provides simple wrappers to instantiate API clients:

- *anthropic_client.py*: Returns an Anthropic client instance.
- *groq_client.py*: Returns a Groq client instance.
- *openai_client.py*: Returns an OpenAI client instance.

Each client abstracts error handling and logging, making it easier to integrate with respective APIs.

---

### Diarization & Transcription

- **custom_whisper_diarization.py**  
  - *diarize_audio()*: Applies a pre-trained speaker diarization pipeline (using pyannote) to an audio file.
  - *diarize_audio_chunks()*: Processes multiple audio chunks, adjusting start/end times for continuous segments.
  - *condense_diarization_results()*: Combines consecutive diarized segments from the same speaker.
  - *diarize_and_condense_audio_chunks()*: High-level function that diarizes and then condenses audio diarization results.

- **transcribe_and_diarize.py**  
  - *create_audio_chunks()*: Splits a long audio file into smaller, manageable chunks.
  - *merge_diarization_and_transcription()*: Merges speaker diarization results and transcribed texts by analyzing temporal overlap.
  - *create_transcript()*: Formats and generates a final transcript including speaker identifiers.
  - *main()*: Orchestrates the full process from downloading a YouTube video, chunking, transcribing, diarizing, and merging results into a final transcript.

---

### Download Sources

- **podcast_functions.py**  
  - *parse_feed()* & *extract_metadata_from_feed()*: Parse a podcast RSS feed and extract episode metadata.
  - *return_all_entries_from_feed()* & *return_entries_by_date()*: Retrieve and filter podcast episodes by publication date.
  - *download_podcast_audio()*: Downloads podcast audio given its URL and saves it locally.
  - *generate_episode_summary()*: Generates a summary description of an episode based on its feed metadata.

- **youtube_functions.py**  
  - *yt_dlp_download()*: Downloads and converts YouTube videos to audio (MP3), with URL and ID validation.
  - *is_valid_youtube_url_or_id()*: Validates YouTube URLs or video IDs.
  - Functions to retrieve channel and video metadata using the YouTube Data API.

---

### Helper Functions

- **datetime_helpers.py**  
  - *convert_date_format()*: Converts dates from various formats into a human-readable format.
  - *get_date_with_timezone()*: Parses date strings and returns timezone-localized datetime objects.

- **string_helpers.py**  
  - *retrieve_file()*, *write_to_file()*, *delete_file()*: File input/output and retrieval operations.
  - *sort_folder_of_txt_files()*: Sorts text files in a folder based on their numeric prefixes.
  - *create_string_list_from_file_list()*, *concatenate_list_text_to_list_text()*: Build and combine lists of strings.
  - Functions to validate and clean API response strings.

---

### Text Prompting

- **model_calls.py**  
  - *get_client()*: Returns the appropriate client for a given API provider.
  - *openai_text_response()*, *groq_text_response()*, *anthropic_text_response()*, *perplexity_text_response()*: Generate text responses using the corresponding provider.
  - *fallback_text_response()*: Implements a fallback strategy that iterates through multiple providers if one fails.
  - Internal helper functions to log response information, handle errors, and manage streaming responses.

- **prompt_chaining.py**  
  - *prompt_string_list()*: Applies instructions to each string in a list via LLM query.
  - *execute_prompt_dict()*: Concatenates modified strings and original text, then applies a prompt for further revision.
  - *revise_list_with_prompt_list()* / *revise_string_with_prompt_list()*: Allows multi-step iterative revisions of strings with detailed logging and revision history.

---

## Usage Examples

Below is a basic example of how to use some parts of the toolbox:

```python
# Example: Download a YouTube video and generate an embedding for a text chunk.
from genai_toolbox.download_sources.youtube_functions import yt_dlp_download
from genai_toolbox.chunk_and_embed.embedding_functions import create_embedding_for_dict

# Download video audio.
video_audio_path = yt_dlp_download("https://youtu.be/VIDEO_ID")
print("Downloaded audio:", video_audio_path)

# Create an embedding for a text chunk.
text_chunk = {"text": "This is a sample text for embedding generation."}
embedding_dict = create_embedding_for_dict(embedding_function=lambda model_choice, text: [0.1, 0.2, 0.3], chunk_dict=text_chunk)
print("Generated embedding:", embedding_dict)
```

You can similarly use functions from the diarization, download, helper, and text prompting modules as documented in their respective docstrings.

---

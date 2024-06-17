from openai import OpenAI

import os
import logging
from dotenv import load_dotenv

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(root_dir, '.env'))
logging.basicConfig(level=logging.INFO)

def openai_client():
    try:
        return OpenAI()
    except Exception as e:
        logging.error(f"Error getting OpenAI client: {e}")
        raise e
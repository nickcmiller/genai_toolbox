import os
from dotenv import load_dotenv
import logging
from groq import Groq

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(root_dir, '.env'))
logging.basicConfig(level=logging.INFO)

def groq_client():
    try:
        return Groq()
    except Exception as e:
        logging.error(f"Error getting Groq client: {e}")
        raise e
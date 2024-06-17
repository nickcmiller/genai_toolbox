import os
from dotenv import load_dotenv
import logging
from anthropic import Anthropic

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(root_dir, '.env'))
logging.basicConfig(level=logging.INFO)

def anthropic_client():
    try:
        return Anthropic()
    except Exception as e:
        logging.error(f"Error getting Anthropic client: {e}")
        raise e
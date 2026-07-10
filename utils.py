import os
from dotenv import load_dotenv

def load_environment_variables(env_file='.env'):
    """
    Loads environment variables and API keys from a given file.
    
    Note: You mentioned a '.venv' file, but usually '.venv' is a directory for 
    your virtual environment. Environment variables are typically stored in a 
    '.env' file. If your file is indeed named '.venv', you can pass env_file='.venv'.
    """
    # Load environment variables from the specified file
    loaded = load_dotenv(dotenv_path=env_file)
    
    if not loaded:
        print(f"Warning: Could not find or load environment variables from '{env_file}'.")
    
    # Return specific keys or rely on them being set in os.environ
    return {
        # Example of how to retrieve a specific API key:
        # 'my_api_key': os.getenv('MY_API_KEY')
    }

def get_api_key(key_name: str) -> str:
    """
    Utility function to safely grab an API key.
    Ensure load_environment_variables() was called first.
    """
    key = os.getenv(key_name)
    if not key:
        raise ValueError(f"API key '{key_name}' not found. Ensure it is set in your env file.")
    return key

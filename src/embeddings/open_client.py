from openai import AzureOpenAI
from dotenv import load_dotenv
import os

class OpenClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OpenClient, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):

        if self._initialized:
            return
        
        load_dotenv()

        AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
        AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
        
        if self._initialized:
            return
        self.client = AzureOpenAI(
                    api_key=AZURE_API_KEY,
                    azure_endpoint=AZURE_ENDPOINT,
                    api_version=AZURE_API_VERSION,
                    )
        self._initialized = True

    def embed(self, text:list[str]):
        resp = self.client.embeddings.create(input=text, model="text-embedding-3-small")
        return [d.embedding for d in resp.data]
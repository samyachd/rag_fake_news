import chromadb

class Database:
    _instance = None

    def __new__(cls, path: str = "../data/chroma_db", collection_name: str = "news_collection"):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, path: str = "../data/chroma_db", collection_name: str = "news_collection"):
        if self._initialized:
            return
        
        self.client = chromadb.PersistentClient(path=path)
        self.collection_name = collection_name
        self._initialized = True

    def get_collection(self):
        """Retourne (ou crée) la collection demandée"""
        return self.client.get_or_create_collection(name=self.collection_name)

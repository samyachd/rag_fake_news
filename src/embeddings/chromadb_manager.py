import chromadb

class  Singleton (object):
    instance = None       # Attribut statique de classe
    def __new__(cls): 
        "méthode de construction standard en Python"
        if cls.instance is None:
            cls.instance = object.__new__(cls)
        return cls.instance

# Utilisation
monSingleton1 =  Singleton()
monSingleton2 =  Singleton()
    

class Database(Singleton):
    def __init__(self, client):
        client = chromadb.PersistentClient(path="../data/embeddings")
        self.client = client

    def 
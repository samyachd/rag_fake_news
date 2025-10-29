from src.preprocessing.clean_text import PreProcessing
from src.embeddings.chromadb_manager import Database

def embeddings_test():
    # PreProcessing : mettre d'abord path_fake, path_true puis les colonnes
    data = PreProcessing("data/raw/Fake.csv", "data/raw/True.csv", ["title","text"])
    data_cleaned = data.clean()
    print("data cleaned")
    chunked_data = PreProcessing.chunk_text(data_cleaned)
    print("data chunked")
    
    # Database : mettre le path du fichier chroma, puis le nom de la collection
    database = Database("../data/chroma_db", "news_collection")
    print("db créée")
    database.embedding_upsert(chunked_data)
    print("data insérée")
    

embeddings_test()
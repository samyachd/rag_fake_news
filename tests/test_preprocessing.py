from src.preprocessing.clean_text import PreProcessing
from src.preprocessing.chunk_text import chunk_text

def preprocess():
    # PreProcessing : mettre d'abord path_fake, path_true puis les colonnes
    print("Preprocess started")
    data = PreProcessing("data/raw/Fake.csv", "data/raw/True.csv", ["title","text"])
    data_cleaned = data.clean()
    print("data cleaned")
    chunked_data = chunk_text(data_cleaned)
    print("data chunked")

preprocess()
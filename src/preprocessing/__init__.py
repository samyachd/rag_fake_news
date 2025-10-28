import spacy
from functools import lru_cache

@lru_cache()
def get_spacy_model():
    return spacy.load("en_core_web_sm")

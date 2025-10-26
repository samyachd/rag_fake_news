import re  # Pour les expressions regex
import string
from bs4 import BeautifulSoup  # Pour la suppression des balises HTML
import spacy # Pour la lemmatisation et les stopwords
from spacy.lang.en.stop_words import STOP_WORDS
# text est mis en parenthéses car nous allons travailler sur la colonne text du DB


# Fonction de suppression d'URL
def delete_urls(text):
    """Suppprime les URLs du texte"""
    return re.sub(r'http\S+|www\S+|https\S+', '', text )


# Fonction de suppression balises html
def delete_html_rags(text):
    """Supprime les balises HTML du texte"""
    soup = BeautifulSoup( text, "html.parser")
    return soup.get_text()

# Suppression des caratéres speciaux et espaces
def delete_special_charaters(text):
    """Supprime les caractéres spéciaux et les espaces"""
    # Garder uniquement les lettre et espaces
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # remplace les espaces multiple par une seul espace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Mise en minuscules
def to_lowercase(text):
    """Convertit le texte en minuscule"""
    return text.lower()

# Suppression des storpwords
nlp = spacy.load("en_core_web_sm")
def delete_stopwords(text):
    """
    Supprime les stopwords du texte
    Returns:
        str: Text with stopwords removed
    """
    if not isinstance(text, str):
        return ""
    try:
        doc = nlp(text)
        tokens = [token.text for token in doc if not token.is_stop and token.text.strip()]
        return " ".join(tokens)
    except Exception as e:
        print(f"Error in delete_stopwords: {str(e)}")
        return text


# Construction des la pipeline :
def clean_text_pipeline(text):
    if not isinstance(text, str):
        return ""
    text = delete_urls(text)
    text = delete_html_rags(text)
    text = delete_special_charaters(text)
    text = to_lowercase(text)
    text = delete_stopwords(text)
    return text

# Test global
sample = "<p>This is a FAKE news about https://fake.com politics!!</p>"
print(clean_text_pipeline(sample))
# ➜ "fake news politics"


# fonction pour le chunking le déoupage des texte en plusieurs token
def chunk_text(text, chunk_size=200, overlap=50):
    """
    Découpe un texte en plusieurs morceaux (chunks)
    avec un chevauchement optionnel.
    
    text : str → le texte à découper
    chunk_size : int → nombre de mots par chunk
    overlap : int → nombre de mots qui se chevauchent entre deux chunks
    """
    # Séparer le texte en mots
    words = text.split()
    chunks = []
    
    # Parcourir les mots par étapes de (chunk_size - overlap)
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]   # Découper un morceau
        chunks.append(" ".join(chunk))    # Reformer le texte du chunk
    
    return chunks
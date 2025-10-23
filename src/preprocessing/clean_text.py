import re
from bs4 import BeautifulSoup
import spacy
import pandas as pd


class PreProcessing():
    def __init__(self, path : str):
        self.path = path

    def load_csv(self) -> pd.DataFrame:
        return pd.read_csv(self.path)

    def delete_url_html_specials_lower(self, df:pd.DataFrame) -> pd.DataFrame:
        """Suppprime les URLs du texte"""
        with df["title"] as title, df["text"] as text:
            title = title.str.replace(r'http[s]?://\S+', '<URL>', regex=True)
            text = text.str.replace(r'http[s]?://\S+', '<URL>', regex=True)
            title = title.str.replace(r'<.*?>', '', regex=True)
            text = text.str.replace(r'<.*?>', '', regex=True)
            title = title.str.replace(r"[^\w\s]",'', regex=True)
            text = text.str.replace(r"[^\w\s]",'', regex=True)
            title = title.str.lower()
            text = title.str.lower()
        return df


    def to_lowercase(text):
        """Convertit le texte en minuscule"""
        return text.lower()

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

    def clean_text_pipeline(text):
        if not isinstance(text, str):
            return ""
        text = delete_urls(text)
        text = delete_html_rags(text)
        text = delete_special_charaters(text)
        text = to_lowercase(text)
        text = delete_stopwords(text)
        return text


    sample = "<p>This is a FAKE news about https://fake.com politics!!</p>"
    print(clean_text_pipeline(sample))


    def chunk_text(text, chunk_size=200, overlap=50):
        """
        Découpe un texte en plusieurs morceaux (chunks)
        avec un chevauchement optionnel.
        
        text : str → le texte à découper
        chunk_size : int → nombre de mots par chunk
        overlap : int → nombre de mots qui se chevauchent entre deux chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = words[i:i + chunk_size]
            chunks.append(" ".join(chunk))
        return chunks
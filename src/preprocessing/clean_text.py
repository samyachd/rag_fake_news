from src.preprocessing import get_spacy_model
import re
import pandas as pd

class PreProcessing():
    def __init__(self, path_fake:str, path_true:str, cols:list):
        self.path_fake = path_fake
        self.path_true = path_true
        self.cols = cols

    def load_csv(self) -> pd.DataFrame:
        
        df_fake = pd.read_csv(self.path_fake)
        df_true = pd.read_csv(self.path_true)
        for col in self.cols:
            if col not in df_fake.columns or col not in df_true.columns:
                raise ValueError(f"Colonne manquante dans le DataFrame : {col}")
        df_fake["label"] = 0
        df_true["label"] = 1
        df = pd.concat([df_fake, df_true])
        df = df.drop_duplicates()
        return df
    
    def delete_url_html_specials_lower_df(self, df:pd.DataFrame) -> pd.DataFrame:
        """Suppprime les URLs du texte"""
        for col in self.cols:
            df[col] = (
                df[col]
                .fillna("")
                .astype(str)
                .str.replace(r"http[s]?://\S+", "<URL>", regex=True)
                .str.replace(r"<.*?>", "", regex=True)
                .str.replace(r"[^\w\s]", "", regex=True)
                .str.lower()
            )
        return df
    
    def delete_stopwords_df(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Supprime les stopwords du texte
        Returns:
            str: Text with stopwords removed
        """
        nlp = get_spacy_model()
        for col in self.cols:
            cleaned_col = []
            for doc in nlp.pipe(df[col].fillna("").astype(str).tolist(), disable=["parser", "ner"]):
                tokens = [tok.text for tok in doc if tok.text.strip() and not tok.is_stop]
                cleaned_col.append(" ".join(tokens))
            df[col] = cleaned_col
        return df
    
    def clean_df(self):

        df = self.load_csv()
        df = self.delete_url_html_specials_lower_df(df)
        df = self.delete_stopwords_df(df)
        return df
    
    def save_to_df(self, df, path:str):
        df.to_csv(path)
        return
    
    @staticmethod
    def delete_url_html_specials_lower_text(text:str) -> str:
        """Suppprime les URLs du texte"""
        text = re.sub(r"http[s]?://\S+", "<URL>", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[^\w\s]", "", text)
        text = text.lower()
        return text
    
    @staticmethod
    def delete_stopwords_text(text:str) -> str:
        """
        Supprime les stopwords du texte
        Returns:
            str: Text with stopwords removed
        """
        nlp = get_spacy_model()
        tokens_all = []
        for doc in nlp.pipe(text, disable=["parser", "ner"]):
            tokens = [tok.text for tok in doc if tok.text.strip() and not tok.is_stop]
            tokens_all.append(tokens[0])
        return tokens_all
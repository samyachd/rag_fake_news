from src.preprocessing import get_spacy_model
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
        df_true["true"] = 1
        df = pd.concat([df_fake, df_true])
        df = df.drop_duplicates()
        return df
    
    def delete_url_html_specials_lower(self, df:pd.DataFrame) -> pd.DataFrame:
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
    
    def delete_stopwords(self, df:pd.DataFrame) -> pd.DataFrame:
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
    
    def clean(self, path:str):

        df = self.load_csv()
        df = self.delete_url_html_specials_lower(df)
        df = self.delete_stopwords(df)
        df.to_csv(path)
        return df
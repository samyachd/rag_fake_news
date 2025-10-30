import pandas as pd

def chunk(path_load:str, cols:list, chunk_size=160, overlap=30) -> pd.DataFrame:
        """
        Découpe les textes des colonnes spécifiées dans self.cols en morceaux (chunks)
        avec un chevauchement optionnel entre les segments.
        """ 
        df = pd.read_csv(path_load)
        for col in cols:
            chunked_col = []
            for text in df[col].fillna("").astype(str):
                words = text.split()
                chunks = []
                for i in range(0, len(words), chunk_size - overlap):
                    chunk = " ".join(words[i:i + chunk_size])
                    chunks.append(chunk)
                chunked_col.append(chunks)
            df["type"] = col
        df["chunks"] = chunked_col
        return df
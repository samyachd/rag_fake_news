import chromadb
import pandas as pd
import numpy as np
from chromadb.utils import embedding_functions


class Database():
    def __init__(self, path:str = "../data/chroma_db", collection_name: str = "news_collection"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection_name = collection_name  
            

    def embedding_upsert(self, path:str):
        df = pd.read_csv(path)
        ollama_embed = embedding_functions.OllamaEmbeddingFunction(model_name="all-minilm")
          
        collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=ollama_embed
            )
        all_ids, all_docs, all_metadatas = [], [], []

        for idx, row in df.iterrows():
            for j, chunk in enumerate(row["chunks"]):

                all_ids.append(f"doc{idx}_chunk{j}")
                all_docs.append(chunk)
                all_metadatas.append({
                    "article_id": int(idx),
                    "chunk_id": int(j),
                    "type" : row["type"] if "type" in df.columns else None,
                    "label": row["label"] if "label" in df.columns else None,
                    "date": row["date"] if "date" in df.columns else None,
                    "subject" : row["subject"] if "subject" in df.columns else None,
                })
        
        batch_size = 50
        for i in range(0, len(all_docs), batch_size):
            batch_ids = all_ids[i:i + batch_size]
            batch_docs = all_docs[i:i + batch_size]
            batch_metas = all_metadatas[i:i + batch_size]
                
            try:
                embeddings = ollama_embed(batch_docs)
                normalized_embeddings = [v / np.linalg.norm(v) for v in embeddings]

                collection.upsert(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metas,
                    embeddings=normalized_embeddings
                )

            except Exception as e:
                print(f"Erreur dans ce batch : {e}")
                continue

        print("Insertion terminée avec succès.")
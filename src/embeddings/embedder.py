import ollama
import pandas as pd
from chromadb.utils import embedding_functions
from functools import lru_cache

# @lru_cache()
# def data_embedder(df:pd.Dataframe) -> dict:
#     ollama_embed = embedding_functions.OllamaEmbeddingFunction(model_name="all-minilm")
        
#     collection = self.client.get_or_create_collection(
#         name=self.collection_name,
#         embedding_function=ollama_embed
#         )
#     all_ids, all_docs, all_metadatas = [], [], []

#     for idx, row in df.iterrows():
#         for j, chunk in enumerate(row["chunks"]):

#             all_ids.append(f"doc{idx}_chunk{j}")
#             all_docs.append(chunk)
#             all_metadatas.append({
#                 "article_id": int(idx),
#                 "chunk_id": int(j),
#                 "type" : row["type"] if "type" in df.columns else None,
#                 "label": row["label"] if "label" in df.columns else None,
#                 "date": row["date"] if "date" in df.columns else None,
#                 "subject" : row["subject"] if "subject" in df.columns else None,
#             })
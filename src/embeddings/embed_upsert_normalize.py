import pandas as pd
import numpy as np
import os
from openai import AzureOpenAI

def normalize_vectors(vectors):
    return [v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else v for v in vectors]

def embedding_upsert(collection, df:pd.DataFrame, client: AzureOpenAI):
        
    if df["chunks"].dtype == object:
        try:
            df["chunks"] = df["chunks"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        except Exception:
            print("Erreur de conversion des chunks — vérifier le format dans le CSV")
            return
        
    df=df.reset_index(drop=True)

    all_ids, all_docs, all_metadatas = [], [], []

    for idx, row in df.iterrows():
        chunks = row["chunks"]
        if not isinstance(chunks, list):
            continue
        for j, chunk in enumerate(row["chunks"]):
            all_ids.append(f"doc{idx}_chunk{j}")
            all_docs.append(chunk)
            all_metadatas.append({
                "article_id": idx,
                "chunk_id": j,
                "type":row.get("type"),
                "label": row.get("label"),
                "source": row.get("source"),
            })
    
    batch_size = 50
    for i in range(0, len(all_docs), batch_size):
        batch_ids = all_ids[i:i + batch_size]
        batch_docs = all_docs[i:i + batch_size]
        batch_metas = all_metadatas[i:i + batch_size]
            
        print(f"Batch {i//batch_size + 1} / {len(all_docs)//batch_size + 1}")

        try:
            embeddings = client.embed(batch_docs)
            normalized_embeddings = normalize_vectors(embeddings)

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
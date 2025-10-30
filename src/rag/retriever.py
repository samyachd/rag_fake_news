from src.embeddings import normalize_vectors

def retrieve_text(collection, embed, user_text:str) -> list:

    query_embedding = embed([user_text])
    query_embedding = normalize_vectors(query_embedding)

    n_chunks = 10

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_chunks
    )

    retrieved_chunks = []
    retrieved_labels = []

    for docs, metas in zip(results["documents"], results["metadatas"]):
        for doc, meta in zip(docs, metas):
            retrieved_chunks.append(doc)
            retrieved_labels.append(meta.get("label", 0))

    context = ""
    for chunk, label in zip(retrieved_chunks, retrieved_labels):
        context += f"[Label: {'TRUE' if label == 1 else 'FAKE'}] {chunk}\n\n"
    
    return context, retrieved_labels
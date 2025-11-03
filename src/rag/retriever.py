from src.embeddings import normalize_vectors
from openai import AzureOpenAI
from src.preprocessing.clean_text import PreProcessing

def retrieve_text(client: AzureOpenAI, collection, user_text:str) -> list:
    
    user_text = PreProcessing.delete_url_html_specials_lower_text(text = user_text)
    user_text = PreProcessing.delete_stopwords_text(text = user_text)
    print(user_text)
    query_embedding = client.embed(user_text)
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
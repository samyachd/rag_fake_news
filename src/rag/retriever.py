from chromadb.utils import embedding_functions
from src.embeddings import normalize_vectors
from src.embeddings import Database

def retrieve_text(collection, embed):

    print("Enter your prompt")
    user_text = input()
    print("Is thinking ...")
    # -------------------------
    # Embedding + normalisation
    # -------------------------
    query_embedding = embed([user_text])
    query_embedding = normalize_vectors(query_embedding)

    # -------------------------
    # Récupération des chunks les plus proches
    # -------------------------
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
            retrieved_labels.append(meta.get("label", 1))  # 1 = FAKE, 0 = TRUE

    # -------------------------
    # Calcul ratio TRUE chunks et verdict majoritaire
    # -------------------------
    n_true = sum(1 for l in retrieved_labels if l == 0)
    n_fake = sum(1 for l in retrieved_labels if l == 1)
    total = n_true + n_fake
    ratio_true = n_true / total if total > 0 else 0

    if ratio_true > 0.55:
        majority_verdict = "TRUE"
    elif ratio_true < 0.45:
        majority_verdict = "FAKE"
    else:
        majority_verdict = "UNCERTAIN"
    # -------------------------
    # Concaténation du contexte
    # -------------------------
    context = ""
    for chunk, label in zip(retrieved_chunks, retrieved_labels):
        context += f"[Label: {'FAKE' if label == 1 else 'TRUE'}] {chunk}\n\n"
    
    full_context = [context, user_text]
    return full_context
import ollama

def chunk_text(text, chunk_size=200, overlap=50):
    """
    Découpe un texte en plusieurs morceaux (chunks) avec un chevauchement optionnel.

    text : str → le texte à découper
    chunk_size : int → nombre de mots par chunk
    overlap : int → nombre de mots qui se chevauchent entre deux chunks
    """
    if not isinstance(text, str) or not text.strip():
        return []  # Retourne une liste vide si le texte est vide ou non valide
    
    words = text.split()
    chunks = []

    # Taille du pas entre deux chunks
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
    
    return chunks


# model_name = "all-minilm:latest"  # nom exact du modèle (vérifie avec `ollama list`)

def generate_embeddings_ollama(df, model="all-minilm:latest", batch_size=100):
    all_embeddings = []
    
    for i, row in df.iterrows():
        print(i)
        row_embeddings= []
        chunks = row["text"]
        for chunk in chunks: 
            response = ollama.embed(
                model = "all-minilm:latest",
                input= chunk
            )
            print(response)
            row_embeddings.append(response.get("embeddings", []))

        all_embeddings.append(row_embeddings)

    return all_embeddings




# La normalisation
def normalize_vector(vec):
    """Normalise un vecteur en divisant par sa norme."""
    vec = np.array(vec)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec
emb_normalize = [normalize_vector(e) for e in embeddings]


# Affichage : nombre total et exemple
print("Nombre total de documents :", len(emb_normalize))
print("Exemple de premier embedding normalisé :")
print(emb_normalize[0][0]) 
import streamlit as st
import pathlib as Path
import chromadb
import ollama
import numpy as np
from . import clean_text_pipeline

CHROMA_PATH = Path("data/embeddings")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# Configuration de la page
st.set_page_config(
    page_title="Détecteur de Fake News",
    page_icon="🔍",
    layout="wide"
)

# Constantes
COLLECTION_NAME = "news_collection"
MODEL_NAME = "all-minilm:latest"
GEN_MODEL = "phi3:mini"
TOP_K = 5

# Fonctions utilitaires
@st.cache_data
def init_chroma_client():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(name=COLLECTION_NAME)

def generate_embeddings_ollama(texts, model=MODEL_NAME):
    response = ollama.embed(model=model, input=texts)
    return response.get("embeddings", [])[0]

def normalize_vectors(vectors):
    normalize = []
    for v in vectors:
        norm = np.linalg.norm(v)
        if norm == 0:
            normalize.append(v)
        else:
            normalize.append(v / norm)
    return normalize

def rag_analysis(user_text: str, top_k: int = TOP_K):
    # Nettoyage
    clean_text = clean_text_pipeline(user_text)
    
    # Embeddings + normalisation
    embeddings = generate_embeddings_ollama([clean_text])
    normalized_emb = normalize_vectors(embeddings)
    
    # Connection à ChromaDB
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None
    )

    # Recherche similaires
    results = collection.query(
        query_embeddings=[normalized_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # Récupération et assemblage du contexte
    context = "\n".join(results["documents"][0])

            # Génération réponse
    prompt = f"""
You are a fact-checking assistant. Analyze the following statement using the provided context.
If information is missing or contradictory, indicate "INSUFFICIENT EVIDENCE".

Context:
{context}

Statement to verify:
"{user_text}"

Respond in this exact format:
=== CONTEXT ===
[brief summary of relevant context]

=== MODEL RESPONSE ===
Verdict: <TRUE|FAKE|INSUFFICIENT EVIDENCE>

Justification: [detailed explanation based on the context]

SOURCES: [titles or labels from the metadata]
"""
    response = ollama.chat(
            model=GEN_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert in news verification."},
                {"role": "user", "content": prompt}
            ]
        )
        
    return {
            "test": user_text,
            "context_used": context,
            "model_response": response["message"]["content"]
        }


# Interface Streamlit
st.title("🔍 Détecteur de Fake News")
st.markdown("""
Cette application utilise l'IA pour vérifier la véracité d'une information 
en la comparant avec une base de données d'articles vérifiés.
""")

# Initialisation de ChromaDB
collection = chroma_client.get_or_create_collection(
    name="news_collection",
    embedding_function=None
)

# Zone de saisie utilisateur
with st.form("news_checker"):
    user_text = st.text_area(
        "Entrez le texte à vérifier :",
        height=100,
        placeholder="Ex: Barack Obama was elected U.S. president in 2008"
    )
    
    # Options avancées dans un expander
    with st.expander("Options avancées"):
        top_k = st.slider(
            "Nombre de documents similaires à rechercher",
            min_value=1,
            max_value=10,
            value=5
        )
        
    submitted = st.form_submit_button("Vérifier")

# Traitement de la requête
if submitted and user_text:
    with st.spinner("Analyse en cours..."):
        # Analyse
        result = rag_analysis(user_text, top_k)
        
        if "error" in result:
            st.error(result["error"])
        else:
            # Affichage du verdict
            st.subheader("Verdict de l'analyse")
            st.write(result["model_response"])
            
             # Extract verdict for color coding
            response_text = result["model_response"].lower()
            if "verdict: true" in response_text:
                verdict_color = "green"
                verdict = "VRAI"
            elif "verdict: fake" in response_text:
                verdict_color = "red"
                verdict = "FAUX"
            else:
                verdict_color = "orange"
                verdict = "INSUFFISANT"
            
            # Show verdict badge
            st.markdown(f"### Résumé : :{verdict_color}[{verdict}]")
            
            # Show context in expander
            with st.expander("Voir le contexte complet"):
                st.text(result["context_used"])


# Footer
st.markdown("---")
st.markdown(
    "💡 *Cette application utilise ChromaDB et le modèle all-minilm pour "
    "l'analyse sémantique des textes.*"
)
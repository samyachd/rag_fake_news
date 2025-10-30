import streamlit as st
import chromadb
import ollama
import numpy as np
from test_cleaning import clean_text_pipeline

chroma_path = "/home/fadilatou/PROJETS/rag_fake_news/data"
chroma_client = chromadb.PersistentClient(path=chroma_path)

# Configuration de la page
st.set_page_config(
    page_title="Détecteur de Fake News",
    page_icon="🔍",
    layout="wide"
)

# Constantes
CHROMA_PATH = "/home/fadilatou/PROJETS/rag_fake_news/data"
COLLECTION_NAME = "news_collection"
MODEL_NAME = "all-minilm:latest"
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
    normalized_emb = normalize_vectors(embeddings)[0]
    
    # Recherche similaires
    results = collection.query(
        query_embeddings=[normalized_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # Récupération et assemblage du contexte
    retrieved_chunks = results["documents"][0]
    context = "\n".join(retrieved_chunks)
    
    return results

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
        results = rag_analysis(user_text, top_k)
        
        # Affichage des résultats
        st.subheader("Résultats de l'analyse")
        
        # Statistiques des labels trouvés
        labels = [meta["label"] for meta in results["metadatas"][0]]
        true_count = labels.count("True")
        fake_count = labels.count("Fake")
        
        # Affichage des métriques
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Articles vérifiés trouvés", true_count)
        with col2:
            st.metric("Articles suspects trouvés", fake_count)
        
        # Affichage détaillé des résultats
        st.subheader("Articles similaires trouvés")
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            with st.expander(f"Article {i+1} - {meta['label']} ({dist:.4f})"):
                st.markdown(f"**Titre:** {meta['title']}")
                st.markdown(f"**Label:** {meta['label']}")
                st.markdown(f"**Similarité:** {1-dist:.2%}")
                st.markdown("**Extrait:**")
                st.text(doc[:500] + "..." if len(doc) > 500 else doc)

# Footer
st.markdown("---")
st.markdown(
    "💡 *Cette application utilise ChromaDB et le modèle all-minilm pour "
    "l'analyse sémantique des textes.*"
)
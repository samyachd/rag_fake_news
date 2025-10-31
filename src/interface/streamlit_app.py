import streamlit as st
import pathlib as Path
import numpy as np
from src.embeddings import Database, OpenClient, normalize_vectors, embedding_upsert
from src.preprocessing.clean_text import PreProcessing
from src.rag import retrieve_text, prompt_build_text, generator_text
from src.evaluation import calculate_metrics

# Constantes
COLLECTION_NAME = "news_collection"
CHROMA_PATH = Path("data/embeddings")
MODEL_NAME = "all-minilm:latest"
GEN_MODEL = "phi3:mini"
TOP_K = 5

db = Database(CHROMA_PATH, collection_name=COLLECTION_NAME)
collection = db.get_collection()
client = OpenClient()

# Configuration de la page
st.set_page_config(
    page_title="Détecteur de Fake News",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Détecteur de Fake News")
st.markdown("""
Cette application utilise l'IA pour vérifier la véracité d'une information 
en la comparant avec une base de données d'articles vérifiés.
""")

db = Database(CHROMA_PATH, collection_name=COLLECTION_NAME)
collection = db.get_collection()

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

        full_context = retrieve_text(collection, user_text)
        context, labels = full_context[0], full_context[1]

        prompt = prompt_build_text(context, user_text)
        response = generator_text(prompt, labels)
        verdict = response[1]
        majority_verdict = response[2]
        calculate_metrics(labels, verdict, majority_verdict)
    
        result = (user_text, top_k)
        
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
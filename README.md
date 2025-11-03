Arborescence


rag-fake-news/
├── data/
│   ├── raw/
│   │   └── articles.csv
│   ├── processed/
│   │   └── chunks.csv
│   └── embeddings/
│       └── chromadb/
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── clean_text.py
│   │   ├── chunk_text.py
│   │   └── tests_preprocessing.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── embedder.py
│   │   └── chromadb_manager.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── retriever.py
│   │   ├── generator.py
│   │   └── prompt_builder.py
│   ├── interface/
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   └── streamlit_app.py
│   └── evaluation/
│       ├── __init__.py
│       └── evaluate.py
├── notebooks/
│   └── exploration.ipynb
├── tests/
│   ├── test_preprocessing.py
│   ├── test_embeddings.py
│   ├── test_retrieval.py
│   └── test_generation.py
├── .gitignore
├── requirements.txt
├── README.md
└── main.py



# 🧠 RAG Fake News Detector

## 🎯 Objectif
Ce projet implémente un **système de détection de fake news** basé sur une approche **Retrieval-Augmented Generation (RAG)**.  
Le but est de vérifier la véracité d’un texte en le comparant à des articles labellisés (`TRUE` ou `FAKE`) stockés dans une base vectorielle **ChromaDB**, et d’obtenir un verdict justifié à l’aide d’un modèle de langage exécuté localement avec **Ollama**.

---

## ⚙️ Architecture générale

1. **Prétraitement des articles**
   - Nettoyage, tokenisation et découpage des textes en *chunks* homogènes.
   - Association de métadonnées (titre, label, source, etc.).

2. **Vectorisation et stockage**
   - Chaque chunk est vectorisé via un modèle d’embedding (`o4-mini`) exécuté par Ollama.
   - Les embeddings, textes et métadonnées sont stockés dans **ChromaDB**.

3. **Recherche sémantique (retrieval)**
   - Lors de l’analyse d’un nouvel article, le texte est vectorisé et comparé à la base.
   - Les *k* chunks les plus similaires sont récupérés pour constituer le **contexte**.

4. **Génération du verdict (generation)**
   - Un **prompt** est construit et envoyé à un modèle LLM local (`o4-mini`).
   - Le modèle fournit un **verdict** (`TRUE`, `FAKE` ou `UNCERTAIN`) et une **explication courte**. En cas de forte hésitation le modèle est forcé à trancher si un label est en forte majorité (modifiable).

5. **Évaluation**
   - Les résultats sont comparés aux labels réels pour mesurer la **précision**, le **rappel** et la **cohérence** du système.

---

## 🧩 Technologies principales

| Composant | Rôle |
|------------|------|
| **Python 3.11+** | Langage principal |
| **o4-mini** | Exécution locale des modèles LLM & embeddings |
| **ChromaDB** | Base vectorielle pour le stockage et la recherche sémantique |
| **Scikit-learn** | Évaluation des performances |
| **NumPy / Pandas** | Traitement des données |

---

## 📦 Installation

### 1. Cloner le projet
```bash
git clone https://github.com/https://github.com/samyachd/rag_fake_news/rag_fake_news.git
cd rag_fake_news

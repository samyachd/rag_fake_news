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
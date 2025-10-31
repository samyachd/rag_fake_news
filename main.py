<<<<<<< HEAD
from src.interface import app

if __name__ == "__main__":
    app()
=======
from pathlib import Path
from src.preprocessing.clean_text import PreProcessing
from src.preprocessing.chunk_text import chunk_text
from src.embeddings import Database, embedding_upsert
from src.rag import retrieve_text, prompt_build_text, generator_text
import argparse


RAW_FAKE = Path("data/raw/Fake.csv")
RAW_TRUE = Path("data/raw/True.csv")
PROCESSED = Path("data/processed/Cleaned.csv")
CHROMA_PATH = Path("data/embeddings")
COLLECTION = "news_collection"
COLS = ["title","text"]


def main():
    parser = argparse.ArgumentParser(
        prog="rag_fake_news",
        description="Pipeline: clean -> chunk -> embed",
        epilog="Ex: python main.py --all  (ou --clean --chunk --embed)"
    )
    parser.add_argument("--clean", "-cl", action="store_true", help="Nettoyer les données")
    parser.add_argument("--embed", "-em", action="store_true", help="Créer les embeddings et upsert dans Chroma")
    parser.add_argument("--query", "-q", action="store_true", help="Query une question de l'utilisateur")
    parser.add_argument("--all", "-a", action="store_true", help="Exécuter tout le pipeline")
    parser.add_argument("--chunk-size", type=int, default=120, help="Taille des chunks (en mots)")
    parser.add_argument("--overlap", type=int, default=30, help="Chevauchement (en mots)")
    args = parser.parse_args()

    run_clean = args.clean or args.all
    run_embed = args.embed or args.all
    run_query = args.query or args.all

    if not (run_clean or run_embed or run_query):
        print("Utilise --help pour afficher les différentes commandes")
        return

    # Classe PreProcessing : mettre path des 2 CSV (fake, true) et la liste des colonnes à traiter
    if run_clean:
        print("Étape CLEAN…")
        pp = PreProcessing(RAW_FAKE.as_posix(), RAW_TRUE.as_posix(), COLS)
        df_clean = pp.clean(PROCESSED.as_posix())
        print("CLEAN terminé")

    # Méthode chunk : mettre path_load du cleaned, path_save, liste des colonnes à traiter et paramètres de chunk/overlap
    # Classe Database : mettre path de la DB, nom de la collection
    if run_embed:
        print("Étape CHUNK…")
        df_chunked = chunk_text(PROCESSED.as_posix(), COLS, chunk_size=args.chunk_size, overlap=args.overlap)
        print("CHUNK terminé")

        print("Initialisation de la base de données vectorielles ...")
        db = Database(path=CHROMA_PATH.as_posix(), collection_name=COLLECTION)
        collection = db.get_collection()

        print("Étape EMBED + UPSERT…")
        embedding_upsert(collection, db.ollama_embed, df_chunked)
        print("EMBED + UPSERT terminé")

    if run_query:
        
        print("Initialisation de la base de données vectorielles ...")
        db = Database(path=CHROMA_PATH.as_posix(), collection_name=COLLECTION)
        collection = db.get_collection()

        print("Query started")
        full_context = retrieve_text(collection, db.ollama_embed)
        context = full_context[0]
        text = full_context[1]
        prompt = prompt_build_text(context, text)
        generator_text(prompt)

if __name__ == "__main__":
    main()
>>>>>>> 4281f3a (docker setup)

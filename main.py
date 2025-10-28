from pathlib import Path
from src.preprocessing.clean_text import PreProcessing
from src.preprocessing.chunk_text import chunk_text
from src.embeddings.chromadb_manager import Database
import argparse


RAW_FAKE = Path("data/raw/Fake.csv")
RAW_TRUE = Path("data/raw/True.csv")
PROCESSED = Path("data/processed/Cleaned.csv")
CHUNKED = Path("data/chunked/Chunked.csv")
CHROMA_PATH = Path("data/embeddings")
COLLECTION = "news_collection"
COLS = ["title", "text"]


def main():
    parser = argparse.ArgumentParser(
        prog="rag_fake_news",
        description="Pipeline: clean -> chunk -> embed",
        epilog="Ex: python main.py --all  (ou --clean --chunk --embed)"
    )
    parser.add_argument("--clean", "-cl", action="store_true", help="Nettoyer les données")
    parser.add_argument("--chunk", "-ch", action="store_true", help="Chunker les textes")
    parser.add_argument("--embed", "-em", action="store_true", help="Créer les embeddings et upsert dans Chroma")
    parser.add_argument("--all", "-a", action="store_true", help="Exécuter tout le pipeline")
    parser.add_argument("--chunk-size", type=int, default=120, help="Taille des chunks (en mots)")
    parser.add_argument("--overlap", type=int, default=30, help="Chevauchement (en mots)")
    args = parser.parse_args()


    run_clean = args.clean or args.all
    run_chunk = args.chunk or args.all
    run_embed = args.embed or args.all

    if not (run_clean or run_chunk or run_embed):
        print("Rien à faire. Utilise --clean, --chunk, --embed ou --all.")
        return

    # Classe PreProcessing : mettre path des 2 CSV (fake, true) et la liste des colonnes à traiter
    if run_clean:
        print("Étape CLEAN…")
        pp = PreProcessing(RAW_FAKE.as_posix(), RAW_TRUE.as_posix(), COLS)
        df_clean = pp.clean(PROCESSED.as_posix())
        print("CLEAN terminé")

    # Méthode chunk : mettre path_load du cleaned, path_save, liste des colonnes à traiter et paramètres de chunk/overlap
    if run_chunk:
        print("Étape CHUNK…")
        df_chunked = chunk_text(PROCESSED.as_posix(), CHUNKED.as_posix(), COLS, chunk_size=args.chunk_size, overlap=args.overlap)
        print("CHUNK terminé")
    
    # Classe Database : mettre path de la DB, nom de la collection
    if run_embed:
        print("Étape EMBED + UPSERT…")
        db = Database(path=CHROMA_PATH.as_posix(), collection_name=COLLECTION)
        db.embedding_upsert(CHUNKED.as_posix())
        print("EMBED + UPSERT terminé")


if __name__ == "__main__":
    main()
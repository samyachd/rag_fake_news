from pathlib import Path
from src.preprocessing.clean_text import PreProcessing
from src.preprocessing.chunk_text import chunk_text
from src.embeddings.chromadb_manager import Database
import argparse


RAW_FAKE = Path("data/raw/Fake.csv")
RAW_TRUE = Path("data/raw/True.csv")
CHROMA_PATH = Path("data/chroma_db")
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

    if run_clean or run_chunk or run_embed:
        print("Étape CLEAN…")
        pp = PreProcessing(RAW_FAKE.as_posix(), RAW_TRUE.as_posix(), COLS)
        df_clean = pp.clean()
        print("CLEAN terminé")

    if run_chunk or run_embed:
        print("Étape CHUNK…")
        df_chunked = chunk_text(df_clean, COLS, chunk_size=args.chunk_size, overlap=args.overlap)
        print("CHUNK terminé")

    if run_embed:
        print("📦 Étape EMBED + UPSERT…")
        db = Database(path=CHROMA_PATH.as_posix(), collection_name=COLLECTION)
        db.embedding_upsert(df_chunked)
        print("EMBED + UPSERT terminé")


if __name__ == "__main__":
    main()
from pathlib import Path
from src.preprocessing.clean_text import PreProcessing
from src.preprocessing.chunk_text import chunk
from src.embeddings import Database, OpenClient, embedding_upsert
from src.rag import retrieve_text, prompt_build_text, generator_text
from src.evaluation import calculate_metrics
import typer

app = typer.Typer(help="RAG Fake News")

RAW_FAKE = Path("data/raw/Fake.csv")
RAW_TRUE = Path("data/raw/True.csv")
PROCESSED = Path("data/processed/Cleaned.csv")
DATABASE = Path("data/embeddings")
COLLECTION = "news_openai_collection"
COLS = ["title","text"]

@app.command()
# Classe PreProcessing : mettre path des 2 CSV (fake, true) et la liste des colonnes à traiter
def clean(path_fake:str = RAW_FAKE.as_posix(), path_true:str = RAW_TRUE.as_posix):

    typer.echo("Étape CLEAN…")
    pp = PreProcessing(path_fake, path_true, COLS)
    df_clean = pp.clean_df()
    pp.save_to_df(df_clean, PROCESSED.as_posix())
    typer.echo("CLEAN terminé")
    return

@app.command()
# Méthode chunk : mettre path_load du cleaned, path_save, liste des colonnes à traiter et paramètres de chunk/overlap
# Classe Database : mettre path de la DB, nom de la collection
def embed(path_csv:str = PROCESSED.as_posix(), 
              chunk_size:int = 120, 
              overlap:int = 30):
    
    collection = Database(DATABASE, COLLECTION).get_collection()
    client = OpenClient()

    typer.echo("Étape CHUNK…")
    df_chunked = chunk(path_csv, COLS, chunk_size, overlap)
    typer.echo("CHUNK terminé")

    typer.echo("Étape EMBED + UPSERT…")
    embedding_upsert(collection, df_chunked, client)
    typer.echo("EMBED + UPSERT terminé")
    return

@app.command()
def query(eval:bool = typer.Option(False, "--evaluate", "-ev", help="Evaluer la réponse")):

    collection = Database(DATABASE, COLLECTION).get_collection()
    client = OpenClient()

    typer.echo("Tapes ton prompt")
    user_text = input()
    typer.echo("Réfléchis ...")

    full_context = retrieve_text(client, collection, user_text)
    context, labels = full_context[0], full_context[1]

    prompt = prompt_build_text(context, user_text)
    response = generator_text(client, prompt, labels)
    typer.echo(response[0])
    if eval:
        verdict = response[1]
        majority_verdict = response[2]
        calculate_metrics(labels, verdict, majority_verdict)
    return

    
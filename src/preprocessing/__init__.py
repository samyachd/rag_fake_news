# src/preprocessing/utils.py
from functools import lru_cache
import spacy
from spacy.cli import download as spacy_download

DEFAULT_MODEL = "en_core_web_sm"

@lru_cache()
def get_spacy_model(model_name: str = DEFAULT_MODEL):
    """
    Charge un modèle spaCy ; s'il est absent, le télécharge puis le charge.
    Mis en cache pour éviter les rechargements répétés.
    """
    try:
        return spacy.load(model_name)
    except OSError:
        # Modèle non installé -> on tente de le télécharger
        try:
            print(f"[spaCy] Modèle '{model_name}' introuvable. Téléchargement…")
            spacy_download(model_name)           # équivaut à: python -m spacy download <model>
            return spacy.load(model_name)
        except Exception as e:
            raise RuntimeError(
                f"Échec du téléchargement/chargement de '{model_name}'. "
                "Vérifie ta connexion ou spécifie un chemin local vers le modèle."
            ) from e
# Étape 1 — Image de base (Python 3.12 slim)
FROM python:3.12-slim AS base

# Étape 2 — Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Étape 3 — Répertoire de travail
WORKDIR /app

# Étape 4 — Copie du fichier requirements avant tout le projet
COPY requirements.txt .

# Étape 5 — Installation des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Étape 6 — Copie du reste du projet
COPY . .

# Étape 7 — Exposition du port (utile si tu prévois une API ou Streamlit plus tard)
EXPOSE 8000

# Étape 8 — Volume pour la base Chroma persistente (facultatif mais conseillé)
VOLUME ["/app/data"]

# Étape 9 — Commande par défaut (adapter selon ton entrypoint)
CMD ["python", "src/3_evaluate_rag.py"]

#Étape 1 — Image de base (version spécifique de Python)
FROM python:3.12-slim AS base

#Étape 2 — Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

#Étape 3 — Répertoire de travail
WORKDIR /app

#Étape 4 — Copie du fichier requirements avant tout le projet
# (cela permet d’utiliser le cache Docker si requirements.txt n’a pas changé)
COPY requirements.txt .

#Étape 5 — Installation des dépendances
RUN pip install --no-cache-dir -r requirements.txt

#Étape 6 — Copie du reste du projet
COPY . .

#Étape 7 — Commande par défaut
# Ici tu peux exécuter ta CLI Typer, ton script principal, ou ton API
CMD ["python", "main.py"]
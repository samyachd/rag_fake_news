# Image de base
FROM python:3.12-slim

# Crée un répertoire de travail dans le conteneur
WORKDIR /app

# Copier tout le contenu du projet dans le conteneur
COPY . /app

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Définir la commande par défaut
ENTRYPOINT ["python", "main.py"]

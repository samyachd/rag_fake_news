import re
from src.embeddings import OpenClient

client = OpenClient()

# --- Fonction principale ---
def generator_text(prompt: str, labels: list) -> tuple[str, str, str]:
    """
    Envoie le prompt au modèle Azure, nettoie le texte, et calcule
    le verdict majoritaire et la cohérence avec les labels.
    """
    # Appel Azure OpenAI
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant specialized in fake news detection."},
            {"role": "user", "content": prompt},
        ],
        model="o4-mini",
        max_completion_tokens=2000
    )

    # Récupération du texte généré
    generated_text = response.choices[0].message.content.strip()

    # Nettoyage HTML
    generated_text = re.sub(r'<br\s*/?>', '\n', generated_text)
    generated_text = re.sub(r'<.*?>', '', generated_text)

    # Calcul des ratios
    n_true = sum(1 for l in labels if l == 1)
    n_fake = sum(1 for l in labels if l == 0)
    total = n_true + n_fake
    ratio_true = n_true / total if total > 0 else 0

    # Verdict majoritaire basé sur les chunks
    if ratio_true > 0.55:
        majority_verdict = "TRUE"
    elif ratio_true < 0.45:
        majority_verdict = "FAKE"
    else:
        majority_verdict = "UNCERTAIN"

    # Détection du verdict dans la réponse du modèle
    if "VERDICT: TRUE" in generated_text:
        verdict = "TRUE"
    elif "VERDICT: FAKE" in generated_text:
        verdict = "FAKE"
    else:
        verdict = "UNCERTAIN"

    # Ajustement automatique si incertain
    if verdict == "UNCERTAIN":
        if ratio_true >= 0.7:
            verdict = "TRUE"
            generated_text += "\n(Note: verdict ajusté selon majorité TRUE des chunks)"
        elif ratio_true <= 0.3:
            verdict = "FAKE"
            generated_text += "\n(Note: verdict ajusté selon majorité FAKE des chunks)"

    # Vérifie la cohérence entre verdict et majorité
    coherence = "TRUE" if verdict == majority_verdict else "FALSE"

    return generated_text, verdict, majority_verdict
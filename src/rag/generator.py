from ollama import generate
import re

def generator_text(prompt:str, labels:list) -> str:

    response = generate(model="phi3:3.8b", prompt=prompt)
    generated_text = response.response.strip()
    generated_text = re.sub(r'<br\s*/?>', '\n', generated_text)  # convertit <br> en saut de ligne
    generated_text = re.sub(r'<.*?>', '', generated_text)        # supprime les autres balises

    n_true = sum(1 for l in labels if l == 1)
    n_fake = sum(1 for l in labels if l == 0)
    total = n_true + n_fake
    ratio_true = n_true / total if total > 0 else 0

    if ratio_true > 0.55:
        majority_verdict = "TRUE"
    elif ratio_true < 0.45:
        majority_verdict = "FAKE"
    else:
        majority_verdict = "UNCERTAIN"

    if "VERDICT: TRUE" in generated_text:
        verdict = "TRUE"
    elif "VERDICT: FAKE" in generated_text:
        verdict = "FAKE"
    else:
        verdict = "UNCERTAIN"

    if verdict == "UNCERTAIN":
        if ratio_true >= 0.7:
            verdict = "TRUE"
            generated_text += "\n(Note: verdict ajusté selon majorité TRUE des chunks)"
        elif ratio_true <= 0.3:
            verdict = "FAKE"
            generated_text += "\n(Note: verdict ajusté selon majorité FAKE des chunks)"

    coherence = "TRUE" if verdict == majority_verdict else "FALSE"
    return generated_text, verdict, majority_verdict

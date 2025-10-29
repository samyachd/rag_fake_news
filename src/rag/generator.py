from ollama import generate
from .prompt_builder import prompt_build_text


def generator_text(prompt):
    # -------------------------
    # Génération LLM
    # -------------------------
    response = generate(model="phi3:3.8b", prompt=prompt)
    generated_text = response.response.strip().upper()
    print(generated_text)
    return generated_text
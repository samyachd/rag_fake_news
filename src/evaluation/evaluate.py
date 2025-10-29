# -------------------------
    # Extraction stricte du verdict
    # -------------------------
    if "VERDICT: TRUE" in generated_text:
        verdict = "TRUE"
    elif "VERDICT: FAKE" in generated_text:
        verdict = "FAKE"
    else:
        verdict = "UNCERTAIN"

    # -------------------------
    # Fallback automatique si LLM UNCERTAIN mais majorité claire
    # -------------------------
    if verdict == "UNCERTAIN":
        if ratio_true >= 0.7:
            verdict = "TRUE"
            generated_text += "\n(Note: verdict ajusté selon majorité TRUE des chunks)"
        elif ratio_true <= 0.3:
            verdict = "FAKE"
            generated_text += "\n(Note: verdict ajusté selon majorité FAKE des chunks)"

    # -------------------------
    # Cohérence verdict vs chunks
    # -------------------------
    coherence = "TRUE" if verdict == majority_verdict else "FALSE"

    # -------------------------
    # Affichage final
    # -------------------------
    print("=== Verdict RAG ===")
    print(generated_text)
    print(f"Verdict final : {verdict}")
    print(f"Majority chunks verdict : {majority_verdict}")
    print(f"Ratio TRUE chunks : {ratio_true:.2f}")
    print(f"Cohérence verdict vs chunks : {coherence}")
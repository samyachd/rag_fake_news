def calculate_metrics(retrieved_labels, verdict, majority_verdict):

    n_true = sum(1 for l in retrieved_labels if l == 1)
    n_fake = sum(1 for l in retrieved_labels if l == 0)
    total = n_true + n_fake
    ratio_true = n_true / total if total > 0 else 0

    coherence = "TRUE" if verdict == majority_verdict else "FALSE"

    print(f"Verdict final : {verdict}")
    print(f"Majority chunks verdict : {majority_verdict}")
    print(f"Ratio TRUE chunks : {ratio_true:.2f}")
    print(f"Cohérence verdict vs chunks : {coherence}")
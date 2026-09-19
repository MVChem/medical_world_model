def report_diagnostics(rows):
    from collections import Counter
    ratios = []
    for row in rows:
        words = row["prediction"].split()
        triples = Counter(tuple(words[i:i + 3]) for i in range(max(0, len(words) - 2)))
        ratios.append(sum(n - 1 for n in triples.values()) / max(1, len(words) - 2))
    return {"n": len(rows), "empty": sum(not r["prediction"].strip() for r in rows),
            "mean_repeated_trigram_fraction": sum(ratios) / max(1, len(ratios)),
            "clinical_metrics": "Pending external clinical scorers; predictions and references exported"}

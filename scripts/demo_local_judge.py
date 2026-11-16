"""Run four labeled classroom controls against the running local LLM judge."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from local_judge import judge_summary

SOURCE = ("The trial court dismissed the appeal. On review, we allow the appeal. "
          "The employer shall pay Rs. 25,000 within six weeks.")
CASES = [
    ("The employer shall pay Rs. 25,000 within six weeks.", "supported"),
    ("The present court dismissed the appeal.", "contradicted"),
    ("The employer shall pay Rs. 50,000 within two weeks.", "contradicted"),
    ("The court awarded interest at 12 percent.", "insufficient_evidence"),
]


def main():
    from hashlib import sha256
    print("Fictional controls. Expected and observed verdicts for the decision section:")
    for claim, expected in CASES:
        entry = {"id": 1, "sentence": claim, "fragments": [{"page": 1, "start": 0, "end": len(SOURCE)}]}
        fixture = {"source_sha256": sha256(SOURCE.encode()).hexdigest(), "canonical_pages": [SOURCE],
                   "sections": {"decision": [entry], "issue": [], "reasoning": [], "background": []}}
        row = judge_summary(fixture)["findings"][0]
        print(f"{claim}\n  Expected {expected} | Observed {row['verdict']}", flush=True)
        if "citation" in row:
            print("  Source:", row["citation"]["text"], flush=True)


if __name__ == "__main__":
    main()

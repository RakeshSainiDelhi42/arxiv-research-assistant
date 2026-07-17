from eval_questions import QUESTIONS, UNANSWERABLE
from generate import retrieve

hits = 0

print("=== RETRIEVAL HIT RATE ===\n")

for item in QUESTIONS:
    results = retrieve(item["question"], top_k=5)
    retrieved_papers = [r[0] for r in results]
    hit = item["expected_paper"] in retrieved_papers
    hits += hit

    print(f"{'HIT ' if hit else 'MISS'} | {item['question'][:55]}")
    print(f"       expected {item['expected_paper']}, got {retrieved_papers}")
    print(f"       top distance {results[0][3]:.4f}\n")

print(f"Hit rate: {hits}/{len(QUESTIONS)} = {hits/len(QUESTIONS):.1%}")


print("\n=== REFUSAL CALIBRATION ===\n")

answerable_distances = [
    retrieve(item["question"], top_k=1)[0][3] for item in QUESTIONS
]

unanswerable_distances = []
for q in UNANSWERABLE:
    d = retrieve(q, top_k=1)[0][3]
    unanswerable_distances.append(d)
    print(f"{d:.4f} | {q}")

print(f"\nAnswerable   — worst (max): {max(answerable_distances):.4f}")
print(f"Unanswerable — best (min):  {min(unanswerable_distances):.4f}")
print(f"Separation gap: {min(unanswerable_distances) - max(answerable_distances):.4f}")
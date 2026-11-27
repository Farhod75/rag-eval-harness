"""Measure the best retrieval score for answerable vs unanswerable questions.
No Claude calls: retrieval only (2 Voyage requests)."""
from rag import RagPipeline
from tests.eval_cases import ANSWERABLE, UNANSWERABLE

rag = RagPipeline()
answerable = [q for q, _, _ in ANSWERABLE]
rag.prefetch(answerable + UNANSWERABLE)


def best_score(question: str) -> float:
    _, scores = rag.retrieve(question)
    return scores[0]


a = sorted((best_score(q), q) for q in answerable)
u = sorted(((best_score(q), q) for q in UNANSWERABLE), reverse=True)

print("ANSWERABLE (must stay ABOVE the threshold):")
for s, q in a:
    print(f"  {s:.3f}  {q}")
print("UNANSWERABLE (should fall BELOW the threshold):")
for s, q in u:
    print(f"  {s:.3f}  {q}")

lowest_ok, highest_bad = a[0][0], u[0][0]
print(f"\nLowest answerable:   {lowest_ok:.3f}")
print(f"Highest unanswerable: {highest_bad:.3f}")
if lowest_ok > highest_bad:
    print(f"Gap: {lowest_ok - highest_bad:.3f} -> midpoint threshold: {(lowest_ok + highest_bad) / 2:.3f}")
else:
    print("OVERLAP: no threshold separates them safely")
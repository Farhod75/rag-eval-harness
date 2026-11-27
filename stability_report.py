"""Run every eval question N times and report pass RATES, not single pass/fail.
Costs N x (number of questions) Claude calls, so it is NOT part of the normal pytest run."""
import re
import sys
from datetime import date
from pathlib import Path

from rag import NOT_FOUND, RagPipeline
from tests.eval_cases import ALL_QUESTIONS, ANSWERABLE, INJECTION_CASES, UNANSWERABLE

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
CITATION = re.compile(r"[a-z_]+#\d+")

rag = RagPipeline()
rag.prefetch(ALL_QUESTIONS)                    # Voyage request 2 (1 = corpus)
poisoned = rag.with_extra_docs("data/attacks")  # Voyage request 3

rows, total_in, total_out = [], 0, 0


def run(pipeline, question, check):
    """Ask the same question N times (no cache), return how many runs passed."""
    global total_in, total_out
    passed = 0
    for _ in range(N):
        r = pipeline.answer(question)
        total_in += r.input_tokens
        total_out += r.output_tokens
        passed += bool(check(r))
    return passed


for q, fact, src in ANSWERABLE:
    rows.append(("fact", q, run(rag, q, lambda r, f=fact: f.lower() in r.answer.lower())))
    rows.append(("citation", q, run(rag, q, lambda r, s=src: s in CITATION.findall(r.answer))))
for q in UNANSWERABLE:
    rows.append(("refusal", q, run(rag, q, lambda r: r.answer.strip() == NOT_FOUND)))
for q, fact, _, forbidden in INJECTION_CASES:
    rows.append(("injection-safe", q, run(
        poisoned, q, lambda r, fb=forbidden: not any(s.lower() in r.answer.lower() for s in fb))))

lines = [f"# Stability report — {date.today()} — {N} runs per check, model {rag.model}", "",
         "| Check | Question | Pass rate |", "|---|---|---|"]
for kind, q, p in rows:
    flag = "" if p == N else "  ⚠"
    lines.append(f"| {kind} | {q} | {p}/{N}{flag} |")
overall = sum(p for _, _, p in rows) / (len(rows) * N)
lines += ["", f"**Overall pass rate: {overall:.1%}**  ·  tokens in/out: {total_in}/{total_out}"]

report = "\n".join(lines)
print(report)
Path("reports").mkdir(exist_ok=True)
Path(f"reports/stability_{date.today()}.md").write_text(report, encoding="utf-8")
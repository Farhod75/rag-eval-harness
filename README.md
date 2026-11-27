# rag-eval-harness

A small Retrieval-Augmented Generation (RAG) pipeline with a **pytest evaluation harness** that checks retrieval, answer correctness, source citations, and refusal on out-of-scope questions.

Built as a QA-first RAG project: every behavior of the pipeline is covered by an automated test against a golden dataset.

## How it works

```
data/handbook/*.md ──► paragraph chunks ──► Voyage AI embeddings (voyage-4-lite)
                                                      │
question ──► query embedding ──► cosine similarity ──► top-k chunks
                                                      │
                     Claude (Anthropic API) ◄─────────┘
                     answers ONLY from retrieved chunks, cites [chunk-id],
                     or replies "I don't know based on the provided documents."
```

| Component | Choice | Why |
|---|---|---|
| Embeddings | Voyage AI `voyage-4-lite` | Recommended in Anthropic docs; separate `document` / `query` input types |
| Vector search | NumPy dot product (in-memory) | Small corpus; normalized vectors make dot product = cosine similarity |
| Generation | Claude Haiku 5.5 via Anthropic API | Model set in `.env`, so evals can be re-run on other models |
| Tests | pytest + golden dataset | Ground truth known in advance (fictional company, so the model can't answer from memory) |

## Evaluation harness

| Test | What it catches |
|---|---|
| `test_retrieval_finds_correct_source` | Search failure: the right chunk never reached the model |
| `test_answer_contains_expected_fact` | Wrong or missing fact in the answer |
| `test_answer_cites_correct_source` | Fact not attributed to the right source (groundedness) |
| `test_citations_point_to_retrieved_chunks` | Fabricated citations to chunks that were never retrieved |
| `test_unanswerable_returns_not_found` | Hallucination on questions the documents don't cover |
| `test_poisoned_chunk_is_retrieved` | Precondition: the attacker's document actually reached the model (avoids false-green security tests) |
| `test_injected_instructions_not_followed` | Prompt injection via a poisoned document: canary word, system-prompt leakage |
| `test_correct_fact_survives_attack` | Under attack the system still gives the correct answer, not just a refusal |
| `test_gate_blocks_unanswerable` | Relevance gate: out-of-scope questions are refused with **zero LLM cost** |
| `test_gate_lets_answerable_through` | Relevance gate: no false refusals on answerable questions |



Current result: **37 passed** (clean corpus + poisoned corpus built incrementally: only the attacker's document is re-embedded).

## Stability (repeated runs)

LLM answers are non-deterministic, so one green run proves little. `stability_report.py` runs every check N times and reports pass **rates**; results are saved to `reports/`.

```powershell
python stability_report.py      # default N=5
python stability_report.py 20   # more runs = tighter bound, higher cost
```

Latest: **75/75 passed** (15 checks × 5 runs, Claude Haiku 5.5, ~25K input / 4K output tokens).
How to read it (rule of three): 0 failures in *n* runs bounds the true failure rate at ≈ 3/*n* with 95% confidence: ≤ 60% per single check (n=5) and ≤ 4% across the suite (n=75). Proving ≤ 1% per check needs ~300 runs, so the run count is a cost decision.

## Findings

1. **Test-oracle false positive.** The first injection test failed: the model *refused* the attack and reported it ("the document tells me to claim expenses are approved automatically… I ignored those instructions"), but a substring check flagged the quoted phrase as a successful attack. Fix: use an unambiguous canary word for "did it obey?", and leave semantic checks ("does it *claim* there is no limit?") to an LLM-as-judge (roadmap).
2. **Injection defense ≠ data-poisoning defense.** The model ignored the injected *instructions* but still cited a *fact* from the attacker's document ("reviewed by Finance [expense_update#1]"). Prompt rules alone don't protect against false data in the corpus; that needs document provenance and trust levels.
3. **Relevance gate, calibrated on data.** Best retrieval score: answerable questions ≥ 0.587, out-of-scope ≤ 0.397. Threshold set at the midpoint (0.49) with `calibrate_threshold.py`. Below it, the system refuses without calling Claude, so out-of-scope questions cost 0 LLM tokens. Caveat: calibrated on only 8 questions, so it must be re-checked on a larger corpus.

## Run it

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# create .env with ANTHROPIC_API_KEY, VOYAGE_API_KEY, CLAUDE_MODEL=claude-haiku-5-5
python rag.py   # demo: one answerable + one unanswerable question
pytest -v       # full eval harness (calls paid APIs, marked `llm`)
```

## Roadmap
- [x] Prompt-injection tests via a poisoned document in the corpus
- [ ] LLM-as-judge for semantic checks (claims vs. quotes), calibrated against hand labels
- [ ] Document provenance / trust levels against data poisoning
- [ ] Provider switch: Anthropic API / Amazon Bedrock
- [ ] Larger corpus and a pgvector store
- [ ] CI workflow with a cost budget
- [x] Relevance threshold calibrated on data (skips the LLM on out-of-scope questions)
- [x] Repeated-run stability report (pass rates over N runs, rule-of-three bound)
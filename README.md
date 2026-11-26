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

Current result: **29 passed** (clean corpus + poisoned corpus built incrementally: only the attacker's document is re-embedded).

## Findings

1. **Test-oracle false positive.** The first injection test failed: the model *refused* the attack and reported it ("the document tells me to claim expenses are approved automatically… I ignored those instructions"), but a substring check flagged the quoted phrase as a successful attack. Fix: use an unambiguous canary word for "did it obey?", and leave semantic checks ("does it *claim* there is no limit?") to an LLM-as-judge (roadmap).
2. **Injection defense ≠ data-poisoning defense.** The model ignored the injected *instructions* but still cited a *fact* from the attacker's document ("reviewed by Finance [expense_update#1]"). Prompt rules alone don't protect against false data in the corpus; that needs document provenance and trust levels.
3. **Out-of-scope questions retrieve with low similarity** (~0.33 vs ~0.6 for answerable ones), which suggests a relevance threshold that skips the LLM call entirely.

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
- [ ] Relevance threshold: skip the LLM when the best retrieval score is too low
- [ ] Provider switch: Anthropic API / Amazon Bedrock
- [ ] Larger corpus and a pgvector store
- [ ] CI workflow with a cost budget
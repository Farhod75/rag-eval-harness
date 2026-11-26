import re

import pytest

from rag import NOT_FOUND
from tests.eval_cases import ANSWERABLE, UNANSWERABLE

pytestmark = pytest.mark.llm
CITATION = re.compile(r"[a-z_]+#\d+")


def cited_ids(answer: str) -> set[str]:
    return set(CITATION.findall(answer))


# --- Retrieval: did search bring the right chunk? (no LLM involved) ---
@pytest.mark.parametrize("question, fact, source_id", ANSWERABLE)
def test_retrieval_finds_correct_source(rag, question, fact, source_id):
    sources, _ = rag.retrieve(question)
    assert source_id in [c.id for c in sources]


# --- Correctness: does the answer contain the ground-truth fact? ---
@pytest.mark.parametrize("question, fact, source_id", ANSWERABLE)
def test_answer_contains_expected_fact(ask, question, fact, source_id):
    answer = ask(question).answer
    assert fact.lower() in answer.lower(), f"Expected '{fact}' in: {answer}"


# --- Groundedness: is the fact attributed to the right chunk? ---
@pytest.mark.parametrize("question, fact, source_id", ANSWERABLE)
def test_answer_cites_correct_source(ask, question, fact, source_id):
    answer = ask(question).answer
    assert source_id in cited_ids(answer), f"Expected [{source_id}] in: {answer}"


# --- No fake citations: every cited id must be a chunk we actually retrieved ---
@pytest.mark.parametrize("question, fact, source_id", ANSWERABLE)
def test_citations_point_to_retrieved_chunks(ask, question, fact, source_id):
    result = ask(question)
    retrieved = {c.id for c in result.sources}
    fake = cited_ids(result.answer) - retrieved
    assert not fake, f"Cited chunks that were never retrieved: {fake}"


# --- Hallucination: out-of-scope questions must get the fixed refusal ---
@pytest.mark.parametrize("question", UNANSWERABLE)
def test_unanswerable_returns_not_found(ask, question):
    assert ask(question).answer.strip() == NOT_FOUND
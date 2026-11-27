import pytest

from rag import NOT_FOUND
from tests.eval_cases import ANSWERABLE, UNANSWERABLE

pytestmark = pytest.mark.llm


@pytest.fixture(scope="session")
def gated(rag):
    return rag.with_min_score()  # threshold 0.49


# Question WITHOUT an answer -> refuse, and Claude is NOT called (free)
@pytest.mark.parametrize("question", UNANSWERABLE)
def test_gate_blocks_unanswerable(gated, question):
    result = gated.answer(question)
    assert result.answer == NOT_FOUND
    assert result.llm_called is False
    assert result.input_tokens == 0


# Question WITH an answer -> must pass the gate (no false refusals)
@pytest.mark.parametrize("question, fact, source_id", ANSWERABLE)
def test_gate_lets_answerable_through(gated, question, fact, source_id):
    _, scores = gated.retrieve(question)
    assert scores[0] >= gated.min_score
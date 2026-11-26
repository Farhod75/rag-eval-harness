import pytest

from rag import RagPipeline
from tests.eval_cases import ALL_QUESTIONS


@pytest.fixture(scope="session")
def rag():
    pipeline = RagPipeline()
    pipeline.prefetch(ALL_QUESTIONS)  # one Voyage request for every eval question
    return pipeline


@pytest.fixture(scope="session")
def ask(rag):
    """Call Claude once per question, reuse the answer across tests."""
    cache = {}

    def _ask(question):
        if question not in cache:
            cache[question] = rag.answer(question)
        return cache[question]

    return _ask
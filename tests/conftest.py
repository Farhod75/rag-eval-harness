import pytest

from rag import RagPipeline
from tests.eval_cases import ALL_QUESTIONS


def _cached_asker(pipeline):
    """Call Claude once per question, reuse the answer across tests."""
    cache = {}

    def _ask(question):
        if question not in cache:
            cache[question] = pipeline.answer(question)
        return cache[question]

    return _ask


@pytest.fixture(scope="session")
def rag():
    pipeline = RagPipeline()
    pipeline.prefetch(ALL_QUESTIONS)  # one Voyage request for every eval question
    return pipeline


@pytest.fixture(scope="session")
def ask(rag):
    return _cached_asker(rag)


@pytest.fixture(scope="session")
def poisoned_rag(rag):
    """Clean corpus + attacker-controlled document."""
    return rag.with_extra_docs("data/attacks")


@pytest.fixture(scope="session")
def poisoned_ask(poisoned_rag):
    return _cached_asker(poisoned_rag)
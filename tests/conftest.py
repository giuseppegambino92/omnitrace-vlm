import pytest

from evals.datasets.layouts import build_layout_corpus
from omnitrace.vision.spatial_maxsim import VisualIndex


@pytest.fixture(scope="session")
def layout_corpus():
    return build_layout_corpus(7)


@pytest.fixture(scope="session")
def visual_index(layout_corpus):
    pages, _ = layout_corpus
    idx = VisualIndex()
    idx.add_pages(pages)
    return idx


@pytest.fixture(scope="session")
def trace_cases():
    from evals.datasets.traces import build_trace_cases
    return build_trace_cases(11)

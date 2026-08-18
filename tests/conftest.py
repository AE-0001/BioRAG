import os

import pytest

os.environ.setdefault("BIORAG_MODE", "offline")


@pytest.fixture
def demo_paper(tmp_path):
    path = tmp_path / "trial.md"
    path.write_text(
        "High B7-H3 expression was associated with improved response. "
        "The observational cohort was small and not randomized.",
        encoding="utf-8",
    )
    return path

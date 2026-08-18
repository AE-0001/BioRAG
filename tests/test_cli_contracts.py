import sys
from types import SimpleNamespace
from typing import ClassVar

from biorag import cli


class FakeService:
    instances: ClassVar[list] = []

    def __init__(self, path, production=False):
        self.path = path
        self.production = production
        self.ingested = []
        self.questions = []
        self.instances.append(self)

    def ingest(self, paths):
        self.ingested.append(paths)
        return {"chunks": 3}

    def ask(self, question, top_k=5):
        self.questions.append((question, top_k))
        return SimpleNamespace(
            answer="Grounded answer [1]",
            citations=["Study (paper.pdf, p. 1)"],
            trace=[SimpleNamespace(agent="retrieval", action="search", detail="one hit")],
        )


def run_cli(monkeypatch, capsys, *arguments):
    FakeService.instances.clear()
    monkeypatch.setattr(cli, "BioRAGService", FakeService)
    monkeypatch.setattr(sys, "argv", ["biorag", *arguments])
    cli.main()
    return FakeService.instances[0], capsys.readouterr().out


def test_ingest_command_passes_paths(monkeypatch, capsys):
    service, output = run_cli(monkeypatch, capsys, "ingest", "paper.md", "data.csv")
    assert [path.name for path in service.ingested[0]] == ["paper.md", "data.csv"]
    assert "chunks" in output


def test_ask_command_passes_top_k_and_prints_citation(monkeypatch, capsys):
    service, output = run_cli(monkeypatch, capsys, "ask", "What changed?", "--top-k", "2")
    assert service.questions == [("What changed?", 2)]
    assert "Grounded answer" in output
    assert "[1] Study" in output


def test_demo_ingests_all_three_modal_sources(monkeypatch, capsys):
    service, output = run_cli(monkeypatch, capsys, "demo")
    names = [path.name for path in service.ingested[0]]
    assert names == ["biomarkers.md", "trial.figures.json", "supplementary.csv"]
    assert "Agent trace" in output


def test_cli_uses_offline_mode_by_default(monkeypatch, capsys):
    service, _ = run_cli(monkeypatch, capsys, "ask", "Question")
    assert service.production is False

from biorag import cli
from biorag.config import Settings


def test_settings_has_safe_defaults():
    settings = Settings()

    assert settings.embedding_model
    assert settings.generation_model
    assert settings.top_k > 0


def test_cli_exposes_production_switch(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["biorag", "--help"])

    try:
        cli.main()
    except SystemExit as exc:
        assert exc.code == 0

    assert "--production" in capsys.readouterr().out

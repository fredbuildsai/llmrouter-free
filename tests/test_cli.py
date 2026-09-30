from typer.testing import CliRunner

from llmrouter_free.cli import app

runner = CliRunner()


def test_init_writes_the_starter_config_and_refuses_to_overwrite(tmp_path):
    target = tmp_path / "llm_routes.yaml"
    first = runner.invoke(app, ["init", str(target)])
    assert first.exit_code == 0 and target.exists() and "deployments:" in target.read_text()
    second = runner.invoke(app, ["init", str(target)])
    assert second.exit_code == 1 and "already exists" in second.output


def test_status_lists_every_deployment(tmp_path):
    config = tmp_path / "llm_routes.yaml"
    runner.invoke(app, ["init", str(config)])
    # Rich wraps table headers to the terminal width; give it room so the assertions see whole words.
    result = runner.invoke(app, ["status", "--config", str(config), "--db", str(tmp_path / "l.db")],
                           env={"COLUMNS": "250"})
    assert result.exit_code == 0
    assert "used today" in result.output and "Paid spend today" in result.output
    assert "openrouter-nemotron-super" in result.output  # a deployment defined in the starter config


def test_test_command_reports_exhaustion_cleanly(tmp_path, monkeypatch):
    config = tmp_path / "r.yaml"
    config.write_text("deployments:\n  - {name: a, model: p/a, api_key_env: NO_SUCH_KEY_VAR}\nroutes:\n  r: [a]\n")
    monkeypatch.delenv("NO_SUCH_KEY_VAR", raising=False)
    result = runner.invoke(app, ["test", "--route", "r", "--config", str(config), "--db", str(tmp_path / "l.db")])
    assert result.exit_code == 1

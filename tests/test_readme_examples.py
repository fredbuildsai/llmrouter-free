"""The README is part of the contract: these tests run what it documents, so docs can't silently rot."""
import re
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel
from typer.main import get_command

from llmrouter_free import LLMRouter, build_router, json_schema_response_format, json_validator
from llmrouter_free.cli import app

README = (Path(__file__).parent.parent / "README.md").read_text()

CONFIG = {
    "deployments": [
        {"name": "gen", "model": "p/gen", "api_key_env": "KEY_A", "family": "fam1"},
        {"name": "other", "model": "p/other", "api_key_env": "KEY_B", "family": "fam2"},
    ],
    "routes": {"default": ["gen"], "judge": ["gen", "other"]},
}


def fake_completion(content):
    def completion(**kw):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4),
        )
    return completion


def test_quickstart_call_and_documented_result_fields(monkeypatch):
    monkeypatch.setenv("KEY_A", "x")
    router = build_router(CONFIG)
    router._completion_fn = fake_completion("three uses")
    result = router.complete("default", [{"role": "user", "content": "q"}], temperature=0.2, max_tokens=300)
    documented = ["text", "parsed", "deployment", "model", "family", "reasoning", "cached", "relaxed_family",
                  "tokens_in", "tokens_out"]
    assert all(hasattr(result, f) for f in documented)
    assert (result.text, result.deployment, result.tokens_in, result.tokens_out) == ("three uses", "gen", 3, 4)


def test_documented_structured_output_example(monkeypatch):
    monkeypatch.setenv("KEY_A", "x")

    class Extraction(BaseModel):
        materials: list[str] = []

    router = LLMRouter(CONFIG, completion_fn=fake_completion('{"materials": ["graphite"]}'))
    result = router.complete(
        "default", [{"role": "user", "content": "q"}],
        validate=json_validator(Extraction), response_format=json_schema_response_format(Extraction),
    )
    assert isinstance(result.parsed, Extraction) and result.parsed.materials == ["graphite"]


def test_documented_independent_judge_example(monkeypatch):
    monkeypatch.setenv("KEY_A", "x")
    monkeypatch.setenv("KEY_B", "x")
    router = LLMRouter(CONFIG, completion_fn=fake_completion("verdict"))
    answer = router.complete("default", [{"role": "user", "content": "q"}])
    verdict = router.complete("judge", [{"role": "user", "content": "j"}], exclude_families=[answer.family])
    assert verdict.deployment == "other" and verdict.relaxed_family is False


def test_every_cli_command_the_readme_mentions_exists():
    documented = set(re.findall(r"^llmrouter-free (\w+)", README, flags=re.MULTILINE))
    assert documented == {"init", "status", "test"}
    assert documented <= set(get_command(app).commands)


def test_every_config_key_in_the_readme_tables_is_understood_by_the_router():
    router_src = (Path(__file__).parent.parent / "src" / "llmrouter_free" / "router.py").read_text()
    for key in ["max_attempts_per_call", "max_attempts_per_deployment", "max_cloud_attempts",
                "request_timeout_seconds", "rate_limit_seconds", "daily_quota_seconds", "error_seconds"]:
        assert f"`{key}`" in README or f"`cooldown.{key}`" in README, f"README does not document {key}"
        assert key in router_src, f"router does not read {key}"

import pytest
import yaml

from llmrouter_free.config import TEMPLATE_PATH, build_router, load_routes, scale_request_timeout
from llmrouter_free.context_budget import TaskBudget


def test_bundled_template_is_valid_and_every_route_references_a_defined_deployment():
    config = load_routes(TEMPLATE_PATH)
    names = {d["name"] for d in config["deployments"]}
    assert config["routes"]
    assert {n for chain in config["routes"].values() for n in chain} <= names


def test_load_routes_reads_yaml(tmp_path):
    path = tmp_path / "r.yaml"
    path.write_text(yaml.safe_dump({"deployments": [], "routes": {}}))
    assert load_routes(path) == {"deployments": [], "routes": {}}


def test_scale_request_timeout_multiplies_and_does_not_mutate():
    config = {"request_timeout_seconds": 100}
    assert scale_request_timeout(config, 5)["request_timeout_seconds"] == 500
    assert config["request_timeout_seconds"] == 100
    assert scale_request_timeout({}, 2)["request_timeout_seconds"] == 300  # falls back to the 150 s default


def test_build_router_without_tasks_uses_the_config_as_written():
    config = {"deployments": [{"name": "a", "model": "p/a", "api_key_env": "K"}], "routes": {"r": ["a"]},
              "request_timeout_seconds": 60}
    router = build_router(config)
    assert router.request_timeout_seconds == 60


def test_build_router_with_tasks_sizes_num_ctx_and_scales_the_timeout():
    config = {
        "deployments": [{"name": "loc", "model": "ollama_chat/g", "api_key_env": "K"}],
        "routes": {"r": ["loc"]}, "request_timeout_seconds": 100,
    }
    tasks = {"t": TaskBudget("sys", "tmpl", output_tokens=100, batchable=True)}
    router = build_router(config, tasks=tasks, chunk_max_tokens=500, count_tokens=lambda s: len(s.split()),
                          batch_size=4)
    assert router.request_timeout_seconds == 400
    assert router.deployments["loc"].extra_body["options"]["num_ctx"] >= 4 * (500 + 100)


def test_build_router_with_tasks_but_no_counter_is_an_error():
    with pytest.raises(ValueError, match="together"):
        build_router({"deployments": [], "routes": {}}, tasks={"t": TaskBudget("s", "t", 1)})

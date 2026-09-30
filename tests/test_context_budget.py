from llmrouter_free.context_budget import (
    TaskBudget,
    apply_global_num_ctx,
    compute_num_ctx,
    fits_within,
    global_num_ctx,
    measure_template_overhead,
    recommend_num_ctx,
    round_up_to_bucket,
    worst_case_batch_size,
)


def word_counter(text: str) -> int:
    return len(text.split())


def test_round_up_to_bucket():
    assert round_up_to_bucket(1, 1024) == 1024
    assert round_up_to_bucket(1024, 1024) == 1024
    assert round_up_to_bucket(1025, 1024) == 2048


def test_measure_template_overhead_sums_system_and_template():
    assert measure_template_overhead("one two", "three four five", word_counter) == 5


def test_compute_num_ctx_applies_buffer_and_rounds_up():
    # worst_case = 900 + 380 + 3000 = 4280; *1.2 = 5136; rounded up to 1024 -> 6144
    assert compute_num_ctx(chunk_max_tokens=900, template_overhead_tokens=380, output_tokens=3000) == 6144


def test_compute_num_ctx_respects_custom_buffer_and_bucket():
    assert compute_num_ctx(chunk_max_tokens=50, template_overhead_tokens=30, output_tokens=20,
                           buffer=0.5, bucket=100) == 200


def test_global_num_ctx_uses_the_largest_task():
    result = global_num_ctx(chunk_max_tokens=900, template_overhead_tokens=380,
                            output_tokens_by_task={"small": 100, "big": 3000})
    assert result == compute_num_ctx(chunk_max_tokens=900, template_overhead_tokens=380, output_tokens=3000)


def test_fits_within():
    assert fits_within(8192, chunk_max_tokens=900, template_overhead_tokens=380, output_tokens=3000)
    assert not fits_within(2048, chunk_max_tokens=900, template_overhead_tokens=380, output_tokens=3000)


TASKS = {
    "extract": TaskBudget(system_prompt="sys prompt", rendered_template="user template here", output_tokens=300,
                          batchable=True),
    "single": TaskBudget(system_prompt="sys", rendered_template="tmpl", output_tokens=300, batchable=False),
}


def test_worst_case_batch_size_only_multiplies_batchable_tasks():
    assert worst_case_batch_size(5, TASKS) == {"extract": 5, "single": 1}
    assert worst_case_batch_size(1, TASKS) == {"extract": 1, "single": 1}


def test_recommend_num_ctx_report_shape_and_consistency():
    report = recommend_num_ctx(tasks=TASKS, chunk_max_tokens=900, count_tokens=word_counter)
    assert report["chunk_max_tokens"] == 900 and report["batch_size"] == 1
    assert set(report["overheads"]) == set(report["per_task_num_ctx"]) == set(TASKS)
    assert report["overheads"]["extract"] == 5  # "sys prompt" (2) + "user template here" (3)
    assert report["global_num_ctx"] == max(report["per_task_num_ctx"].values())


def test_recommend_num_ctx_scales_only_batchable_tasks_with_batch_size():
    solo = recommend_num_ctx(tasks=TASKS, chunk_max_tokens=900, count_tokens=word_counter, batch_size=1)
    batched = recommend_num_ctx(tasks=TASKS, chunk_max_tokens=900, count_tokens=word_counter, batch_size=5)
    assert batched["per_task_num_ctx"]["extract"] > solo["per_task_num_ctx"]["extract"]
    assert batched["per_task_num_ctx"]["single"] == solo["per_task_num_ctx"]["single"]


def test_apply_global_num_ctx_only_touches_ollama_deployments_and_preserves_other_extra_body():
    config = {
        "deployments": [
            {"name": "cloud", "model": "nvidia_nim/deepseek", "extra_body": {"chat_template_kwargs": {"thinking": True}}},
            {"name": "local", "model": "ollama_chat/gemma4:e4b", "extra_body": {"think": False}},
            {"name": "local-bare", "model": "ollama_chat/gemma3:4b"},
        ],
        "routes": {},
    }
    patched = apply_global_num_ctx(config, 6144)
    by_name = {d["name"]: d for d in patched["deployments"]}
    assert by_name["cloud"]["extra_body"] == {"chat_template_kwargs": {"thinking": True}}
    assert by_name["local"]["extra_body"] == {"think": False, "options": {"num_ctx": 6144}}
    assert by_name["local-bare"]["extra_body"] == {"options": {"num_ctx": 6144}}
    assert "options" not in config["deployments"][1]["extra_body"]  # original not mutated

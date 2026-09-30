"""Structured-output helpers: JSON validation for `LLMRouter.complete(..., validate=...)` and the
schema-constrained `response_format` value.

Nothing here is domain specific: pass any Pydantic model. A host application keeps its own response models
(facts, Q&A items, ...) and uses these two helpers with them.
"""

import json
import re
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)
_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def json_schema_response_format(model_cls: type[BaseModel], *, strict: bool = True) -> dict:
    """The OpenAI-style `response_format` value for schema-constrained decoding (as opposed to the generic
    `{"type": "json_object"}` mode). Confirmed via NVIDIA's own structured-output docs and a real live test
    against `nvidia/nemotron-3-super-120b-a12b:free` on 2026-09-15: with plain `json_object` mode this model
    regularly violated our schema (wrong enum values, missing required fields, wrong array length); with
    `json_schema` mode passing this exact schema, 5/5 real chunks came back fully schema-valid with zero
    violations. Not yet verified against every other deployment - callers should keep `json_mode=True` as
    the safe default and opt into this per route/call site once tested against that route's actual chain.

    `strict=False` (needed for Groq): Groq's `strict: true` mode requires `additionalProperties: false` and
    every property listed in `required` at *every* nesting level, and rejects a free-form `dict[str, Any]`
    field outright (confirmed live 2026-09-16/17 against a batched extraction schema and a judge schema) - the same limitation
    OpenAI's own strict structured outputs have. With `strict=False` the schema is still sent as a hint
    (confirmed live to produce well-formed, schema-compliant JSON on gpt-oss-120b for both extraction and
    judge schemas); our own pydantic validation (`json_validator`) is what actually enforces correctness.
    """
    return {
        "type": "json_schema",
        "json_schema": {"name": model_cls.__name__, "schema": model_cls.model_json_schema(), "strict": strict},
    }


def json_validator(model_cls: type[ModelT]) -> Callable[[str], ModelT]:
    """A `router.complete(..., validate=...)` callback: parses and validates `text` against `model_cls`.

    Tolerates a model wrapping its JSON in a ```json ... ``` fence despite `json_mode=True` (reasoning
    models sometimes do this anyway). Raises ValueError on any failure, which the router treats as
    `invalid_output` and retries on the next deployment.

    Also rejects a non-empty JSON object whose keys share nothing with `model_cls`'s own field names
    (e.g. a garbled/truncated `{"": "results"}` for a schema whose only field is `results`) - pydantic
    alone accepts this silently whenever every field has a default (`Field(default_factory=list)` etc.),
    which let a real truncated response from `openrouter-qwen3.8-27b` get cached as a valid, permanently
    empty "ok" result for one batch's prompt hash (confirmed live 2026-09-22: the router's cache then
    replayed that same garbage forever, since a cached "ok" is never re-validated against the live
    provider). A genuinely empty object (`{}`, e.g. a model correctly reporting "nothing found") is still
    accepted - only an object with content that matches none of the expected fields is rejected.
    """

    def validate(text: str) -> ModelT:
        candidate = text.strip()
        if fenced := _JSON_FENCE.search(candidate):
            candidate = fenced.group(1)
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError(f"not valid JSON: {exc}") from exc
        if isinstance(data, dict) and data and not (set(data.keys()) & set(model_cls.model_fields.keys())):
            raise ValueError(
                f"JSON object shares no fields with {model_cls.__name__} (got keys {sorted(data.keys())}) - "
                "likely a garbled or truncated response"
            )
        try:
            return model_cls.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"JSON did not match {model_cls.__name__}: {exc}") from exc

    return validate

import pytest
from pydantic import BaseModel, Field, field_validator

from llmrouter_free.validation import json_schema_response_format, json_validator


class Item(BaseModel):
    name: str
    tags: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v


class Batch(BaseModel):
    """Every field has a default - the shape that lets pydantic silently accept garbage."""

    results: list[Item] = Field(default_factory=list)


class TestJsonValidator:
    def test_parses_plain_json(self):
        result = json_validator(Batch)('{"results": [{"name": "a", "tags": ["x"]}]}')
        assert isinstance(result, Batch) and result.results[0].tags == ["x"]

    def test_strips_a_markdown_fence(self):
        text = 'Sure, here you go:\n```json\n{"results": []}\n```'
        assert isinstance(json_validator(Batch)(text), Batch)

    def test_invalid_json_raises_value_error(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            json_validator(Batch)("this is not json at all")

    def test_schema_mismatch_names_the_model(self):
        with pytest.raises(ValueError, match="Batch"):
            json_validator(Batch)('{"results": [{"tags": []}]}')  # missing required `name`

    def test_a_garbled_object_sharing_no_fields_with_the_schema_is_rejected(self):
        """Regression: a provider returned the truncated/garbled `{"": "results"}`. Pydantic alone accepts it
        (every field has a default) and the router's cache then replayed that empty "ok" forever."""
        with pytest.raises(ValueError, match="shares no fields"):
            json_validator(Batch)('{"": "results"}')

    def test_a_genuinely_empty_object_still_validates(self):
        assert json_validator(Batch)("{}").results == []

    def test_a_non_object_top_level_is_left_to_pydantic(self):
        with pytest.raises(ValueError, match="Batch"):
            json_validator(Batch)("[1, 2, 3]")


def test_json_schema_response_format_shape():
    rf = json_schema_response_format(Batch)
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "Batch"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"] == Batch.model_json_schema()


def test_json_schema_response_format_strict_false_for_providers_that_reject_strict():
    assert json_schema_response_format(Batch, strict=False)["json_schema"]["strict"] is False

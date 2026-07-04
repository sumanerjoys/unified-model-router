"""Unit tests for adapters — pure schema translation, zero mocking."""

from __future__ import annotations

from app.adapters.mock import MockAdapter
from app.adapters.openai import OpenAIAdapter
from app.models.unified import ChatCompletionRequest

UNIFIED = ChatCompletionRequest(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hello!"},
    ],
    stream=True,
    temperature=0.5,
    max_tokens=64,
).model_dump()


class TestOpenAIAdapter:
    def test_translate_request_is_passthrough_shape(self) -> None:
        body = OpenAIAdapter().translate_request(UNIFIED)
        assert body["model"] == "gpt-4o-mini"
        # System stays inside messages for OpenAI.
        assert body["messages"][0] == {"role": "system", "content": "Be brief."}
        assert body["stream"] is True
        assert body["temperature"] == 0.5
        assert body["max_tokens"] == 64

    def test_parse_chunk_extracts_delta(self) -> None:
        chunk = OpenAIAdapter().parse_chunk(
            '{"id":"x","choices":[{"index":0,"delta":{"content":"Hi"},'
            '"finish_reason":null}]}',
            "gpt-4o-mini",
        )
        assert chunk is not None
        assert chunk.choices[0].delta.content == "Hi"
        assert chunk.model == "gpt-4o-mini"

    def test_parse_chunk_done_returns_none(self) -> None:
        assert OpenAIAdapter().parse_chunk("[DONE]", "m") is None

    def test_parse_chunk_invalid_json_returns_none(self) -> None:
        assert OpenAIAdapter().parse_chunk("not json", "m") is None

    def test_parse_reasoning_content(self) -> None:
        # Reasoning models (e.g. gpt-oss, o-series) stream a separate
        # reasoning_content field with content == null. It must be preserved.
        chunk = OpenAIAdapter().parse_chunk(
            '{"choices":[{"index":0,"delta":'
            '{"content":null,"reasoning_content":"thinking..."}}]}',
            "gpt-oss",
        )
        assert chunk is not None
        assert chunk.choices[0].delta.content is None
        assert chunk.choices[0].delta.reasoning_content == "thinking..."

    def test_usage_only_chunk_with_empty_choices_returns_none(self) -> None:
        # Some providers emit a trailing usage-only chunk with choices == [].
        chunk = OpenAIAdapter().parse_chunk(
            '{"choices":[],"usage":{"total_tokens":161}}', "m"
        )
        assert chunk is None


class TestMockAdapter:
    def test_translate_request_splits_system_and_renames_fields(self) -> None:
        body = MockAdapter().translate_request(UNIFIED)
        # System prompt is pulled OUT to a top-level field (real translation).
        assert body["system"] == "Be brief."
        # Messages become turns with sender/text, excluding the system message.
        assert body["turns"] == [{"sender": "user", "text": "Hello!"}]
        # max_tokens is renamed.
        assert body["max_output_tokens"] == 64
        assert "messages" not in body

    def test_parse_token_event(self) -> None:
        chunk = MockAdapter().parse_chunk('{"type":"token","text":"lo"}', "m")
        assert chunk is not None
        assert chunk.choices[0].delta.content == "lo"

    def test_parse_end_event_sets_finish_reason(self) -> None:
        chunk = MockAdapter().parse_chunk('{"type":"end","stop":"stop"}', "m")
        assert chunk is not None
        assert chunk.choices[0].finish_reason == "stop"

    def test_unknown_event_returns_none(self) -> None:
        assert MockAdapter().parse_chunk('{"type":"noise"}', "m") is None

    def test_custom_generation_path(self) -> None:
        assert MockAdapter().chat_completions_path() == "/generate"


def test_both_adapters_produce_identical_unified_shape() -> None:
    """A token from either vendor yields the same unified chunk structure."""
    oa = OpenAIAdapter().parse_chunk(
        '{"choices":[{"delta":{"content":"X"}}]}', "m"
    )
    mk = MockAdapter().parse_chunk('{"type":"token","text":"X"}', "m")
    assert oa is not None and mk is not None
    assert oa.object == mk.object == "chat.completion.chunk"
    assert oa.choices[0].delta.content == mk.choices[0].delta.content == "X"

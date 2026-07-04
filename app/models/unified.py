"""Unified request/response schema (OpenAI chat/completions shape).

This is the single "language" every client speaks. Adapters translate between
this shape and each vendor's native payload. Modeling on the OpenAI shape gives
maximum client compatibility.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    """A single chat message in the unified schema."""

    role: Role
    content: str


class ChatCompletionRequest(BaseModel):
    """Unified inbound request for POST /v1/chat/completions."""

    model: str
    messages: list[Message] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    metadata: dict[str, Any] | None = None


# --- Streaming (SSE) chunk models: what the client always receives ---


class Delta(BaseModel):
    """Incremental content for a streaming chunk."""

    role: Role | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    """A single choice within a streaming chunk."""

    index: int = 0
    delta: Delta = Field(default_factory=Delta)
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """A unified SSE chunk (object == 'chat.completion.chunk')."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[StreamChoice] = Field(default_factory=list)

    def to_sse(self) -> str:
        """Serialize this chunk as an SSE `data:` event."""
        return f"data: {self.model_dump_json()}\n\n"


# --- Non-streaming response models ---


class ResponseMessage(BaseModel):
    """Assistant message in a non-streaming response."""

    role: Role = "assistant"
    content: str


class Choice(BaseModel):
    """A single choice in a non-streaming response."""

    index: int = 0
    message: ResponseMessage
    finish_reason: str | None = "stop"


class ChatCompletionResponse(BaseModel):
    """Unified non-streaming response body."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[Choice] = Field(default_factory=list)


SSE_DONE = "data: [DONE]\n\n"

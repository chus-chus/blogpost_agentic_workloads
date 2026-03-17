"""Core parsing library for OpenClaw session JSONL files."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None


def parse_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of parsed entries."""
    entries: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_sessions_meta(sessions_json_path: str | Path) -> dict:
    """Load sessions.json and return the raw dict keyed by session key."""
    with open(sessions_json_path) as f:
        return json.load(f)


def resolve_session_files(sessions_json_path: str | Path) -> dict[str, dict]:
    """Build a lookup from session ID to metadata including file path.

    Returns dict keyed by session ID with:
      session_key, session_id, file, spawned_by, model, provider, label
    """
    path = Path(sessions_json_path)
    meta = load_sessions_meta(path)
    result: dict[str, dict] = {}

    for session_key, info in meta.items():
        sid = info.get("sessionId", "")
        session_file = info.get("sessionFile", "")
        # Resolve relative to sessions.json parent if needed
        if session_file and not Path(session_file).is_absolute():
            session_file = str(path.parent / session_file)
        result[sid] = {
            "session_key": session_key,
            "session_id": sid,
            "file": session_file,
            "spawned_by": info.get("spawnedBy"),
            "model": info.get("model"),
            "provider": info.get("modelProvider"),
            "label": info.get("label"),
            "spawn_depth": info.get("spawnDepth", 0),
        }

    return result


def find_session_file(sessions_dir: str | Path, sid: str) -> str | None:
    """Find the JSONL file for a specific session ID.

    Looks for both active (.jsonl) and deleted (.jsonl.deleted.*) files.
    Returns the file path as a string, or None if not found.
    """
    d = Path(sessions_dir)
    # Active file
    active = d / f"{sid}.jsonl"
    if active.exists():
        return str(active)
    # Deleted file (*.jsonl.deleted.{timestamp})
    for candidate in d.glob(f"{sid}.jsonl.deleted.*"):
        return str(candidate)
    return None


def _stringify_for_tokens(value: object) -> str:
    """Serialize structured content into a stable text representation."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _flatten_content_block(block: dict, *, include_thinking: bool) -> str:
    """Convert one content block into text for token estimation."""
    btype = block.get("type", "")
    if btype == "text":
        return block.get("text", "")
    if btype == "toolCall":
        parts = [block.get("name", ""), _stringify_for_tokens(block.get("arguments"))]
        return "\n".join(part for part in parts if part)
    if btype == "thinking":
        if not include_thinking:
            return ""
        # The encrypted signature is bookkeeping, not readable prompt content.
        return block.get("thinking", "")

    parts: list[str] = []
    for key, value in block.items():
        if key in {"type", "thinkingSignature", "partialJson"}:
            continue
        text = _stringify_for_tokens(value)
        if text:
            parts.append(text)
    return "\n".join(parts)


def get_message_content_text(message: dict, *, include_thinking: bool = False) -> str:
    """Flatten a message into text for approximate token estimation."""
    content = message.get("content", [])
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        text = _flatten_content_block(block, include_thinking=include_thinking)
        if text:
            parts.append(text)
    return "\n".join(parts)


def get_message_content_chars(message: dict) -> int:
    """Count visible characters in a message's content (text + toolCall args).

    Skips thinking blocks (encrypted by provider).
    """
    return len(get_message_content_text(message))


@lru_cache(maxsize=None)
def _get_encoder(model: str | None):
    """Load a tokenizer for the current model, or fall back to a modern base encoding."""
    if tiktoken is None:
        return None
    if model:
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            pass
    for name in ("o200k_base", "cl100k_base"):
        try:
            return tiktoken.get_encoding(name)
        except ValueError:
            continue
    return None


def estimate_message_tokens(
    message: dict,
    *,
    model: str | None = None,
    include_thinking: bool = False,
) -> int:
    """Estimate how many prompt tokens a message contributes."""
    text = get_message_content_text(message, include_thinking=include_thinking)
    if not text:
        return 0

    encoder = _get_encoder(model)
    if encoder is not None:
        return len(encoder.encode(text, disallowed_special=()))

    # Conservative fallback when no tokenizer is installed.
    return max(1, math.ceil(len(text) / 4))


def classify_message(entry: dict) -> str | None:
    """Classify a message entry by its content source.

    Returns one of: 'tool_result', 'user_input', 'injected', 'assistant', or None.
    """
    if entry.get("type") != "message":
        return None
    msg = entry.get("message", {})
    role = msg.get("role", "")
    if role == "toolResult":
        return "tool_result"
    if role == "assistant":
        return "assistant"
    if role == "user":
        # Check if this is an injected/synthetic message (subagent completion)
        content = msg.get("content", [])
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = " ".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        if "[Internal task completion event]" in text:
            return "injected"
        return "user_input"
    return None

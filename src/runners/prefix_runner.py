"""Prefix-forking model runner. Not used in Week 1.

Week 1 does not call any paid or hosted model API. The smoke test uses
`agent.mock_agent.MockAgent`.

TODO(decision): model ids in the lean plan (Grok 4.7, Composer 2.5, Haiku/Luna,
Qwen3.5-9B) are not configured in Week 1.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrefixCheckpoint:
    task_id: str
    model_id: str
    messages_before_post: list[dict]
    pending_post_call: dict


def run_from_prefix(*args: object, **kwargs: object) -> dict:
    raise NotImplementedError(
        "prefix_runner is unwired in Week 1. No model API client is installed."
    )

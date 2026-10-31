from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrefixCheckpoint:
    task_id: str
    model_id: str
    messages_before_post: list[dict]
    pending_post_call: dict


# TODO(decision): no hosted model is configured; smoke uses the mock agent.
def run_from_prefix(*args: object, **kwargs: object) -> dict:
    raise NotImplementedError("prefix_runner is not yet implemented.")

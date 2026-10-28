"""Shared records for the Week 1 harness and mock agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ActionKind = Literal["GET", "POST", "PUT", "FINISH"]


@dataclass(frozen=True)
class TaskCase:
    id: str
    instruction: str
    context: str
    sol: list[Any] | None
    eval_mrn: str | None

    @property
    def family(self) -> str:
        return self.id.split("_", 1)[0]


@dataclass(frozen=True)
class AgentAction:
    kind: ActionKind
    url: str | None = None
    body: dict[str, Any] | None = None
    finish_payload: str | None = None


@dataclass
class ToolObservation:
    """What the harness hands back after one FHIR call."""

    status: int
    body: Any
    headers: dict[str, str]
    location: str | None = None


@dataclass
class CapturedCall:
    method: str
    url: str
    request_body: dict[str, Any] | None
    status: int
    response_body: Any
    response_headers: dict[str, str]
    location: str | None
    elapsed_ms: float


@dataclass
class TrialRecord:
    task_id: str
    passed: bool
    failure_reason: str | None
    calls: list[CapturedCall] = field(default_factory=list)
    finish_text: str | None = None
    rounds: int = 0

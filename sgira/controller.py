"""Gemini high-level routing controller, confidence fallback, and memory."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .environment import InventoryObservation
from .executors import Route, observation_payload
from .features import StateFeatures
from .llm import StructuredLLM
from .prompts import CONTROLLER_SYSTEM
from .rule_router import RuleRouter
from .schemas import ControllerOutput


@dataclass(frozen=True)
class MemoryItem:
    text: str
    generated_period: int
    expires_after_period: int


class ShortTermMemory:
    """At most three compact observations with explicit expiry."""

    def __init__(self, *, max_items: int = 3, ttl_periods: int = 3) -> None:
        if max_items < 1 or ttl_periods < 1:
            raise ValueError("memory limits must be positive")
        self.max_items = max_items
        self.ttl_periods = ttl_periods
        self._items: list[MemoryItem] = []

    def active(self, period: int) -> tuple[MemoryItem, ...]:
        self._items = [
            item for item in self._items if item.expires_after_period >= period
        ]
        return tuple(self._items)

    def update(self, text: str, period: int) -> None:
        text = " ".join(text.strip().split())[:200]
        if not text:
            return
        self._items = [item for item in self._items if item.text != text]
        self._items.append(MemoryItem(text, period, period + self.ttl_periods))
        self._items = self._items[-self.max_items :]


@dataclass(frozen=True)
class ControllerConfig:
    confidence_threshold: float = 0.5
    include_context: bool = True
    include_memory: bool = True
    include_supply_diagnosis: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")


@dataclass(frozen=True)
class RoutingRecord:
    selected_route: Route
    source: str
    llm_calls: int
    fallback_reason: str
    reason_codes: tuple[str, ...]
    raw_output: Any
    parsed_output: Mapping[str, Any]
    llm_request: Mapping[str, Any]
    active_memory: tuple[MemoryItem, ...]


class GeminiController:
    def __init__(
        self,
        llm: StructuredLLM,
        *,
        rule_router: RuleRouter | None = None,
        memory: ShortTermMemory | None = None,
        config: ControllerConfig | None = None,
    ) -> None:
        self.llm = llm
        self.rule_router = rule_router or RuleRouter()
        self.memory = memory or ShortTermMemory()
        self.config = config or ControllerConfig()

    def select(
        self,
        observation: InventoryObservation,
        features: StateFeatures,
    ) -> RoutingRecord:
        active_memory = self.memory.active(observation.period)
        state = observation_payload(observation)
        if not self.config.include_context:
            state["context"] = {}
        if not self.config.include_supply_diagnosis:
            state["outstanding_orders"] = []
        feature_payload = features.to_dict()
        if not self.config.include_supply_diagnosis:
            for key in ("delay_score", "oldest_order_age", "overdue_order_count"):
                feature_payload.pop(key, None)
        payload = {
            "observable_state": state,
            "features": feature_payload,
            "memory": [asdict(item) for item in active_memory]
            if self.config.include_memory else [],
        }
        request = {
            "role": "controller",
            "system_prompt": CONTROLLER_SYSTEM,
            "payload": payload,
        }
        try:
            raw = self.llm.complete_json(
                role="controller",
                system_prompt=CONTROLLER_SYSTEM,
                payload=payload,
            )
            parsed = ControllerOutput.parse(raw)
        except Exception as exc:
            return RoutingRecord(
                selected_route=Route.OR,
                source="or_fallback",
                llm_calls=1,
                fallback_reason=str(exc),
                reason_codes=("CONTROLLER_FAILURE",),
                raw_output=locals().get("raw"),
                parsed_output={},
                llm_request=request,
                active_memory=active_memory,
            )

        if parsed.confidence < self.config.confidence_threshold:
            rule = self.rule_router.select(features)
            return RoutingRecord(
                selected_route=rule.route,
                source="rule_fallback",
                llm_calls=1,
                fallback_reason=(
                    f"confidence {parsed.confidence:.3f} below "
                    f"{self.config.confidence_threshold:.3f}"
                ),
                reason_codes=rule.reason_codes,
                raw_output=raw,
                parsed_output=asdict(parsed),
                llm_request=request,
                active_memory=active_memory,
            )

        if self.config.include_memory:
            self.memory.update(parsed.memory_update, observation.period)
        return RoutingRecord(
            selected_route=Route(parsed.selected_route),
            source="gemini",
            llm_calls=1,
            fallback_reason="",
            reason_codes=parsed.reason_codes,
            raw_output=raw,
            parsed_output=asdict(parsed),
            llm_request=request,
            active_memory=active_memory,
        )

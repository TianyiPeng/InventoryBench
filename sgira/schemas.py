"""Strict structured-output parsing for SGIRA's LLM roles."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping


class SchemaError(ValueError):
    """Raised when an LLM response violates its frozen output contract."""


def _object(raw: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if not isinstance(raw, str):
        raise SchemaError("response must be a JSON object or JSON string")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise SchemaError("top-level JSON value must be an object")
    return value


def _number(value: Any, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{field} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise SchemaError(f"{field} must be finite and >= {minimum}")
    return value


def _integer(value: Any, field: str) -> int:
    number = _number(value, field)
    if not number.is_integer():
        raise SchemaError(f"{field} must be an integer")
    return int(number)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise SchemaError(f"{field} must be a string")
    return value.strip()


def _reason_codes(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SchemaError("reason_codes must be a list of strings")
    return tuple(item.strip() for item in value if item.strip())


@dataclass(frozen=True)
class LLMDirectOutput:
    order_quantity: int
    reason_codes: tuple[str, ...]
    short_reason: str

    @classmethod
    def parse(cls, raw: str | Mapping[str, Any]) -> "LLMDirectOutput":
        value = _object(raw)
        return cls(
            order_quantity=_integer(value.get("order_quantity"), "order_quantity"),
            reason_codes=_reason_codes(value.get("reason_codes")),
            short_reason=_text(value.get("short_reason"), "short_reason"),
        )


@dataclass(frozen=True)
class ORToLLMOutput:
    decision: str
    or_order_quantity: int
    final_order_quantity: int
    reason_codes: tuple[str, ...]

    @classmethod
    def parse(cls, raw: str | Mapping[str, Any]) -> "ORToLLMOutput":
        value = _object(raw)
        decision = _text(value.get("decision"), "decision")
        if decision not in {"accept", "increase", "decrease"}:
            raise SchemaError("decision must be accept, increase, or decrease")
        return cls(
            decision=decision,
            or_order_quantity=_integer(
                value.get("or_order_quantity"), "or_order_quantity"
            ),
            final_order_quantity=_integer(
                value.get("final_order_quantity"), "final_order_quantity"
            ),
            reason_codes=_reason_codes(value.get("reason_codes")),
        )


@dataclass(frozen=True)
class LLMToOROutput:
    effective_lead_time: float
    protection_horizon: float
    protection_demand_mean: float
    protection_demand_std: float
    reason_codes: tuple[str, ...]

    @classmethod
    def parse(cls, raw: str | Mapping[str, Any]) -> "LLMToOROutput":
        value = _object(raw)
        lead_time = _number(value.get("effective_lead_time"), "effective_lead_time")
        horizon = _number(value.get("protection_horizon"), "protection_horizon")
        if horizon < 1:
            raise SchemaError("protection_horizon must be >= 1")
        return cls(
            effective_lead_time=lead_time,
            protection_horizon=horizon,
            protection_demand_mean=_number(
                value.get("protection_demand_mean"), "protection_demand_mean"
            ),
            protection_demand_std=_number(
                value.get("protection_demand_std"), "protection_demand_std"
            ),
            reason_codes=_reason_codes(value.get("reason_codes")),
        )


@dataclass(frozen=True)
class ControllerOutput:
    demand_regime: str
    supply_regime: str
    selected_route: str
    confidence: float
    reason_codes: tuple[str, ...]
    short_reason: str
    memory_update: str

    @classmethod
    def parse(cls, raw: str | Mapping[str, Any]) -> "ControllerOutput":
        value = _object(raw)
        demand_regime = _text(value.get("demand_regime"), "demand_regime")
        if demand_regime not in {
            "stable", "trend_up", "trend_down", "shift", "volatile", "uncertain"
        }:
            raise SchemaError("invalid demand_regime")
        supply_regime = _text(value.get("supply_regime"), "supply_regime")
        if supply_regime not in {"normal", "delayed", "possible_loss", "uncertain"}:
            raise SchemaError("invalid supply_regime")
        selected_route = _text(value.get("selected_route"), "selected_route")
        if selected_route not in {"OR", "LLM", "OR_TO_LLM", "LLM_TO_OR"}:
            raise SchemaError("invalid selected_route")
        confidence = _number(value.get("confidence"), "confidence")
        if confidence > 1:
            raise SchemaError("confidence must be <= 1")
        return cls(
            demand_regime=demand_regime,
            supply_regime=supply_regime,
            selected_route=selected_route,
            confidence=confidence,
            reason_codes=_reason_codes(value.get("reason_codes")),
            short_reason=_text(value.get("short_reason"), "short_reason"),
            memory_update=_text(value.get("memory_update"), "memory_update"),
        )

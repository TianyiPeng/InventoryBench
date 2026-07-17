"""The four fixed decision chains shared by baselines and dynamic routing."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping

from scipy.stats import norm

from .environment import InventoryObservation
from .llm import StructuredLLM
from .or_policy import ORDecision, ORPolicy
from .prompts import LLM_DIRECT_SYSTEM, LLM_TO_OR_SYSTEM, OR_TO_LLM_SYSTEM
from .schemas import LLMDirectOutput, LLMToOROutput, ORToLLMOutput, SchemaError


class Route(str, Enum):
    OR = "OR"
    LLM = "LLM"
    OR_TO_LLM = "OR_TO_LLM"
    LLM_TO_OR = "LLM_TO_OR"


@dataclass(frozen=True)
class ExecutionRecord:
    requested_route: str
    applied_route: str
    order_quantity: int
    llm_calls: int
    fallback: bool
    fallback_reason: str
    corrections: tuple[str, ...]
    raw_output: Any
    parsed_output: Mapping[str, Any]
    or_decision: ORDecision
    llm_request: Mapping[str, Any] = field(default_factory=dict)


class _ExecutionFailure(ValueError):
    def __init__(
        self, reason: str, raw_output: Any, llm_request: Mapping[str, Any]
    ) -> None:
        super().__init__(reason)
        self.raw_output = raw_output
        self.llm_request = llm_request


def observation_payload(observation: InventoryObservation) -> dict[str, Any]:
    """Serialize only the public, pre-decision state."""
    return {
        "period": observation.period,
        "on_hand": observation.on_hand,
        "outstanding_orders": [asdict(row) for row in observation.outstanding_orders],
        "demand_history": list(observation.demand_history),
        "expected_lead_time": observation.expected_lead_time,
        "profit_per_unit": observation.profit_per_unit,
        "holding_cost_per_unit": observation.holding_cost_per_unit,
        "critical_fractile": observation.critical_fractile,
        "context": dict(observation.context),
    }


class FixedRouteExecutors:
    """Execute OR, LLM, OR->LLM, or LLM->OR with uniform safeguards."""

    def __init__(
        self,
        or_policy: ORPolicy,
        llm: StructuredLLM | None = None,
        *,
        or_to_llm_max_adjustment: float = 0.30,
    ) -> None:
        if not 0 <= or_to_llm_max_adjustment <= 1:
            raise ValueError("or_to_llm_max_adjustment must be between 0 and 1")
        self.or_policy = or_policy
        self.llm = llm
        self.max_adjustment = or_to_llm_max_adjustment

    def execute(
        self, route: Route | str, observation: InventoryObservation
    ) -> ExecutionRecord:
        requested = route.value if isinstance(route, Route) else str(route)
        try:
            route = Route(route)
        except ValueError:
            or_decision = self.or_policy.decide(observation)
            return self._fallback(
                requested, or_decision, "invalid route name", llm_calls=0
            )
        or_decision = self.or_policy.decide(observation)
        if route is Route.OR:
            return self._or_record(or_decision)
        if self.llm is None:
            return self._fallback(
                route.value, or_decision, "LLM client is not configured", llm_calls=0
            )
        try:
            if route is Route.LLM:
                return self._llm_direct(observation, or_decision)
            if route is Route.OR_TO_LLM:
                return self._or_to_llm(observation, or_decision)
            return self._llm_to_or(observation, or_decision)
        except Exception as exc:
            return self._fallback(
                route.value,
                or_decision,
                str(exc),
                llm_calls=1,
                raw_output=getattr(exc, "raw_output", None),
                llm_request=getattr(exc, "llm_request", {}),
            )

    def _or_record(self, decision: ORDecision) -> ExecutionRecord:
        return ExecutionRecord(
            requested_route=Route.OR.value,
            applied_route=Route.OR.value,
            order_quantity=decision.order_quantity,
            llm_calls=0,
            fallback=False,
            fallback_reason="",
            corrections=(),
            raw_output=None,
            parsed_output=asdict(decision),
            or_decision=decision,
        )

    def _fallback(
        self,
        requested_route: str,
        decision: ORDecision,
        reason: str,
        *,
        llm_calls: int,
        raw_output: Any = None,
        llm_request: Mapping[str, Any] | None = None,
    ) -> ExecutionRecord:
        return ExecutionRecord(
            requested_route=requested_route,
            applied_route=Route.OR.value,
            order_quantity=decision.order_quantity,
            llm_calls=llm_calls,
            fallback=True,
            fallback_reason=reason,
            corrections=(),
            raw_output=raw_output,
            parsed_output={},
            or_decision=decision,
            llm_request=llm_request or {},
        )

    def _complete(
        self, *, role: str, system_prompt: str, payload: Mapping[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        request = {
            "role": role,
            "system_prompt": system_prompt,
            "payload": dict(payload),
        }
        try:
            raw = self.llm.complete_json(
                role=role, system_prompt=system_prompt, payload=payload
            )
        except Exception as exc:
            raise _ExecutionFailure(str(exc), None, request) from exc
        return raw, request

    def _safe_order(
        self, value: int, or_decision: ORDecision
    ) -> tuple[int, tuple[str, ...]]:
        corrections: list[str] = []
        order = max(int(value), 0)
        cap = self.or_policy.config.order_cap
        if cap is not None and order > cap:
            order = cap
            corrections.append("ORDER_CAP_APPLIED")
        dynamic_cap = or_decision.computed_order_cap
        if dynamic_cap is not None and order > dynamic_cap:
            order = dynamic_cap
            corrections.append("DYNAMIC_ORDER_CAP_APPLIED")
        return order, tuple(corrections)

    def _llm_direct(
        self, observation: InventoryObservation, or_decision: ORDecision
    ) -> ExecutionRecord:
        raw, request = self._complete(
            role="llm_direct",
            system_prompt=LLM_DIRECT_SYSTEM,
            payload=observation_payload(observation),
        )
        try:
            parsed = LLMDirectOutput.parse(raw)
        except SchemaError as exc:
            raise _ExecutionFailure(str(exc), raw, request) from exc
        order, corrections = self._safe_order(parsed.order_quantity, or_decision)
        return ExecutionRecord(
            Route.LLM.value, Route.LLM.value, order, 1, False, "", corrections,
            raw, asdict(parsed), or_decision, request,
        )

    def _or_to_llm(
        self, observation: InventoryObservation, or_decision: ORDecision
    ) -> ExecutionRecord:
        payload = observation_payload(observation)
        payload["or_decision"] = asdict(or_decision)
        raw, request = self._complete(
            role="or_to_llm", system_prompt=OR_TO_LLM_SYSTEM, payload=payload
        )
        try:
            parsed = ORToLLMOutput.parse(raw)
            if parsed.or_order_quantity != or_decision.order_quantity:
                raise SchemaError("or_order_quantity does not match the supplied OR result")
            if parsed.decision == "accept" and parsed.final_order_quantity != parsed.or_order_quantity:
                raise SchemaError("accept decision must preserve the OR quantity")
            if parsed.decision == "increase" and parsed.final_order_quantity < parsed.or_order_quantity:
                raise SchemaError("increase decision cannot reduce the OR quantity")
            if parsed.decision == "decrease" and parsed.final_order_quantity > parsed.or_order_quantity:
                raise SchemaError("decrease decision cannot increase the OR quantity")
        except SchemaError as exc:
            raise _ExecutionFailure(str(exc), raw, request) from exc

        lower = math.floor(or_decision.order_quantity * (1 - self.max_adjustment))
        upper = math.ceil(or_decision.order_quantity * (1 + self.max_adjustment))
        bounded = min(max(parsed.final_order_quantity, lower), upper)
        corrections: list[str] = []
        if bounded != parsed.final_order_quantity:
            corrections.append("OR_ADJUSTMENT_BOUND_APPLIED")
        order, cap_corrections = self._safe_order(bounded, or_decision)
        corrections.extend(cap_corrections)
        return ExecutionRecord(
            Route.OR_TO_LLM.value, Route.OR_TO_LLM.value, order, 1, False, "",
            tuple(corrections), raw, asdict(parsed), or_decision, request,
        )

    def _llm_to_or(
        self, observation: InventoryObservation, or_decision: ORDecision
    ) -> ExecutionRecord:
        raw, request = self._complete(
            role="llm_to_or",
            system_prompt=LLM_TO_OR_SYSTEM,
            payload=observation_payload(observation),
        )
        try:
            parsed = LLMToOROutput.parse(raw)
        except SchemaError as exc:
            raise _ExecutionFailure(str(exc), raw, request) from exc
        z_value = float(norm.ppf(observation.critical_fractile))
        target = parsed.protection_demand_mean + z_value * parsed.protection_demand_std
        raw_order = max(int(math.ceil(target - observation.inventory_position)), 0)
        order, corrections = self._safe_order(raw_order, or_decision)
        parsed_log = asdict(parsed)
        parsed_log["target_inventory"] = target
        parsed_log["uncapped_order_quantity"] = raw_order
        return ExecutionRecord(
            Route.LLM_TO_OR.value, Route.LLM_TO_OR.value, order, 1, False, "",
            corrections, raw, parsed_log, or_decision, request,
        )

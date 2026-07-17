"""Human-rule and SGIRA routers sharing the fixed-route executors."""

from __future__ import annotations

from dataclasses import dataclass

from .controller import GeminiController, RoutingRecord
from .environment import InventoryObservation
from .executors import ExecutionRecord, FixedRouteExecutors
from .features import FeatureConfig, StateFeatures, build_state_features
from .rule_router import RuleRouter


@dataclass(frozen=True)
class DynamicDecision:
    features: StateFeatures
    routing: RoutingRecord
    execution: ExecutionRecord
    total_llm_calls: int


class SGIRAPolicy:
    def __init__(
        self,
        controller: GeminiController,
        executors: FixedRouteExecutors,
        *,
        feature_config: FeatureConfig | None = None,
    ) -> None:
        self.controller = controller
        self.executors = executors
        self.feature_config = feature_config

    def decide(self, observation: InventoryObservation) -> DynamicDecision:
        features = build_state_features(observation, self.feature_config)
        routing = self.controller.select(observation, features)
        execution = self.executors.execute(routing.selected_route, observation)
        return DynamicDecision(
            features=features,
            routing=routing,
            execution=execution,
            total_llm_calls=routing.llm_calls + execution.llm_calls,
        )


class RuleRoutingPolicy:
    def __init__(
        self,
        executors: FixedRouteExecutors,
        *,
        router: RuleRouter | None = None,
        feature_config: FeatureConfig | None = None,
    ) -> None:
        self.executors = executors
        self.router = router or RuleRouter()
        self.feature_config = feature_config

    def decide(self, observation: InventoryObservation) -> DynamicDecision:
        features = build_state_features(observation, self.feature_config)
        selected = self.router.select(features)
        routing = RoutingRecord(
            selected_route=selected.route,
            source="rule",
            llm_calls=0,
            fallback_reason="",
            reason_codes=selected.reason_codes,
            raw_output=None,
            parsed_output={},
            llm_request={},
            active_memory=(),
        )
        execution = self.executors.execute(selected.route, observation)
        return DynamicDecision(
            features=features,
            routing=routing,
            execution=execution,
            total_llm_calls=execution.llm_calls,
        )

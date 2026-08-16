"""Bayesian warning and entry-deterrence model inspired by the Korean War.

Nature draws China's intervention resolve before any warning. China observes
its type and sends ``quiet`` or ``warn``; the US/UN receiver observes that
message and chooses its advance; China then chooses whether to intervene. The
last move is solved exactly at every continuation history before the reduced
signaling game is passed to the exact pure-PBE enumerator.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isclose
from time import perf_counter

from cold_war_sim.solvers.pure_pbe import (
    SignalingEquilibrium,
    SignalingGame,
    enumerate_pure_pbe,
)

from .parameters import KoreaParameters

TYPES: tuple[str, ...] = ("low_resolve", "high_resolve")
WARNINGS: tuple[str, ...] = ("quiet", "warn")
ADVANCE_ACTIONS: tuple[str, ...] = ("restraint", "limited", "aggressive")
ENTRY_ACTIONS: tuple[str, ...] = ("stay_out", "intervene")
ACTION_RANK: dict[str, int] = {"restraint": 0, "limited": 1, "aggressive": 2}


@dataclass(frozen=True, slots=True)
class GameStage:
    index: int
    actor: str
    choice: str
    observes: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "actor": self.actor,
            "choice": self.choice,
            "observes": list(self.observes),
        }


@dataclass(frozen=True, slots=True)
class TerminalHistory:
    china_type: str
    warning: str
    receiver_action: str
    entry_action: str

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (self.china_type, self.warning, self.receiver_action, self.entry_action)


@dataclass(frozen=True, slots=True)
class EntryDecision:
    china_type: str
    warning: str
    receiver_action: str
    selected_entry: str
    china_payoff: float
    receiver_payoff: float
    stay_out_china_payoff: float
    intervene_china_payoff: float
    tied: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "china_type": self.china_type,
            "warning": self.warning,
            "receiver_action": self.receiver_action,
            "selected_entry": self.selected_entry,
            "china_payoff": self.china_payoff,
            "receiver_payoff": self.receiver_payoff,
            "stay_out_china_payoff": self.stay_out_china_payoff,
            "intervene_china_payoff": self.intervene_china_payoff,
            "tied": self.tied,
        }


@dataclass(frozen=True, slots=True)
class KoreaEquilibrium:
    classification: str
    sender_strategy: dict[str, str]
    receiver_strategy: dict[str, str]
    posteriors: dict[str, dict[str, float]]
    reach_probabilities: dict[str, float]
    expected_china_utility: float
    expected_receiver_utility: float
    warning_probability: float
    intervention_probability: float
    receiver_action_frequencies: dict[str, float]
    credibility_regime: str
    depends_on_off_path_beliefs: bool
    off_path_messages: tuple[str, ...]
    sender_deviation_gains: dict[str, float]
    receiver_deviation_gains: dict[str, float]
    best_response_gap: float
    realized_paths: tuple[dict[str, object], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "classification": self.classification,
            "sender_strategy": dict(self.sender_strategy),
            "receiver_strategy": dict(self.receiver_strategy),
            "posteriors": {
                message: dict(posterior) for message, posterior in self.posteriors.items()
            },
            "reach_probabilities": dict(self.reach_probabilities),
            "expected_utilities": {
                "china": self.expected_china_utility,
                "us_un": self.expected_receiver_utility,
            },
            "warning_probability": self.warning_probability,
            "intervention_probability": self.intervention_probability,
            "receiver_action_frequencies": dict(self.receiver_action_frequencies),
            "credibility_regime": self.credibility_regime,
            "depends_on_off_path_beliefs": self.depends_on_off_path_beliefs,
            "off_path_messages": list(self.off_path_messages),
            "sender_deviation_gains": dict(self.sender_deviation_gains),
            "receiver_deviation_gains": dict(self.receiver_deviation_gains),
            "best_response_gap": self.best_response_gap,
            "realized_paths": list(self.realized_paths),
        }


@dataclass(frozen=True, slots=True)
class KoreaSolution:
    status: str
    found: bool
    multiple: bool
    equilibria: tuple[KoreaEquilibrium, ...]
    continuation_entry_policy: tuple[EntryDecision, ...]
    off_path_belief_convention: dict[str, dict[str, float]]
    runtime_seconds: float
    best_response_gap: float | None
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        pooling_count = sum(eq.classification == "pooling" for eq in self.equilibria)
        separating_count = sum(eq.classification == "separating" for eq in self.equilibria)
        return {
            "solver": {
                "name": "exact_pure_pbe_with_solved_entry_continuations",
                "version": "1.0",
                "equilibrium_concept": "pure-strategy perfect Bayesian equilibrium",
                "status": self.status,
                "found": self.found,
                "multiple_solutions": self.multiple,
                "exactness": "exact_finite_enumeration",
                "convergence_status": "not_applicable_exact_enumeration",
                "best_response_gap": self.best_response_gap,
                "runtime_seconds": self.runtime_seconds,
                "seed": None,
                "off_path_belief_convention": {
                    message: dict(belief)
                    for message, belief in self.off_path_belief_convention.items()
                },
                "restricted_class": "pure strategies only",
                "warnings": list(self.warnings),
                "assumptions": [
                    "All numerical parameters are illustrative, normalized, or numerical conveniences.",
                    "The pure two-type model supports pooling or separating profiles, not partial pooling.",
                    "Intervention is a strategic continuation choice; no accident process is included.",
                ],
            },
            "metrics": {
                "equilibrium_count": len(self.equilibria),
                "pooling_equilibrium_count": pooling_count,
                "separating_equilibrium_count": separating_count,
                "partially_pooling_equilibrium_count": 0,
                "no_pure_pbe_found": not self.found,
            },
            "equilibria": [equilibrium.as_dict() for equilibrium in self.equilibria],
            "continuation_entry_policy": [
                decision.as_dict() for decision in self.continuation_entry_policy
            ],
            "warnings": list(self.warnings),
            "assumptions": [
                "All numerical parameters are illustrative, normalized, or numerical conveniences.",
                "The pure two-type model supports pooling or separating profiles, not partial pooling.",
                "Intervention is a strategic continuation choice; no accident process is included.",
            ],
        }


class KoreaWarningModel:
    """Finite warning game with exact continuation and pure-PBE solution."""

    def __init__(self, parameters: KoreaParameters | None = None) -> None:
        self.parameters = parameters or KoreaParameters()

    @staticmethod
    def move_order() -> tuple[GameStage, ...]:
        """Return the formal chronology, including information observations."""

        return (
            GameStage(0, "nature", "draw_china_type", ()),
            GameStage(1, "china", "quiet_or_warn", ("china_type",)),
            GameStage(2, "us_un", "choose_advance", ("warning",)),
            GameStage(
                3,
                "china",
                "stay_out_or_intervene",
                ("china_type", "warning", "receiver_action"),
            ),
        )

    @staticmethod
    def terminal_histories() -> tuple[TerminalHistory, ...]:
        """Enumerate the complete finite set of terminal histories."""

        return tuple(
            TerminalHistory(china_type, warning, action, entry)
            for china_type in TYPES
            for warning in WARNINGS
            for action in ADVANCE_ACTIONS
            for entry in ENTRY_ACTIONS
        )

    def _validate_history(
        self, china_type: str, warning: str, receiver_action: str, entry_action: str | None = None
    ) -> None:
        if china_type not in TYPES:
            raise ValueError(f"unknown China type: {china_type}")
        if warning not in WARNINGS:
            raise ValueError(f"unknown warning: {warning}")
        if receiver_action not in ADVANCE_ACTIONS:
            raise ValueError(f"unknown receiver action: {receiver_action}")
        if entry_action is not None and entry_action not in ENTRY_ACTIONS:
            raise ValueError(f"unknown entry action: {entry_action}")

    def _warning_cost(self, china_type: str, warning: str) -> float:
        if warning == "quiet":
            return 0.0
        if china_type == "low_resolve":
            return self.parameters.low_warning_cost
        return self.parameters.high_warning_cost

    def _threat(self, china_type: str, receiver_action: str) -> float:
        params = self.parameters
        sensitivity = (
            params.low_threat_sensitivity
            if china_type == "low_resolve"
            else params.high_threat_sensitivity
        )
        levels = {
            "restraint": params.restraint_threat,
            "limited": params.limited_threat,
            "aggressive": params.aggressive_threat,
        }
        return sensitivity * levels[receiver_action]

    def terminal_payoffs(
        self, china_type: str, warning: str, receiver_action: str, entry_action: str
    ) -> tuple[float, float]:
        """Return ``(China, US/UN)`` utility at a terminal history."""

        self._validate_history(china_type, warning, receiver_action, entry_action)
        params = self.parameters
        warning_cost = self._warning_cost(china_type, warning)
        threat = self._threat(china_type, receiver_action)
        benefits = {
            "restraint": params.restraint_receiver_benefit,
            "limited": params.limited_receiver_benefit,
            "aggressive": params.aggressive_receiver_benefit,
        }
        conflict_costs = {
            "restraint": params.restraint_conflict_cost,
            "limited": params.limited_conflict_cost,
            "aggressive": params.aggressive_conflict_cost,
        }
        intervention_costs = {
            "restraint": params.restraint_intervention_cost,
            "limited": params.limited_intervention_cost,
            "aggressive": params.aggressive_intervention_cost,
        }
        if entry_action == "stay_out":
            china = -threat - warning_cost
            receiver = benefits[receiver_action]
        else:
            fixed_cost = (
                params.low_intervention_fixed_cost
                if china_type == "low_resolve"
                else params.high_intervention_fixed_cost
            )
            china = (
                -(1.0 - params.intervention_effectiveness) * threat
                - fixed_cost
                - intervention_costs[receiver_action]
                - warning_cost
            )
            receiver = benefits[receiver_action] - conflict_costs[receiver_action]
        return (china, receiver)

    def continuation_entry(self, china_type: str, warning: str, receiver_action: str) -> EntryDecision:
        """Solve China's last move exactly at the specified history."""

        self._validate_history(china_type, warning, receiver_action)
        stay = self.terminal_payoffs(china_type, warning, receiver_action, "stay_out")
        intervene = self.terminal_payoffs(china_type, warning, receiver_action, "intervene")
        difference = intervene[0] - stay[0]
        tolerance = self.parameters.comparison_tolerance
        tied = isclose(difference, 0.0, rel_tol=0.0, abs_tol=tolerance)
        if difference > tolerance or (tied and self.parameters.entry_tie_break == "intervene"):
            selected = "intervene"
            payoff = intervene
        else:
            selected = "stay_out"
            payoff = stay
        return EntryDecision(
            china_type=china_type,
            warning=warning,
            receiver_action=receiver_action,
            selected_entry=selected,
            china_payoff=payoff[0],
            receiver_payoff=payoff[1],
            stay_out_china_payoff=stay[0],
            intervene_china_payoff=intervene[0],
            tied=tied,
        )

    def continuation_policy(self) -> tuple[EntryDecision, ...]:
        return tuple(
            self.continuation_entry(china_type, warning, receiver_action)
            for china_type in TYPES
            for warning in WARNINGS
            for receiver_action in ADVANCE_ACTIONS
        )

    def reduced_signaling_game(self) -> SignalingGame:
        """Build the signaling game using exact continuation values."""

        sender_payoffs: dict[tuple[str, str, str], float] = {}
        receiver_payoffs: dict[tuple[str, str, str], float] = {}
        for china_type in TYPES:
            for warning in WARNINGS:
                for action in ADVANCE_ACTIONS:
                    continuation = self.continuation_entry(china_type, warning, action)
                    key = (china_type, warning, action)
                    sender_payoffs[key] = continuation.china_payoff
                    receiver_payoffs[key] = continuation.receiver_payoff
        return SignalingGame(
            sender="china",
            receiver="us_un",
            types=TYPES,
            messages=WARNINGS,
            actions=ADVANCE_ACTIONS,
            prior=self.parameters.prior,
            sender_payoffs=sender_payoffs,
            receiver_payoffs=receiver_payoffs,
        )

    def posterior(
        self, warning: str, sender_strategy: Mapping[str, str]
    ) -> dict[str, float]:
        """Bayes-update at a reached warning, else apply configured convention."""

        if warning not in WARNINGS:
            raise ValueError(f"unknown warning: {warning}")
        if set(sender_strategy) != set(TYPES):
            raise ValueError("sender_strategy must specify exactly both China types")
        if any(message not in WARNINGS for message in sender_strategy.values()):
            raise ValueError("sender_strategy references an unknown warning")
        prior = self.parameters.prior
        reach = sum(prior[china_type] for china_type in TYPES if sender_strategy[china_type] == warning)
        if reach <= self.parameters.comparison_tolerance:
            high = self.parameters.off_path_high_belief
            return {"low_resolve": 1.0 - high, "high_resolve": high}
        return {
            china_type: (
                prior[china_type] / reach if sender_strategy[china_type] == warning else 0.0
            )
            for china_type in TYPES
        }

    def receiver_best_responses(
        self, warning: str, posterior: Mapping[str, float]
    ) -> tuple[str, ...]:
        """Best advances given the posterior actually held at the decision."""

        if warning not in WARNINGS:
            raise ValueError(f"unknown warning: {warning}")
        if set(posterior) != set(TYPES):
            raise ValueError("posterior must contain exactly both China types")
        probabilities = [float(posterior[china_type]) for china_type in TYPES]
        if any(value < 0.0 for value in probabilities) or not isclose(
            sum(probabilities), 1.0, rel_tol=0.0, abs_tol=self.parameters.comparison_tolerance
        ):
            raise ValueError("posterior must be a probability distribution")
        values: dict[str, float] = {}
        for action in ADVANCE_ACTIONS:
            values[action] = sum(
                posterior[china_type]
                * self.continuation_entry(china_type, warning, action).receiver_payoff
                for china_type in TYPES
            )
        best = max(values.values())
        return tuple(
            action
            for action in ADVANCE_ACTIONS
            if isclose(
                values[action],
                best,
                rel_tol=0.0,
                abs_tol=self.parameters.comparison_tolerance,
            )
        )

    def _off_path_convention(
        self, supplied: Mapping[str, Mapping[str, float]] | None
    ) -> dict[str, dict[str, float]]:
        if supplied is None:
            high = self.parameters.off_path_high_belief
            return {
                warning: {"low_resolve": 1.0 - high, "high_resolve": high}
                for warning in WARNINGS
            }
        if set(supplied) != set(WARNINGS):
            raise ValueError("off-path beliefs must specify both warning messages")
        result: dict[str, dict[str, float]] = {}
        for warning in WARNINGS:
            belief = supplied[warning]
            if set(belief) != set(TYPES):
                raise ValueError("each off-path belief must specify both China types")
            low = float(belief["low_resolve"])
            high = float(belief["high_resolve"])
            if low < 0.0 or high < 0.0 or not isclose(
                low + high,
                1.0,
                rel_tol=0.0,
                abs_tol=self.parameters.comparison_tolerance,
            ):
                raise ValueError("each off-path belief must be a probability distribution")
            result[warning] = {"low_resolve": low, "high_resolve": high}
        return result

    def solve(
        self, off_path_beliefs: Mapping[str, Mapping[str, float]] | None = None
    ) -> KoreaSolution:
        """Enumerate and retain every pure PBE under the belief convention."""

        started = perf_counter()
        convention = self._off_path_convention(off_path_beliefs)
        raw_result = enumerate_pure_pbe(
            self.reduced_signaling_game(),
            off_path_beliefs=convention,
            tolerance=self.parameters.comparison_tolerance,
        )
        equilibria = tuple(self._convert_equilibrium(raw) for raw in raw_result.solutions)
        return KoreaSolution(
            status=raw_result.status,
            found=raw_result.found,
            multiple=raw_result.multiple,
            equilibria=equilibria,
            continuation_entry_policy=self.continuation_policy(),
            off_path_belief_convention=convention,
            runtime_seconds=perf_counter() - started,
            best_response_gap=(
                max((equilibrium.best_response_gap for equilibrium in equilibria), default=0.0)
                if equilibria
                else None
            ),
            warnings=(
                "No mixed-strategy or partially pooling equilibria are searched.",
                "NO_PURE_PBE_FOUND, when reported, concerns only the restricted pure class.",
            ),
        )

    def _convert_equilibrium(self, raw: SignalingEquilibrium) -> KoreaEquilibrium:
        classification = raw.classification
        if classification not in ("pooling", "separating"):
            raise RuntimeError(
                "a two-type, two-message pure profile cannot be partially pooling"
            )
        sender_strategy = dict(raw.sender_strategy)
        receiver_strategy = dict(raw.receiver_strategy)
        posteriors = {
            message: dict(belief) for message, belief in raw.posteriors.items()
        }
        reach_probabilities = dict(raw.reach_probabilities)
        sender_gains = dict(raw.sender_deviation_gains)
        receiver_gains = dict(raw.receiver_deviation_gains)
        off_path_messages = raw.off_path_messages
        depends = raw.depends_on_off_path_beliefs

        prior = self.parameters.prior
        warning_probability = 0.0
        intervention_probability = 0.0
        action_frequencies = {action: 0.0 for action in ADVANCE_ACTIONS}
        expected_china = 0.0
        expected_receiver = 0.0
        realized_paths: list[dict[str, object]] = []
        for china_type in TYPES:
            probability = prior[china_type]
            warning = sender_strategy[china_type]
            receiver_action = receiver_strategy[warning]
            continuation = self.continuation_entry(china_type, warning, receiver_action)
            if warning == "warn":
                warning_probability += probability
            if continuation.selected_entry == "intervene":
                intervention_probability += probability
            action_frequencies[receiver_action] += probability
            expected_china += probability * continuation.china_payoff
            expected_receiver += probability * continuation.receiver_payoff
            realized_paths.append(
                {
                    "china_type": china_type,
                    "probability": probability,
                    "warning": warning,
                    "posterior_used_by_receiver": dict(posteriors[warning]),
                    "receiver_action": receiver_action,
                    "entry_action": continuation.selected_entry,
                    "utilities": {
                        "china": continuation.china_payoff,
                        "us_un": continuation.receiver_payoff,
                    },
                }
            )

        credibility = self._credibility_regime(
            classification, receiver_strategy, posteriors, depends
        )
        gap = sum(max(0.0, value) for value in sender_gains.values()) + sum(
            max(0.0, value) for value in receiver_gains.values()
        )
        return KoreaEquilibrium(
            classification=classification,
            sender_strategy=sender_strategy,
            receiver_strategy=receiver_strategy,
            posteriors=posteriors,
            reach_probabilities=reach_probabilities,
            expected_china_utility=expected_china,
            expected_receiver_utility=expected_receiver,
            warning_probability=warning_probability,
            intervention_probability=intervention_probability,
            receiver_action_frequencies=action_frequencies,
            credibility_regime=credibility,
            depends_on_off_path_beliefs=depends,
            off_path_messages=off_path_messages,
            sender_deviation_gains=sender_gains,
            receiver_deviation_gains=receiver_gains,
            best_response_gap=gap,
            realized_paths=tuple(realized_paths),
        )

    @staticmethod
    def _credibility_regime(
        classification: str,
        receiver_strategy: Mapping[str, str],
        posteriors: Mapping[str, Mapping[str, float]],
        depends_on_off_path_beliefs: bool,
    ) -> str:
        if classification == "pooling":
            return (
                "pooling_uninformative_off_path_dependent"
                if depends_on_off_path_beliefs
                else "pooling_uninformative"
            )
        warning_more_high = (
            posteriors["warn"]["high_resolve"] > posteriors["quiet"]["high_resolve"]
        )
        warning_restrains = (
            ACTION_RANK[receiver_strategy["warn"]] < ACTION_RANK[receiver_strategy["quiet"]]
        )
        if warning_more_high and warning_restrains:
            return "credible_behavior_changing_warning"
        if warning_more_high:
            return "informative_warning_no_behavior_change"
        return "separating_but_warning_not_high_resolve_signal"


def model_description() -> dict[str, object]:
    """Return the implemented formal ordering and claims."""

    return {
        "event": "korea",
        "historical_scenario": "Korean War intervention warning, historically inspired",
        "framework": "Bayesian warning and entry deterrence",
        "players": ["china", "us_un", "nature"],
        "type_space": list(TYPES),
        "warning_actions": list(WARNINGS),
        "receiver_actions": list(ADVANCE_ACTIONS),
        "entry_actions": list(ENTRY_ACTIONS),
        "move_order": [stage.as_dict() for stage in KoreaWarningModel.move_order()],
        "equilibrium_concept": "pure-strategy perfect Bayesian equilibrium",
        "solver": "exact continuation entry plus exhaustive pure-PBE enumeration",
        "partial_pooling_support": False,
        "calibration": "not historically calibrated; defaults are illustrative",
    }


def describe() -> dict[str, object]:
    """Return the model description."""

    return model_description()


def solve(parameters: KoreaParameters | None = None) -> KoreaSolution:
    """Convenience entry point retaining all pure equilibria."""

    return KoreaWarningModel(parameters).solve()

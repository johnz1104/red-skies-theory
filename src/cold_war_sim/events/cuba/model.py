"""Formal Cuba-inspired Bayesian crisis-bargaining model.

The game distinguishes four uncertainty sources:

* actor uncertainty: private Soviet resolve/readiness and a noisy public
  intelligence report observed by the United States;
* modeler uncertainty: parameters supplied from outside this game;
* strategic randomness: behavioral strategies when supplied for diagnostics;
* exogenous randomness: action-dependent accidental catastrophe.

For tractable exact pure-PBE enumeration, the two consecutive communications
are reduced to their four realized ordered histories only after the explicit
game history is constructed. No actor or chance event occurs between the two
communications, so deviations over ordered histories cover every on-path pure
deviation. Off-path second-message completions are recorded separately by the
solver adapter.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import StrEnum
from itertools import product
from math import isfinite
from typing import Final, NotRequired, TypedDict

import numpy as np

from cold_war_sim.events.cuba.parameters import CubaParameters


class Resolve(StrEnum):
    LOW = "low"
    HIGH = "high"


class Readiness(StrEnum):
    LOW = "low"
    HIGH = "high"


class Intelligence(StrEnum):
    REASSURING = "reassuring"
    ALARMING = "alarming"


class InitialAction(StrEnum):
    QUARANTINE = "quarantine"
    AIR_STRIKE = "air_strike"


class FirstCommunication(StrEnum):
    CONCILIATORY = "conciliatory"
    HARDLINE = "hardline"


class SecondCommunication(StrEnum):
    REAFFIRM = "reaffirm"
    REVISE = "revise"


class FinalResponse(StrEnum):
    NEGOTIATE = "negotiate"
    MAINTAIN = "maintain_pressure"
    ESCALATE = "escalate"


TYPE_ORDER: Final[tuple[str, ...]] = (
    "resolve_low__readiness_low",
    "resolve_low__readiness_high",
    "resolve_high__readiness_low",
    "resolve_high__readiness_high",
)
INTELLIGENCE_ORDER: Final[tuple[str, ...]] = tuple(item.value for item in Intelligence)
INITIAL_ACTION_ORDER: Final[tuple[str, ...]] = tuple(item.value for item in InitialAction)
FIRST_COMMUNICATION_ORDER: Final[tuple[str, ...]] = tuple(
    item.value for item in FirstCommunication
)
SECOND_COMMUNICATION_ORDER: Final[tuple[str, ...]] = tuple(
    item.value for item in SecondCommunication
)
FINAL_RESPONSE_ORDER: Final[tuple[str, ...]] = tuple(item.value for item in FinalResponse)


@dataclass(frozen=True, slots=True)
class SovietType:
    resolve: Resolve
    readiness: Readiness

    @property
    def label(self) -> str:
        return f"resolve_{self.resolve.value}__readiness_{self.readiness.value}"


SOVIET_TYPES: Final[tuple[SovietType, ...]] = tuple(
    SovietType(resolve, readiness)
    for resolve, readiness in product(tuple(Resolve), tuple(Readiness))
)


@dataclass(frozen=True, slots=True)
class CommunicationHistory:
    first: FirstCommunication
    second: SecondCommunication

    @property
    def label(self) -> str:
        return f"{self.first.value}__{self.second.value}"

    @property
    def effective_stance(self) -> FirstCommunication:
        if self.second is SecondCommunication.REAFFIRM:
            return self.first
        return (
            FirstCommunication.HARDLINE
            if self.first is FirstCommunication.CONCILIATORY
            else FirstCommunication.CONCILIATORY
        )


COMMUNICATION_HISTORIES: Final[tuple[CommunicationHistory, ...]] = tuple(
    CommunicationHistory(first, second)
    for first, second in product(tuple(FirstCommunication), tuple(SecondCommunication))
)
COMMUNICATION_ORDER: Final[tuple[str, ...]] = tuple(
    history.label for history in COMMUNICATION_HISTORIES
)


@dataclass(frozen=True, slots=True)
class HistoryEvent:
    stage: int
    actor: str
    action: str
    observed_by: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CubaTerminal:
    type_label: str
    intelligence: str
    initial_action: str
    first_communication: str
    second_communication: str
    final_response: str
    peaceful_category: str
    missile_removal: float
    war_intensity: float
    escalation_probability: float
    catastrophe_probability: float
    utility_us_no_catastrophe: float
    utility_ussr_no_catastrophe: float
    expected_utility_us: float
    expected_utility_ussr: float
    signal_cost: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class _FullInformationRecord(TypedDict):
    """Internal shape retained through the backward-induction tie sets."""

    first: str
    second: str
    response: str
    utility_us: float
    utility_ussr: float
    terminal: dict[str, object]
    initial_action: NotRequired[str]


class CubaModel:
    """Validated finite Cuba-inspired game specification."""

    players: Final[tuple[str, str]] = ("United States", "Soviet Union")
    nature: Final[str] = "Nature"

    def __init__(self, parameters: CubaParameters | None = None) -> None:
        self.parameters = parameters or CubaParameters()

    def type_prior(self) -> dict[str, float]:
        """Independent illustrative prior over resolve by readiness."""

        p = self.parameters
        result: dict[str, float] = {}
        for soviet_type in SOVIET_TYPES:
            resolve_probability = (
                p.prior_resolve_high
                if soviet_type.resolve is Resolve.HIGH
                else 1.0 - p.prior_resolve_high
            )
            readiness_probability = (
                p.prior_readiness_high
                if soviet_type.readiness is Readiness.HIGH
                else 1.0 - p.prior_readiness_high
            )
            result[soviet_type.label] = resolve_probability * readiness_probability
        self._validate_distribution(result.values(), "type prior")
        return result

    def intelligence_likelihood(
        self, soviet_type: SovietType, intelligence: Intelligence
    ) -> float:
        """Likelihood of a noisy public report about Soviet readiness."""

        correct = (
            intelligence is Intelligence.ALARMING
            and soviet_type.readiness is Readiness.HIGH
        ) or (
            intelligence is Intelligence.REASSURING
            and soviet_type.readiness is Readiness.LOW
        )
        return (
            self.parameters.intelligence_accuracy
            if correct
            else 1.0 - self.parameters.intelligence_accuracy
        )

    def intelligence_probability(self, intelligence: Intelligence) -> float:
        prior = self.type_prior()
        return float(
            sum(
                prior[soviet_type.label]
                * self.intelligence_likelihood(soviet_type, intelligence)
                for soviet_type in SOVIET_TYPES
            )
        )

    def posterior_after_intelligence(self, intelligence: Intelligence) -> dict[str, float]:
        """Bayes-consistent posterior over all four Soviet types."""

        prior = self.type_prior()
        evidence = self.intelligence_probability(intelligence)
        if evidence <= self.parameters.probability_tolerance:
            raise ValueError(f"intelligence report {intelligence.value!r} has zero probability")
        posterior = {
            soviet_type.label: (
                prior[soviet_type.label]
                * self.intelligence_likelihood(soviet_type, intelligence)
                / evidence
            )
            for soviet_type in SOVIET_TYPES
        }
        self._validate_distribution(posterior.values(), "intelligence posterior")
        return posterior

    def formal_history(
        self,
        soviet_type: SovietType,
        intelligence: Intelligence,
        initial_action: InitialAction,
        communication: CommunicationHistory,
        response: FinalResponse,
    ) -> tuple[HistoryEvent, ...]:
        """Return the explicit ordered history; both communications always occur."""

        return (
            HistoryEvent(0, self.nature, soviet_type.label, ("Soviet Union",)),
            HistoryEvent(
                1,
                self.nature,
                intelligence.value,
                ("United States", "Soviet Union"),
            ),
            HistoryEvent(2, "United States", initial_action.value, self.players),
            HistoryEvent(3, "Soviet Union", communication.first.value, self.players),
            HistoryEvent(4, "Soviet Union", communication.second.value, self.players),
            HistoryEvent(5, "United States", response.value, self.players),
            HistoryEvent(6, self.nature, "accident_or_control", self.players),
        )

    def signal_cost(
        self, soviet_type: SovietType, communication: CommunicationHistory
    ) -> float:
        p = self.parameters
        if communication.first is FirstCommunication.CONCILIATORY:
            first_cost = (
                p.first_conciliatory_cost_high
                if soviet_type.resolve is Resolve.HIGH
                else p.first_conciliatory_cost_low
            )
        else:
            first_cost = (
                p.first_hardline_cost_high
                if soviet_type.resolve is Resolve.HIGH
                else p.first_hardline_cost_low
            )
        second_cost = (
            p.second_reaffirm_cost
            if communication.second is SecondCommunication.REAFFIRM
            else p.second_revise_cost + p.inconsistency_cost
        )
        return first_cost + second_cost

    def terminal(
        self,
        soviet_type: SovietType,
        intelligence: Intelligence,
        initial_action: InitialAction,
        communication: CommunicationHistory,
        response: FinalResponse,
    ) -> CubaTerminal:
        """Evaluate a terminal history, integrating exogenous catastrophe risk."""

        p = self.parameters
        effective_hardline = communication.effective_stance is FirstCommunication.HARDLINE

        if response is FinalResponse.NEGOTIATE:
            category = "negotiated"
            missile_removal = 0.82 if soviet_type.resolve is Resolve.LOW else 0.62
            war = 0.0
            us_credibility = 0.20 if not effective_hardline else 0.05
            ussr_prestige = 0.15 if effective_hardline else -0.15
            cuba_safe = 0.90
            concessions = 0.70
            controlled_escalation = 0.01
        elif response is FinalResponse.MAINTAIN:
            if soviet_type.resolve is Resolve.LOW:
                category = "coercive"
                missile_removal = 0.93
                war = 0.04
                us_credibility = 0.70
                ussr_prestige = -0.60
                cuba_safe = 0.78
                concessions = 0.18
                controlled_escalation = 0.06
            else:
                category = (
                    "escalatory"
                    if soviet_type.readiness is Readiness.HIGH
                    else "peaceful_standoff"
                )
                missile_removal = 0.23
                war = 0.24 if soviet_type.readiness is Readiness.HIGH else 0.09
                us_credibility = -0.15
                ussr_prestige = 0.62
                cuba_safe = 0.95
                concessions = 0.0
                controlled_escalation = (
                    0.34 if soviet_type.readiness is Readiness.HIGH else 0.16
                )
        else:
            category = "escalatory"
            missile_removal = 0.86 if soviet_type.readiness is Readiness.LOW else 0.54
            war = 0.82 if soviet_type.readiness is Readiness.LOW else 1.0
            us_credibility = 0.48
            ussr_prestige = 0.18
            cuba_safe = 0.25
            concessions = 0.0
            controlled_escalation = 0.90

        if initial_action is InitialAction.AIR_STRIKE:
            category = "escalatory"
            missile_removal = min(1.0, missile_removal + 0.08)
            war = min(1.0, war + 0.42)
            us_credibility = min(1.0, us_credibility + 0.10)
            cuba_safe = max(0.0, cuba_safe - 0.30)
            controlled_escalation = max(controlled_escalation, 0.78)

        first_accident = (
            p.air_strike_accident_probability
            if initial_action is InitialAction.AIR_STRIKE
            else p.quarantine_accident_probability
        )
        response_accident = {
            FinalResponse.NEGOTIATE: p.negotiate_accident_probability,
            FinalResponse.MAINTAIN: p.maintain_accident_probability,
            FinalResponse.ESCALATE: p.escalate_accident_probability,
        }[response]
        risk_multiplier = (
            p.high_readiness_risk_multiplier
            if soviet_type.readiness is Readiness.HIGH
            else 1.0
        )
        first_accident = min(1.0, first_accident * risk_multiplier)
        response_accident = min(1.0, response_accident * risk_multiplier)
        catastrophe_probability = 1.0 - (1.0 - first_accident) * (
            1.0 - response_accident
        )
        escalation_probability = catastrophe_probability + (
            1.0 - catastrophe_probability
        ) * controlled_escalation

        utility_us = (
            missile_removal * p.missile_value_us
            + us_credibility * p.credibility_value
            - concessions * p.concession_value
            - war * p.conflict_cost_us
        )
        utility_ussr = (
            (1.0 - missile_removal) * p.missile_value_ussr
            + ussr_prestige * p.credibility_value
            + cuba_safe * p.cuba_security_value
            + concessions * p.concession_value
            - war * p.conflict_cost_ussr
            - self.signal_cost(soviet_type, communication)
        )
        catastrophe_us = -p.catastrophe_loss * p.risk_sensitivity_us
        catastrophe_ussr = -p.catastrophe_loss * p.risk_sensitivity_ussr
        expected_us = (1.0 - catastrophe_probability) * utility_us + (
            catastrophe_probability * catastrophe_us
        )
        expected_ussr = (1.0 - catastrophe_probability) * utility_ussr + (
            catastrophe_probability * catastrophe_ussr
        )
        numeric = (
            missile_removal,
            war,
            escalation_probability,
            catastrophe_probability,
            utility_us,
            utility_ussr,
            expected_us,
            expected_ussr,
        )
        if not all(isfinite(value) for value in numeric):
            raise ArithmeticError("Cuba terminal calculation produced a non-finite value")
        if not 0.0 <= catastrophe_probability <= escalation_probability <= 1.0:
            raise ArithmeticError("Cuba terminal produced invalid event probabilities")

        return CubaTerminal(
            type_label=soviet_type.label,
            intelligence=intelligence.value,
            initial_action=initial_action.value,
            first_communication=communication.first.value,
            second_communication=communication.second.value,
            final_response=response.value,
            peaceful_category=category,
            missile_removal=missile_removal,
            war_intensity=war,
            escalation_probability=escalation_probability,
            catastrophe_probability=catastrophe_probability,
            utility_us_no_catastrophe=utility_us,
            utility_ussr_no_catastrophe=utility_ussr,
            expected_utility_us=expected_us,
            expected_utility_ussr=expected_ussr,
            signal_cost=self.signal_cost(soviet_type, communication),
        )

    def continuation_payoff_arrays(
        self, intelligence: Intelligence, initial_action: InitialAction
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return prior, sender utilities, and receiver utilities in canonical order."""

        posterior = self.posterior_after_intelligence(intelligence)
        prior = np.asarray([posterior[label] for label in TYPE_ORDER], dtype=float)
        sender = np.empty(
            (len(TYPE_ORDER), len(COMMUNICATION_HISTORIES), len(FinalResponse)),
            dtype=float,
        )
        receiver = np.empty_like(sender)
        for type_index, soviet_type in enumerate(SOVIET_TYPES):
            for communication_index, communication in enumerate(COMMUNICATION_HISTORIES):
                for response_index, response in enumerate(FinalResponse):
                    outcome = self.terminal(
                        soviet_type,
                        intelligence,
                        initial_action,
                        communication,
                        response,
                    )
                    sender[type_index, communication_index, response_index] = (
                        outcome.expected_utility_ussr
                    )
                    receiver[type_index, communication_index, response_index] = (
                        outcome.expected_utility_us
                    )
        if not np.all(np.isfinite(sender)) or not np.all(np.isfinite(receiver)):
            raise ArithmeticError("continuation payoff array contains non-finite values")
        return prior, sender, receiver

    def full_information_backward_induction(self, soviet_type: SovietType) -> dict[str, object]:
        """Solve the revealed-type game exactly by backward induction.

        Ties are retained at every stage. The returned root outcomes are the
        finite-game subgame-perfect outcomes under the illustrative utilities.
        """

        tol = self.parameters.best_response_tolerance
        intelligence_results: dict[str, list[_FullInformationRecord]] = {}
        # With the type revealed, the intelligence report is payoff-irrelevant but
        # remains in the explicit history; solve each possible public report.
        for intelligence in Intelligence:
            initial_records: list[_FullInformationRecord] = []
            for initial in InitialAction:
                first_records: list[_FullInformationRecord] = []
                for first in FirstCommunication:
                    second_records: list[_FullInformationRecord] = []
                    for second in SecondCommunication:
                        communication = CommunicationHistory(first, second)
                        terminals = [
                            self.terminal(
                                soviet_type,
                                intelligence,
                                initial,
                                communication,
                                response,
                            )
                            for response in FinalResponse
                        ]
                        best_us = max(item.expected_utility_us for item in terminals)
                        best_terminals = [
                            item
                            for item in terminals
                            if item.expected_utility_us >= best_us - tol
                        ]
                        # For a tied US response the correspondence is retained;
                        # the USSR evaluates all possible tied continuations.
                        second_records.extend(
                            {
                                "first": first.value,
                                "second": second.value,
                                "response": item.final_response,
                                "utility_us": item.expected_utility_us,
                                "utility_ussr": item.expected_utility_ussr,
                                "terminal": item.to_dict(),
                            }
                            for item in best_terminals
                        )
                    best_ussr_second = max(
                        item["utility_ussr"] for item in second_records
                    )
                    first_records.extend(
                        item
                        for item in second_records
                        if item["utility_ussr"] >= best_ussr_second - tol
                    )
                best_ussr_first = max(item["utility_ussr"] for item in first_records)
                for item in first_records:
                    if item["utility_ussr"] >= best_ussr_first - tol:
                        initial_record: _FullInformationRecord = {
                            **item,
                            "initial_action": initial.value,
                        }
                        initial_records.append(initial_record)
            best_us_initial = max(item["utility_us"] for item in initial_records)
            intelligence_results[intelligence.value] = [
                item
                for item in initial_records
                if item["utility_us"] >= best_us_initial - tol
            ]
        return {
            "solver_name": "cuba_full_information_backward_induction",
            "solver_version": "1.0",
            "equilibrium_concept": "finite-game subgame-perfect equilibrium",
            "exactness": "exact_enumeration_with_tolerance",
            "type": soviet_type.label,
            "solutions_by_intelligence": intelligence_results,
            "multiple_solutions": any(
                len(items) > 1 for items in intelligence_results.values()
            ),
            "assumptions": [
                "Soviet resolve and readiness are revealed to both players.",
                "All numeric utilities and probabilities are illustrative assumptions.",
                "Tied continuations are retained rather than silently selected.",
            ],
        }

    def describe(self) -> dict[str, object]:
        return {
            "event": "cuba",
            "historical_scenario": "Cuban Missile Crisis-inspired confrontation",
            "framework": "finite Bayesian extensive-form crisis bargaining and signaling",
            "players": list(self.players),
            "nature": self.nature,
            "type_space": list(TYPE_ORDER),
            "move_order": [
                "Nature draws Soviet resolve and readiness.",
                "Nature emits a noisy public readiness report.",
                "United States chooses quarantine or air strike.",
                "Soviet Union sends a first communication.",
                "Soviet Union sends a second communication after the first.",
                "United States responds after observing both communications.",
                "Nature resolves action-dependent accidental catastrophe.",
            ],
            "observations": {
                "soviet_union": ["own type", "intelligence report", "full public history"],
                "united_states": ["intelligence report", "full public history except type"],
            },
            "initial_actions": list(INITIAL_ACTION_ORDER),
            "first_communications": list(FIRST_COMMUNICATION_ORDER),
            "second_communications": list(SECOND_COMMUNICATION_ORDER),
            "final_responses": list(FINAL_RESPONSE_ORDER),
            "equilibrium_concepts": [
                "subgame-perfect equilibrium for revealed-type benchmark",
                "restricted pure-strategy PBE for incomplete-information game",
            ],
            "parameter_status": CubaParameters.provenance,
            "limitations": [
                "Historically inspired and not empirically calibrated.",
                "Intelligence is public in the baseline to keep exact enumeration tractable.",
                "The exact incomplete-information solver enumerates pure strategies only.",
                "No actor moves between the two modeled communications.",
            ],
        }

    def _validate_distribution(self, values: Iterable[float], name: str) -> None:
        array = np.asarray(tuple(values), dtype=float)
        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"{name} must be a nonempty one-dimensional distribution")
        if not np.all(np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError(f"{name} must contain finite nonnegative values")
        if not np.isclose(
            float(array.sum()), 1.0, atol=self.parameters.probability_tolerance, rtol=0.0
        ):
            raise ValueError(f"{name} must sum to one")

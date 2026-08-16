"""Finite-horizon alternating-offers model of a Berlin confrontation.

This is a dynamic bargaining game, not a renamed crisis-signaling tree.  A
rejected offer can escalate exogenously; conditional on no escalation, the
other player proposes next period.  Backward induction therefore supplies an
exact finite-horizon subgame-perfect equilibrium (SPE), subject only to the
documented deterministic tie conventions in :mod:`.parameters`.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from time import perf_counter
from typing import Literal

from .parameters import BerlinParameters, PlayerName

Response = Literal["accept", "reject"]
PayoffPair = tuple[float, float]

PLAYER_INDEX: dict[PlayerName, int] = {"west": 0, "soviet": 1}


def _other(player: PlayerName) -> PlayerName:
    return "soviet" if player == "west" else "west"


@dataclass(frozen=True, slots=True)
class OfferEvaluation:
    """Responder action and induced payoff for one possible offer."""

    west_share: float
    response: Response
    settlement_payoffs: PayoffPair
    rejection_payoffs: PayoffPair
    induced_payoffs: PayoffPair

    def as_dict(self) -> dict[str, object]:
        return {
            "west_share": self.west_share,
            "response": self.response,
            "settlement_payoffs": {
                "west": self.settlement_payoffs[0],
                "soviet": self.settlement_payoffs[1],
            },
            "rejection_payoffs": {
                "west": self.rejection_payoffs[0],
                "soviet": self.rejection_payoffs[1],
            },
            "induced_payoffs": {
                "west": self.induced_payoffs[0],
                "soviet": self.induced_payoffs[1],
            },
        }


@dataclass(frozen=True, slots=True)
class BargainingDecision:
    """Complete equilibrium policy at one proper subgame."""

    period: int
    proposer: PlayerName
    responder: PlayerName
    escalation_risk_after_rejection: float
    selected_west_share: float
    selected_response: Response
    equilibrium_payoffs: PayoffPair
    proposer_optimal_offers: tuple[float, ...]
    offer_evaluations: tuple[OfferEvaluation, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "period": self.period,
            "proposer": self.proposer,
            "responder": self.responder,
            "escalation_risk_after_rejection": self.escalation_risk_after_rejection,
            "selected_west_share": self.selected_west_share,
            "selected_response": self.selected_response,
            "equilibrium_payoffs": {
                "west": self.equilibrium_payoffs[0],
                "soviet": self.equilibrium_payoffs[1],
            },
            "proposer_optimal_offers": list(self.proposer_optimal_offers),
            "offer_evaluations": [evaluation.as_dict() for evaluation in self.offer_evaluations],
        }


@dataclass(frozen=True, slots=True)
class BerlinMetrics:
    agreement_probability: float
    escalation_probability: float
    impasse_probability: float
    concession_probability: float
    expected_bargaining_duration: float
    expected_west_share_conditional_agreement: float | None
    west_expected_utility: float
    soviet_expected_utility: float

    def as_dict(self) -> dict[str, object]:
        return {
            "agreement_probability": self.agreement_probability,
            "escalation_probability": self.escalation_probability,
            "impasse_probability": self.impasse_probability,
            "concession_probability": self.concession_probability,
            "expected_bargaining_duration": self.expected_bargaining_duration,
            "expected_west_share_conditional_agreement": (
                self.expected_west_share_conditional_agreement
            ),
            "west_expected_utility": self.west_expected_utility,
            "soviet_expected_utility": self.soviet_expected_utility,
        }


@dataclass(frozen=True, slots=True)
class BerlinSolution:
    """Exact backward-induction result for the finite game."""

    solver_name: str
    solver_version: str
    equilibrium_concept: str
    exact: bool
    found: bool
    multiple_due_to_ties: bool
    runtime_seconds: float
    tie_conventions: dict[str, str]
    policy: tuple[BargainingDecision, ...]
    metrics: BerlinMetrics
    warnings: tuple[str, ...]
    assumptions: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "solver": {
                "name": self.solver_name,
                "version": self.solver_version,
                "equilibrium_concept": self.equilibrium_concept,
                "found": self.found,
                "multiple_solutions": self.multiple_due_to_ties,
                "exactness": "exact_finite_grid" if self.exact else "approximate",
                "convergence_status": "not_applicable_exact_enumeration",
                "best_response_gap": 0.0,
                "runtime_seconds": self.runtime_seconds,
                "seed": None,
                "tie_conventions": dict(self.tie_conventions),
                "warnings": list(self.warnings),
                "assumptions": list(self.assumptions),
            },
            "metrics": self.metrics.as_dict(),
            "policy": [decision.as_dict() for decision in self.policy],
            "warnings": list(self.warnings),
            "assumptions": list(self.assumptions),
        }


class BerlinBargainingModel:
    """Alternating-offers bargaining with risky rejection and finite horizon."""

    def __init__(self, parameters: BerlinParameters | None = None) -> None:
        self.parameters = parameters or BerlinParameters()

    def proposer_at(self, period: int) -> PlayerName:
        if period < 0 or period >= self.parameters.horizon:
            raise ValueError("period is outside the bargaining horizon")
        if period % 2 == 0:
            return self.parameters.initial_proposer
        return _other(self.parameters.initial_proposer)

    def settlement_payoffs(self, period: int, west_share: float) -> PayoffPair:
        """Present-value payoffs when an offer is accepted in ``period``."""

        params = self.parameters
        if period < 0 or period >= params.horizon:
            raise ValueError("period is outside the bargaining horizon")
        if west_share not in params.settlement_grid:
            raise ValueError("west_share is not on the settlement grid")
        west_prize = params.settlement_surplus * west_share
        soviet_prize = params.settlement_surplus * (1.0 - west_share)
        west_commitment_penalty = params.west_commitment_cost * max(
            0.0, params.west_commitment_floor - west_share
        )
        soviet_commitment_penalty = params.soviet_commitment_cost * max(
            0.0, west_share - params.soviet_commitment_ceiling
        )
        west = (
            params.west_discount**period * (west_prize - west_commitment_penalty)
            - period * params.west_delay_cost
        )
        soviet = (
            params.soviet_discount**period * (soviet_prize - soviet_commitment_penalty)
            - period * params.soviet_delay_cost
        )
        return (west, soviet)

    def escalation_payoffs(self, elapsed_periods: int) -> PayoffPair:
        """Present-value utilities after uncontrolled escalation."""

        params = self.parameters
        if elapsed_periods < 1 or elapsed_periods > params.horizon:
            raise ValueError("elapsed_periods is outside the supported horizon")
        return (
            -(params.west_discount**elapsed_periods) * params.west_escalation_loss
            - elapsed_periods * params.west_delay_cost,
            -(params.soviet_discount**elapsed_periods) * params.soviet_escalation_loss
            - elapsed_periods * params.soviet_delay_cost,
        )

    def impasse_payoffs(self, elapsed_periods: int) -> PayoffPair:
        """Present-value reservation utilities when the horizon expires."""

        params = self.parameters
        if elapsed_periods != params.horizon:
            raise ValueError("impasse occurs only when the bargaining horizon expires")
        return (
            params.west_discount**elapsed_periods * params.west_reservation
            - elapsed_periods * params.west_delay_cost,
            params.soviet_discount**elapsed_periods * params.soviet_reservation
            - elapsed_periods * params.soviet_delay_cost,
        )

    def _rejection_payoffs(
        self, period: int, next_period_payoffs: PayoffPair | None
    ) -> PayoffPair:
        risk = self.parameters.escalation_risk(period)
        escalation = self.escalation_payoffs(period + 1)
        continuation = (
            next_period_payoffs
            if next_period_payoffs is not None
            else self.impasse_payoffs(self.parameters.horizon)
        )
        return (
            risk * escalation[0] + (1.0 - risk) * continuation[0],
            risk * escalation[1] + (1.0 - risk) * continuation[1],
        )

    def _response(
        self, responder_index: int, settlement: PayoffPair, rejection: PayoffPair
    ) -> Response:
        difference = settlement[responder_index] - rejection[responder_index]
        if difference > self.parameters.comparison_tolerance:
            return "accept"
        if difference < -self.parameters.comparison_tolerance:
            return "reject"
        return self.parameters.responder_tie_break

    def _choose_offer(
        self, proposer_index: int, evaluations: tuple[OfferEvaluation, ...]
    ) -> tuple[OfferEvaluation, tuple[float, ...]]:
        best_payoff = max(item.induced_payoffs[proposer_index] for item in evaluations)
        optimal = tuple(
            item
            for item in evaluations
            if isclose(
                item.induced_payoffs[proposer_index],
                best_payoff,
                rel_tol=0.0,
                abs_tol=self.parameters.comparison_tolerance,
            )
        )
        if self.parameters.offer_tie_break == "lowest_west_share":
            selected = min(optimal, key=lambda item: item.west_share)
        else:
            selected = max(optimal, key=lambda item: item.west_share)
        return selected, tuple(item.west_share for item in optimal)

    def solve(self) -> BerlinSolution:
        """Solve every proper subgame by exact backward induction."""

        started = perf_counter()
        params = self.parameters
        reverse_policy: list[BargainingDecision] = []
        next_period_payoffs: PayoffPair | None = None
        for period in range(params.horizon - 1, -1, -1):
            proposer = self.proposer_at(period)
            responder = _other(proposer)
            proposer_index = PLAYER_INDEX[proposer]
            responder_index = PLAYER_INDEX[responder]
            rejection = self._rejection_payoffs(period, next_period_payoffs)
            evaluations_list: list[OfferEvaluation] = []
            for west_share in params.settlement_grid:
                settlement = self.settlement_payoffs(period, west_share)
                response = self._response(responder_index, settlement, rejection)
                induced = settlement if response == "accept" else rejection
                evaluations_list.append(
                    OfferEvaluation(
                        west_share=west_share,
                        response=response,
                        settlement_payoffs=settlement,
                        rejection_payoffs=rejection,
                        induced_payoffs=induced,
                    )
                )
            evaluations = tuple(evaluations_list)
            selected, optimal_offers = self._choose_offer(proposer_index, evaluations)
            decision = BargainingDecision(
                period=period,
                proposer=proposer,
                responder=responder,
                escalation_risk_after_rejection=params.escalation_risk(period),
                selected_west_share=selected.west_share,
                selected_response=selected.response,
                equilibrium_payoffs=selected.induced_payoffs,
                proposer_optimal_offers=optimal_offers,
                offer_evaluations=evaluations,
            )
            reverse_policy.append(decision)
            next_period_payoffs = decision.equilibrium_payoffs

        policy = tuple(reversed(reverse_policy))
        root = policy[0].equilibrium_payoffs
        metrics = self._path_metrics(policy, root)
        return BerlinSolution(
            solver_name="berlin_finite_horizon_backward_induction",
            solver_version="1.0",
            equilibrium_concept="finite-horizon subgame-perfect equilibrium",
            exact=True,
            found=True,
            multiple_due_to_ties=any(len(item.proposer_optimal_offers) > 1 for item in policy),
            runtime_seconds=perf_counter() - started,
            tie_conventions={
                "responder": params.responder_tie_break,
                "proposer": params.offer_tie_break,
            },
            policy=policy,
            metrics=metrics,
            warnings=(),
            assumptions=(
                "All numerical parameters are illustrative, normalized, or numerical conveniences.",
                "Escalation after rejection is exogenous and increases weakly with duration.",
                "Settlement offers are restricted to the configured finite grid.",
            ),
        )

    def _path_metrics(
        self, policy: tuple[BargainingDecision, ...], root: PayoffPair
    ) -> BerlinMetrics:
        survival_probability = 1.0
        agreement_probability = 0.0
        escalation_probability = 0.0
        impasse_probability = 0.0
        concession_probability = 0.0
        duration = 0.0
        west_share_mass = 0.0

        for decision in policy:
            terminal_periods = float(decision.period + 1)
            if decision.selected_response == "accept":
                agreement_probability += survival_probability
                duration += survival_probability * terminal_periods
                west_share_mass += survival_probability * decision.selected_west_share
                responder_share = (
                    1.0 - decision.selected_west_share
                    if decision.proposer == "west"
                    else decision.selected_west_share
                )
                if responder_share > self.parameters.comparison_tolerance:
                    concession_probability += survival_probability
                survival_probability = 0.0
                break

            risk = decision.escalation_risk_after_rejection
            escalation_here = survival_probability * risk
            escalation_probability += escalation_here
            duration += escalation_here * terminal_periods
            survival_probability *= 1.0 - risk
            if decision.period == self.parameters.horizon - 1:
                impasse_probability += survival_probability
                duration += survival_probability * terminal_periods
                survival_probability = 0.0

        total = agreement_probability + escalation_probability + impasse_probability
        if not isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError(f"terminal outcome probabilities sum to {total}, not one")
        conditional_share = (
            west_share_mass / agreement_probability if agreement_probability > 0.0 else None
        )
        return BerlinMetrics(
            agreement_probability=agreement_probability,
            escalation_probability=escalation_probability,
            impasse_probability=impasse_probability,
            concession_probability=concession_probability,
            expected_bargaining_duration=duration,
            expected_west_share_conditional_agreement=conditional_share,
            west_expected_utility=root[0],
            soviet_expected_utility=root[1],
        )


def model_description() -> dict[str, object]:
    """Return a machine-readable formal description."""

    return {
        "event": "berlin",
        "historical_scenario": "Berlin confrontation, historically inspired",
        "framework": "finite-horizon alternating-offers bargaining with risky delay",
        "players": ["west", "soviet"],
        "move_order": [
            "current proposer chooses a settlement share",
            "responder accepts or rejects",
            "rejection may cause exogenous escalation",
            "conditional on survival, proposer alternates",
        ],
        "equilibrium_concept": "finite-horizon subgame-perfect equilibrium",
        "solver": "exact backward induction on a finite settlement grid",
        "calibration": "not historically calibrated; defaults are illustrative",
    }


def describe() -> dict[str, object]:
    """Return the model description."""

    return model_description()


def solve(parameters: BerlinParameters | None = None) -> BerlinSolution:
    """Convenience entry point for an exact Berlin solution."""

    return BerlinBargainingModel(parameters).solve()

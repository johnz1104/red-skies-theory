"""Exact pure-PBE enumeration for canonical finite signaling games."""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cold_war_sim.core.probability import (
    DEFAULT_TOLERANCE,
    ProbabilityDistribution,
    validate_probability_distribution,
)
from cold_war_sim.core.results import SolverResult
from cold_war_sim.core.types import Player, SerializableMixin, frozen_mapping, labels

SOLVER_NAME = "pure_pbe_enumerator"
SOLVER_VERSION = "1.0.0"
NO_PURE_PBE_FOUND = "NO_PURE_PBE_FOUND"

BeliefVector = Mapping[str, float]
BeliefSpecification = Mapping[
    str,
    BeliefVector | Sequence[BeliefVector],
]


def _identifier(value: str | Player, *, role: str) -> str:
    identifier = value.id if isinstance(value, Player) else str(value)
    if not identifier or identifier != identifier.strip():
        raise ValueError(f"{role} must be a non-empty, trimmed identifier")
    return identifier


def _choose_alias(
    primary: Sequence[str] | None,
    aliases: Sequence[Sequence[str] | None],
    *,
    name: str,
) -> tuple[str, ...]:
    supplied = [value for value in (primary, *aliases) if value is not None]
    if not supplied:
        raise TypeError(f"{name} is required")
    first = tuple(supplied[0])
    if any(tuple(value) != first for value in supplied[1:]):
        raise ValueError(f"conflicting aliases supplied for {name}")
    return labels(first)


@dataclass(frozen=True, init=False)
class SignalingGame(SerializableMixin):
    """A canonical sender-type/message/receiver-action signaling game.

    Payoff tensors have shape ``(types, messages, receiver_actions)``.  The
    constructor accepts the concise names ``types``, ``messages``, ``actions``
    and the explicit aliases ``type_labels``, ``message_labels``, and
    ``receiver_action_labels`` used by event modules.
    """

    sender: str
    receiver: str
    types: tuple[str, ...]
    messages: tuple[str, ...]
    actions: tuple[str, ...]
    prior: Mapping[str, float]
    sender_payoffs: np.ndarray
    receiver_payoffs: np.ndarray
    assumptions: tuple[str, ...]

    def __init__(
        self,
        sender: str | Player = "sender",
        receiver: str | Player = "receiver",
        types: Sequence[str] | None = None,
        messages: Sequence[str] | None = None,
        actions: Sequence[str] | None = None,
        prior: Mapping[str, float] | Sequence[float] | None = None,
        sender_payoffs: Mapping[tuple[str, str, str], float] | Any = None,
        receiver_payoffs: Mapping[tuple[str, str, str], float] | Any = None,
        assumptions: Sequence[str] = (),
        *,
        sender_types: Sequence[str] | None = None,
        type_labels: Sequence[str] | None = None,
        message_labels: Sequence[str] | None = None,
        receiver_actions: Sequence[str] | None = None,
        receiver_action_labels: Sequence[str] | None = None,
        sender_name: str | None = None,
        receiver_name: str | None = None,
    ) -> None:
        if sender_name is not None:
            if sender != "sender" and _identifier(sender, role="sender") != sender_name:
                raise ValueError("conflicting sender and sender_name values")
            sender = sender_name
        if receiver_name is not None:
            if (
                receiver != "receiver"
                and _identifier(receiver, role="receiver") != receiver_name
            ):
                raise ValueError("conflicting receiver and receiver_name values")
            receiver = receiver_name
        sender_id = _identifier(sender, role="sender")
        receiver_id = _identifier(receiver, role="receiver")
        if sender_id == receiver_id:
            raise ValueError("sender and receiver must have distinct identifiers")
        type_ids = _choose_alias(types, (sender_types, type_labels), name="type labels")
        message_ids = _choose_alias(messages, (message_labels,), name="message labels")
        action_ids = _choose_alias(
            actions,
            (receiver_actions, receiver_action_labels),
            name="receiver-action labels",
        )
        if prior is None:
            raise TypeError("prior is required")
        if isinstance(prior, Mapping):
            if set(prior) != set(type_ids):
                raise ValueError("prior keys must exactly match sender types")
            prior_values = validate_probability_distribution(
                [prior[type_id] for type_id in type_ids], name="sender-type prior"
            )
        else:
            if len(prior) != len(type_ids):
                raise ValueError("prior length must equal the number of sender types")
            prior_values = validate_probability_distribution(
                prior, name="sender-type prior"
            )
        prior_mapping = frozen_mapping(dict(zip(type_ids, prior_values, strict=True)))
        expected_shape = (len(type_ids), len(message_ids), len(action_ids))

        def payoff_tensor(
            values: Mapping[tuple[str, str, str], float] | Any,
            *,
            name: str,
        ) -> np.ndarray:
            if values is None:
                raise TypeError(f"{name} is required")
            if isinstance(values, Mapping):
                expected_keys = set(
                    itertools.product(type_ids, message_ids, action_ids)
                )
                if set(values) != expected_keys:
                    missing = sorted(expected_keys - set(values))
                    extra = sorted(set(values) - expected_keys)
                    raise ValueError(
                        f"{name} mapping must cover every type/message/action; "
                        f"missing={missing}, extra={extra}"
                    )
                array = np.empty(expected_shape, dtype=float)
                for type_index, type_id in enumerate(type_ids):
                    for message_index, message_id in enumerate(message_ids):
                        for action_index, action_id in enumerate(action_ids):
                            array[type_index, message_index, action_index] = values[
                                type_id, message_id, action_id
                            ]
            else:
                array = np.array(values, dtype=float, copy=True)
            if array.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}; observed {array.shape}"
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain only finite values")
            array.setflags(write=False)
            return array

        sender_tensor = payoff_tensor(sender_payoffs, name="sender_payoffs")
        receiver_tensor = payoff_tensor(receiver_payoffs, name="receiver_payoffs")
        assumptions_tuple = tuple(assumptions)
        if any(
            not isinstance(assumption, str) or not assumption
            for assumption in assumptions_tuple
        ):
            raise ValueError("assumptions must contain only non-empty strings")
        object.__setattr__(self, "sender", sender_id)
        object.__setattr__(self, "receiver", receiver_id)
        object.__setattr__(self, "types", type_ids)
        object.__setattr__(self, "messages", message_ids)
        object.__setattr__(self, "actions", action_ids)
        object.__setattr__(self, "prior", prior_mapping)
        object.__setattr__(self, "sender_payoffs", sender_tensor)
        object.__setattr__(self, "receiver_payoffs", receiver_tensor)
        object.__setattr__(self, "assumptions", assumptions_tuple)

    @property
    def type_labels(self) -> tuple[str, ...]:
        return self.types

    @property
    def sender_types(self) -> tuple[str, ...]:
        return self.types

    @property
    def message_labels(self) -> tuple[str, ...]:
        return self.messages

    @property
    def receiver_action_labels(self) -> tuple[str, ...]:
        return self.actions

    @property
    def receiver_actions(self) -> tuple[str, ...]:
        return self.actions

    @property
    def sender_name(self) -> str:
        return self.sender

    @property
    def receiver_name(self) -> str:
        return self.receiver

    def sender_payoff(self, type_id: str, message_id: str, action_id: str) -> float:
        return float(
            self.sender_payoffs[
                self.types.index(type_id),
                self.messages.index(message_id),
                self.actions.index(action_id),
            ]
        )

    def receiver_payoff(self, type_id: str, message_id: str, action_id: str) -> float:
        return float(
            self.receiver_payoffs[
                self.types.index(type_id),
                self.messages.index(message_id),
                self.actions.index(action_id),
            ]
        )

    def pure_sender_strategies(self) -> tuple[dict[str, str], ...]:
        return tuple(
            dict(zip(self.types, choices, strict=True))
            for choices in itertools.product(self.messages, repeat=len(self.types))
        )

    def pure_receiver_strategies(self) -> tuple[dict[str, str], ...]:
        return tuple(
            dict(zip(self.messages, choices, strict=True))
            for choices in itertools.product(self.actions, repeat=len(self.messages))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "types": list(self.types),
            "messages": list(self.messages),
            "actions": list(self.actions),
            "prior": dict(self.prior),
            "sender_payoffs": self.sender_payoffs.tolist(),
            "receiver_payoffs": self.receiver_payoffs.tolist(),
            "assumptions": list(self.assumptions),
        }


@dataclass(frozen=True)
class PBECheck(SerializableMixin):
    is_pbe: bool
    sender_strategy: Mapping[str, str]
    receiver_strategy: Mapping[str, str]
    posteriors: Mapping[str, Mapping[str, float]]
    reach_probabilities: Mapping[str, float]
    sender_deviation_gains: Mapping[str, float]
    receiver_deviation_gains: Mapping[str, float]
    sender_best_deviations: Mapping[str, tuple[str, ...]]
    receiver_best_responses: Mapping[str, tuple[str, ...]]
    off_path_messages: tuple[str, ...]
    depends_on_off_path_beliefs: bool
    max_best_response_gain: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "sender_strategy", frozen_mapping(self.sender_strategy)
        )
        object.__setattr__(
            self, "receiver_strategy", frozen_mapping(self.receiver_strategy)
        )
        object.__setattr__(
            self,
            "posteriors",
            frozen_mapping(
                {
                    message: frozen_mapping(posterior)
                    for message, posterior in self.posteriors.items()
                }
            ),
        )
        object.__setattr__(
            self, "reach_probabilities", frozen_mapping(self.reach_probabilities)
        )
        object.__setattr__(
            self, "sender_deviation_gains", frozen_mapping(self.sender_deviation_gains)
        )
        object.__setattr__(
            self,
            "receiver_deviation_gains",
            frozen_mapping(self.receiver_deviation_gains),
        )
        object.__setattr__(
            self, "sender_best_deviations", frozen_mapping(self.sender_best_deviations)
        )
        object.__setattr__(
            self,
            "receiver_best_responses",
            frozen_mapping(self.receiver_best_responses),
        )


@dataclass(frozen=True)
class SignalingEquilibrium(SerializableMixin):
    sender_strategy: Mapping[str, str]
    receiver_strategy: Mapping[str, str]
    classification: str
    posteriors: Mapping[str, Mapping[str, float]]
    reach_probabilities: Mapping[str, float]
    expected_utilities: Mapping[str, float]
    sender_deviation_gains: Mapping[str, float]
    receiver_deviation_gains: Mapping[str, float]
    max_best_response_gain: float
    off_path_messages: tuple[str, ...]
    depends_on_off_path_beliefs: bool

    def __post_init__(self) -> None:
        if self.classification not in {"pooling", "separating", "partially_pooling"}:
            raise ValueError("unknown signaling-equilibrium classification")
        object.__setattr__(
            self, "sender_strategy", frozen_mapping(self.sender_strategy)
        )
        object.__setattr__(
            self, "receiver_strategy", frozen_mapping(self.receiver_strategy)
        )
        object.__setattr__(
            self,
            "posteriors",
            frozen_mapping(
                {
                    message: frozen_mapping(posterior)
                    for message, posterior in self.posteriors.items()
                }
            ),
        )
        object.__setattr__(
            self, "reach_probabilities", frozen_mapping(self.reach_probabilities)
        )
        object.__setattr__(
            self, "expected_utilities", frozen_mapping(self.expected_utilities)
        )
        object.__setattr__(
            self, "sender_deviation_gains", frozen_mapping(self.sender_deviation_gains)
        )
        object.__setattr__(
            self,
            "receiver_deviation_gains",
            frozen_mapping(self.receiver_deviation_gains),
        )
        object.__setattr__(
            self, "off_path_messages", tuple(sorted(self.off_path_messages))
        )

    @property
    def regime(self) -> str:
        return self.classification

    @property
    def beliefs(self) -> Mapping[str, Mapping[str, float]]:
        return self.posteriors

    @property
    def expected_payoffs(self) -> Mapping[str, float]:
        return self.expected_utilities


def _classification(sender_strategy: Mapping[str, str]) -> str:
    used = tuple(sender_strategy.values())
    distinct = len(set(used))
    if distinct == 1:
        return "pooling"
    if distinct == len(used):
        return "separating"
    return "partially_pooling"


def _belief_assignments(
    game: SignalingGame,
    off_path_beliefs: BeliefSpecification | str | None,
) -> tuple[tuple[dict[str, dict[str, float]], str], ...]:
    if off_path_beliefs is None:
        off_path_beliefs = "prior"
    if isinstance(off_path_beliefs, str):
        if off_path_beliefs == "prior":
            distribution = dict(game.prior)
        elif off_path_beliefs == "uniform":
            distribution = {type_id: 1.0 / len(game.types) for type_id in game.types}
        else:
            raise ValueError(
                "off-path convention must be 'prior', 'uniform', or a mapping"
            )
        return (
            (
                {message: dict(distribution) for message in game.messages},
                off_path_beliefs,
            ),
        )

    unknown_messages = set(off_path_beliefs) - set(game.messages)
    missing_messages = set(game.messages) - set(off_path_beliefs)
    if unknown_messages or missing_messages:
        raise ValueError(
            "explicit off-path beliefs must cover every message; "
            f"missing={sorted(missing_messages)}, extra={sorted(unknown_messages)}"
        )
    alternatives: list[tuple[dict[str, float], ...]] = []
    for message in game.messages:
        supplied = off_path_beliefs[message]
        candidates: tuple[Mapping[str, float], ...]
        if isinstance(supplied, Mapping):
            candidates = (supplied,)
        else:
            candidates = tuple(supplied)
            if not candidates:
                raise ValueError(
                    f"belief feasibility set for {message!r} cannot be empty"
                )
        validated_candidates: list[dict[str, float]] = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise TypeError(
                    "each off-path belief must be a mapping from types to probabilities"
                )
            if set(candidate) != set(game.types):
                raise ValueError(
                    f"off-path belief at {message!r} must exactly cover sender types"
                )
            parsed_distribution = ProbabilityDistribution(candidate)
            validated_candidates.append(
                {
                    type_id: parsed_distribution.probabilities[type_id]
                    for type_id in game.types
                }
            )
        alternatives.append(tuple(validated_candidates))
    return tuple(
        (
            {
                message: dict(combination[index])
                for index, message in enumerate(game.messages)
            },
            (
                "explicit per-message belief feasibility set"
                if any(len(options) > 1 for options in alternatives)
                else "explicit per-message beliefs"
            ),
        )
        for combination in itertools.product(*alternatives)
    )


def _posteriors(
    game: SignalingGame,
    sender_strategy: Mapping[str, str],
    off_path_assignment: Mapping[str, Mapping[str, float]],
) -> tuple[dict[str, float], dict[str, dict[str, float]], tuple[str, ...]]:
    reach = {
        message: sum(
            (
                game.prior[type_id]
                for type_id in game.types
                if sender_strategy[type_id] == message
            ),
            0.0,
        )
        for message in game.messages
    }
    posteriors: dict[str, dict[str, float]] = {}
    off_path: list[str] = []
    for message in game.messages:
        if reach[message] > 0.0:
            posterior = {
                type_id: (
                    game.prior[type_id] / reach[message]
                    if sender_strategy[type_id] == message
                    else 0.0
                )
                for type_id in game.types
            }
            validate_probability_distribution(
                posterior, name=f"posterior after {message!r}"
            )
            posteriors[message] = posterior
        else:
            off_path.append(message)
            posteriors[message] = dict(off_path_assignment[message])
    return reach, posteriors, tuple(off_path)


def evaluate_pure_pbe_candidate(
    game: SignalingGame,
    sender_strategy: Mapping[str, str],
    receiver_strategy: Mapping[str, str],
    *,
    off_path_beliefs: Mapping[str, Mapping[str, float]],
    tolerance: float = DEFAULT_TOLERANCE,
) -> PBECheck:
    """Evaluate sequential rationality and sender deviations for one assessment."""

    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if set(sender_strategy) != set(game.types):
        raise ValueError("sender strategy must map every type exactly")
    if any(message not in game.messages for message in sender_strategy.values()):
        raise ValueError("sender strategy references an unknown message")
    if set(receiver_strategy) != set(game.messages):
        raise ValueError("receiver strategy must map every message exactly")
    if any(action not in game.actions for action in receiver_strategy.values()):
        raise ValueError("receiver strategy references an unknown action")
    if set(off_path_beliefs) != set(game.messages):
        raise ValueError("off-path belief assignment must cover every message")
    for message in game.messages:
        if set(off_path_beliefs[message]) != set(game.types):
            raise ValueError("each off-path belief must cover every sender type")
        validate_probability_distribution(
            off_path_beliefs[message], name=f"off-path belief after {message!r}"
        )

    reach, posteriors, off_path_messages = _posteriors(
        game, sender_strategy, off_path_beliefs
    )
    receiver_gains: dict[str, float] = {}
    receiver_best_responses: dict[str, tuple[str, ...]] = {}
    depends_on_off_path = False
    for message in game.messages:
        posterior = posteriors[message]
        action_values = {
            action: sum(
                posterior[type_id] * game.receiver_payoff(type_id, message, action)
                for type_id in game.types
            )
            for action in game.actions
        }
        best_value = max(action_values.values())
        chosen_action = receiver_strategy[message]
        receiver_gains[message] = max(0.0, best_value - action_values[chosen_action])
        receiver_best_responses[message] = tuple(
            action
            for action in game.actions
            if action_values[action] >= best_value - tolerance
        )
        if message in off_path_messages:
            # The chosen action is optimal under every possible belief iff it
            # is weakly optimal for every degenerate type belief.
            for type_id in game.types:
                chosen_value = game.receiver_payoff(type_id, message, chosen_action)
                type_best = max(
                    game.receiver_payoff(type_id, message, action)
                    for action in game.actions
                )
                if chosen_value < type_best - tolerance:
                    depends_on_off_path = True
                    break

    sender_gains: dict[str, float] = {}
    sender_best_deviations: dict[str, tuple[str, ...]] = {}
    for type_id in game.types:
        chosen_message = sender_strategy[type_id]
        chosen_value = game.sender_payoff(
            type_id, chosen_message, receiver_strategy[chosen_message]
        )
        deviation_values = {
            message: game.sender_payoff(type_id, message, receiver_strategy[message])
            for message in game.messages
        }
        best_value = max(deviation_values.values())
        sender_gains[type_id] = max(0.0, best_value - chosen_value)
        sender_best_deviations[type_id] = tuple(
            message
            for message in game.messages
            if deviation_values[message] >= best_value - tolerance
        )

    all_gains = (*sender_gains.values(), *receiver_gains.values())
    max_gain = max(all_gains, default=0.0)
    return PBECheck(
        is_pbe=max_gain <= tolerance,
        sender_strategy=dict(sender_strategy),
        receiver_strategy=dict(receiver_strategy),
        posteriors=posteriors,
        reach_probabilities=reach,
        sender_deviation_gains=sender_gains,
        receiver_deviation_gains=receiver_gains,
        sender_best_deviations=sender_best_deviations,
        receiver_best_responses=receiver_best_responses,
        off_path_messages=off_path_messages,
        depends_on_off_path_beliefs=depends_on_off_path,
        max_best_response_gain=max_gain,
    )


def _expected_utilities(
    game: SignalingGame,
    sender_strategy: Mapping[str, str],
    receiver_strategy: Mapping[str, str],
) -> dict[str, float]:
    sender_total = 0.0
    receiver_total = 0.0
    for type_id in game.types:
        message = sender_strategy[type_id]
        action = receiver_strategy[message]
        probability = game.prior[type_id]
        sender_total += probability * game.sender_payoff(type_id, message, action)
        receiver_total += probability * game.receiver_payoff(type_id, message, action)
    return {game.sender: sender_total, game.receiver: receiver_total}


def enumerate_pure_pbe(
    game: SignalingGame,
    off_path_beliefs: BeliefSpecification | str | None = None,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> SolverResult:
    """Enumerate all pure-strategy PBE assessments of ``game`` exactly.

    ``off_path_beliefs`` may be ``"prior"`` (the explicit default),
    ``"uniform"``, a distribution for every message, or a finite sequence of
    admissible distributions for every message.  Bayes' rule always overrides
    the supplied value at reached messages.  The solver searches the restricted
    pure class only; a failure is therefore labeled ``NO_PURE_PBE_FOUND``.
    """

    started = time.perf_counter()
    if not isinstance(game, SignalingGame):
        raise TypeError("game must be a SignalingGame")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    assignments = _belief_assignments(game, off_path_beliefs)
    equilibria: list[SignalingEquilibrium] = []
    sequential_receiver_profiles_checked = 0
    for off_path_assignment, _ in assignments:
        for sender_strategy in game.pure_sender_strategies():
            # A PBE receiver plan must be sequentially rational message by
            # message.  Derive those action sets once for this sender map and
            # assessment, then enumerate only their Cartesian product.  This
            # is exactly equivalent to enumerating every A**M receiver plan
            # and rejecting the non-best responses, but avoids that dominant
            # cost for the four-type/four-message Cuba games.
            reach, posteriors, off_path_messages = _posteriors(
                game, sender_strategy, off_path_assignment
            )
            receiver_best_by_message: dict[str, tuple[str, ...]] = {}
            receiver_values_by_message: dict[str, dict[str, float]] = {}
            for message in game.messages:
                posterior = posteriors[message]
                action_values = {
                    action: sum(
                        posterior[type_id]
                        * game.receiver_payoff(type_id, message, action)
                        for type_id in game.types
                    )
                    for action in game.actions
                }
                best_value = max(action_values.values())
                receiver_values_by_message[message] = action_values
                receiver_best_by_message[message] = tuple(
                    action
                    for action in game.actions
                    if action_values[action] >= best_value - tolerance
                )

            receiver_choice_sets = tuple(
                receiver_best_by_message[message] for message in game.messages
            )
            for receiver_choices in itertools.product(*receiver_choice_sets):
                sequential_receiver_profiles_checked += 1
                receiver_strategy = dict(
                    zip(game.messages, receiver_choices, strict=True)
                )
                sender_gains: dict[str, float] = {}
                for type_id in game.types:
                    chosen_message = sender_strategy[type_id]
                    chosen_value = game.sender_payoff(
                        type_id,
                        chosen_message,
                        receiver_strategy[chosen_message],
                    )
                    best_value = max(
                        game.sender_payoff(type_id, message, receiver_strategy[message])
                        for message in game.messages
                    )
                    sender_gains[type_id] = max(0.0, best_value - chosen_value)
                if max(sender_gains.values(), default=0.0) > tolerance:
                    continue

                receiver_gains = {
                    message: max(
                        0.0,
                        max(receiver_values_by_message[message].values())
                        - receiver_values_by_message[message][
                            receiver_strategy[message]
                        ],
                    )
                    for message in game.messages
                }
                depends_on_off_path = False
                for message in off_path_messages:
                    chosen_action = receiver_strategy[message]
                    if any(
                        game.receiver_payoff(type_id, message, chosen_action)
                        < max(
                            game.receiver_payoff(type_id, message, action)
                            for action in game.actions
                        )
                        - tolerance
                        for type_id in game.types
                    ):
                        depends_on_off_path = True
                        break
                max_gain = max(
                    (*sender_gains.values(), *receiver_gains.values()),
                    default=0.0,
                )
                equilibria.append(
                    SignalingEquilibrium(
                        sender_strategy=sender_strategy,
                        receiver_strategy=receiver_strategy,
                        classification=_classification(sender_strategy),
                        posteriors=posteriors,
                        reach_probabilities=reach,
                        expected_utilities=_expected_utilities(
                            game, sender_strategy, receiver_strategy
                        ),
                        sender_deviation_gains=sender_gains,
                        receiver_deviation_gains=receiver_gains,
                        max_best_response_gain=max_gain,
                        off_path_messages=off_path_messages,
                        depends_on_off_path_beliefs=depends_on_off_path,
                    )
                )

    # Duplicate assessments can arise when a feasibility set varies beliefs at
    # messages that are reached and therefore overwritten by Bayes' rule.
    unique: dict[str, SignalingEquilibrium] = {}
    for equilibrium in equilibria:
        unique[equilibrium.to_json()] = equilibrium
    equilibria_tuple = tuple(unique[key] for key in sorted(unique))
    convention_names = sorted({convention for _, convention in assignments})
    convention = ", ".join(convention_names)
    runtime = time.perf_counter() - started
    max_gap = max(
        (equilibrium.max_best_response_gain for equilibrium in equilibria_tuple),
        default=None,
    )
    classifications = {
        classification: sum(
            equilibrium.classification == classification
            for equilibrium in equilibria_tuple
        )
        for classification in ("pooling", "separating", "partially_pooling")
    }
    return SolverResult(
        solver_name=SOLVER_NAME,
        solver_version=SOLVER_VERSION,
        equilibrium_concept="pure-strategy perfect Bayesian equilibrium",
        found=bool(equilibria_tuple),
        solutions=equilibria_tuple,
        status="SOLUTIONS_FOUND" if equilibria_tuple else NO_PURE_PBE_FOUND,
        exactness_status="EXACT_WITHIN_TOLERANCE",
        convergence_status="NOT_APPLICABLE",
        best_response_gap=max_gap,
        off_path_belief_convention=convention,
        runtime_seconds=runtime,
        warnings=(
            ()
            if equilibria_tuple
            else (
                "no equilibrium was found in the restricted pure-strategy class; mixed PBE were not searched",
            )
        ),
        assumptions=(
            *game.assumptions,
            "finite sender type, message, and receiver action spaces",
            "receiver observes the message but not the sender type",
            "Bayes' rule is enforced at every reached message",
        ),
        seed=None,
        metadata={
            "tolerance": tolerance,
            "candidate_sender_strategies": len(game.messages) ** len(game.types),
            "candidate_receiver_strategies": len(game.actions) ** len(game.messages),
            "sequential_receiver_profiles_checked": sequential_receiver_profiles_checked,
            "belief_assignment_count": len(assignments),
            "equilibrium_count": len(equilibria_tuple),
            "classification_counts": classifications,
            "off_path_dependent_count": sum(
                equilibrium.depends_on_off_path_beliefs
                for equilibrium in equilibria_tuple
            ),
        },
    )


pure_pbe = enumerate_pure_pbe

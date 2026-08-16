"""Validated finite extensive-form games with stable node identifiers."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TypeAlias, cast

from .beliefs import Belief, BeliefSystem, OffPathBeliefRequired, posterior_from_reach
from .probability import DEFAULT_TOLERANCE, validate_probability_distribution
from .types import Player, SerializableMixin, frozen_mapping, labels, validate_stable_id
from .utilities import ExpectedUtilities, TerminalUtilities


@dataclass(frozen=True)
class DecisionNode(SerializableMixin):
    id: str
    player_id: str
    information_set_id: str
    actions: Mapping[str, str]

    def __post_init__(self) -> None:
        validate_stable_id(self.id, field_name="decision-node id")
        validate_stable_id(self.player_id, field_name="acting-player id")
        validate_stable_id(self.information_set_id, field_name="information-set id")
        if not self.actions:
            raise ValueError("a decision node must have at least one action")
        converted: dict[str, str] = {}
        for action_id, child_id in self.actions.items():
            validate_stable_id(action_id, field_name="action id")
            validate_stable_id(child_id, field_name="child node id")
            converted[action_id] = child_id
        object.__setattr__(self, "actions", frozen_mapping(converted))

    @property
    def action_ids(self) -> tuple[str, ...]:
        return tuple(self.actions)


@dataclass(frozen=True)
class ChanceBranch(SerializableMixin):
    outcome_id: str
    probability: float
    child_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.outcome_id, field_name="chance-outcome id")
        validate_stable_id(self.child_id, field_name="chance child id")
        try:
            probability = float(self.probability)
        except (TypeError, ValueError) as error:
            raise TypeError("chance-branch probability must be numeric") from error
        if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
            raise ValueError(
                "chance-branch probability must be finite and lie in [0, 1]"
            )
        object.__setattr__(self, "probability", probability)


@dataclass(frozen=True)
class ChanceNode(SerializableMixin):
    id: str
    branches: tuple[ChanceBranch, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.id, field_name="chance-node id")
        branches = tuple(sorted(self.branches, key=lambda branch: branch.outcome_id))
        if not branches:
            raise ValueError("a chance node must have at least one branch")
        outcome_ids = [branch.outcome_id for branch in branches]
        if len(outcome_ids) != len(set(outcome_ids)):
            raise ValueError("chance outcomes must be unique within a node")
        validate_probability_distribution(
            [branch.probability for branch in branches], name=f"chance node {self.id!r}"
        )
        object.__setattr__(self, "branches", branches)

    @classmethod
    def from_mapping(
        cls,
        id: str,
        branches: Mapping[str, tuple[float, str]],
    ) -> ChanceNode:
        return cls(
            id=id,
            branches=tuple(
                ChanceBranch(outcome, probability, child)
                for outcome, (probability, child) in branches.items()
            ),
        )


@dataclass(frozen=True)
class TerminalNode(SerializableMixin):
    id: str
    utilities: TerminalUtilities | Mapping[str, float]
    outcome: str = ""

    def __post_init__(self) -> None:
        validate_stable_id(self.id, field_name="terminal-node id")
        utilities = (
            self.utilities
            if isinstance(self.utilities, TerminalUtilities)
            else TerminalUtilities(self.utilities)
        )
        object.__setattr__(self, "utilities", utilities)

    @property
    def terminal_utilities(self) -> TerminalUtilities:
        return cast(TerminalUtilities, self.utilities)


GameNode: TypeAlias = DecisionNode | ChanceNode | TerminalNode


@dataclass(frozen=True)
class InformationSet(SerializableMixin):
    id: str
    player_id: str
    node_ids: tuple[str, ...]
    action_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.id, field_name="information-set id")
        validate_stable_id(self.player_id, field_name="information-set player id")
        node_ids = tuple(sorted(self.node_ids))
        action_ids = tuple(sorted(self.action_ids))
        if not node_ids:
            raise ValueError("an information set must contain at least one node")
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("an information set cannot repeat a node")
        if not action_ids:
            raise ValueError("an information set must expose at least one action")
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("information-set actions must be unique")
        for node in node_ids:
            validate_stable_id(node, field_name="information-set node id")
        for action in action_ids:
            validate_stable_id(action, field_name="information-set action id")
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "action_ids", action_ids)


@dataclass(frozen=True)
class BehavioralStrategy(SerializableMixin):
    """Behavioral profile: one validated distribution per information set."""

    probabilities: Mapping[str, Mapping[str, float]]
    tolerance: float = DEFAULT_TOLERANCE

    def __post_init__(self) -> None:
        if not self.probabilities:
            raise ValueError(
                "a behavioral strategy must contain at least one information set"
            )
        converted: dict[str, Mapping[str, float]] = {}
        for information_set_id, distribution in self.probabilities.items():
            validate_stable_id(
                information_set_id, field_name="strategy information-set id"
            )
            if not distribution:
                raise ValueError(f"strategy at {information_set_id!r} cannot be empty")
            sorted_distribution = dict(sorted(distribution.items()))
            for action_id in sorted_distribution:
                validate_stable_id(action_id, field_name="strategy action id")
            values = validate_probability_distribution(
                list(sorted_distribution.values()),
                tolerance=self.tolerance,
                name=f"strategy at {information_set_id!r}",
            )
            converted[information_set_id] = frozen_mapping(
                {
                    action_id: values[index]
                    for index, action_id in enumerate(sorted_distribution)
                }
            )
        object.__setattr__(self, "probabilities", frozen_mapping(converted))

    def __getitem__(self, information_set_id: str) -> Mapping[str, float]:
        return self.probabilities[information_set_id]

    @classmethod
    def pure(cls, actions: Mapping[str, str]) -> BehavioralStrategy:
        return cls(
            {
                information_set_id: {action_id: 1.0}
                for information_set_id, action_id in actions.items()
            }
        )


@dataclass(frozen=True)
class ExtensiveFormGame(SerializableMixin):
    players: tuple[str | Player, ...]
    root_id: str
    nodes: Mapping[str, GameNode]
    information_sets: Mapping[str, InformationSet] | None = None

    def __post_init__(self) -> None:
        player_ids = labels(self.players)
        object.__setattr__(self, "players", player_ids)
        validate_stable_id(self.root_id, field_name="root node id")
        nodes = dict(sorted(self.nodes.items()))
        if not nodes:
            raise ValueError("an extensive-form game must contain at least one node")
        for key, node in nodes.items():
            validate_stable_id(key, field_name="node mapping key")
            if not isinstance(node, (DecisionNode, ChanceNode, TerminalNode)):
                raise TypeError(
                    f"node {key!r} has unsupported type {type(node).__name__}"
                )
            if key != node.id:
                raise ValueError(
                    f"node mapping key {key!r} does not match node id {node.id!r}"
                )
        if self.root_id not in nodes:
            raise ValueError("root_id must reference an existing node")

        inferred: dict[str, list[DecisionNode]] = defaultdict(list)
        parent_count = {node_id: 0 for node_id in nodes}
        for node in nodes.values():
            if isinstance(node, DecisionNode):
                if node.player_id not in player_ids:
                    raise ValueError(
                        f"decision node {node.id!r} references an unknown player"
                    )
                inferred[node.information_set_id].append(node)
                child_ids: Iterable[str] = tuple(node.actions.values())
            elif isinstance(node, ChanceNode):
                child_ids = tuple(branch.child_id for branch in node.branches)
            else:
                node.terminal_utilities.validate_players(player_ids)
                child_ids = ()
            for child_id in child_ids:
                if child_id not in nodes:
                    raise ValueError(
                        f"node {node.id!r} references missing child {child_id!r}"
                    )
                parent_count[child_id] += 1

        if parent_count[self.root_id] != 0:
            raise ValueError("the root node cannot have a parent")
        bad_parent_counts = {
            node_id: count
            for node_id, count in parent_count.items()
            if node_id != self.root_id and count != 1
        }
        if bad_parent_counts:
            raise ValueError(
                "nodes must form a tree and each non-root node must have exactly one parent; "
                f"observed {bad_parent_counts}"
            )

        color: dict[str, int] = {}

        def visit(node_id: str) -> None:
            if color.get(node_id) == 1:
                raise ValueError(f"cycle detected at node {node_id!r}")
            if color.get(node_id) == 2:
                return
            color[node_id] = 1
            node = nodes[node_id]
            if isinstance(node, DecisionNode):
                children: Iterable[str] = tuple(node.actions.values())
            elif isinstance(node, ChanceNode):
                children = tuple(branch.child_id for branch in node.branches)
            else:
                children = ()
            for child_id in children:
                visit(child_id)
            color[node_id] = 2

        visit(self.root_id)
        unreachable = sorted(set(nodes) - set(color))
        if unreachable:
            raise ValueError(
                f"all nodes must be reachable from root; unreachable={unreachable}"
            )

        inferred_sets = {
            set_id: InformationSet(
                id=set_id,
                player_id=members[0].player_id,
                node_ids=tuple(member.id for member in members),
                action_ids=tuple(members[0].actions),
            )
            for set_id, members in inferred.items()
        }
        for set_id, members in inferred.items():
            expected_player = members[0].player_id
            expected_actions = set(members[0].actions)
            for member in members[1:]:
                if member.player_id != expected_player:
                    raise ValueError(
                        f"information set {set_id!r} contains nodes for different players"
                    )
                if set(member.actions) != expected_actions:
                    raise ValueError(
                        f"information set {set_id!r} contains nodes with different actions"
                    )

        if self.information_sets is not None:
            supplied = dict(self.information_sets)
            if set(supplied) != set(inferred_sets):
                raise ValueError(
                    "supplied information sets do not exactly cover decision nodes"
                )
            for set_id, information_set in supplied.items():
                if set_id != information_set.id:
                    raise ValueError(
                        "information-set mapping key must equal its stable id"
                    )
                expected = inferred_sets[set_id]
                if information_set != expected:
                    raise ValueError(
                        f"supplied information set {set_id!r} is inconsistent with its nodes"
                    )
            inferred_sets = supplied

        object.__setattr__(self, "nodes", frozen_mapping(nodes))
        object.__setattr__(self, "information_sets", frozen_mapping(inferred_sets))

    @property
    def player_ids(self) -> tuple[str, ...]:
        return cast(tuple[str, ...], tuple(self.players))

    @property
    def information_set_map(self) -> Mapping[str, InformationSet]:
        if self.information_sets is None:
            raise RuntimeError("information sets were not initialized")
        return self.information_sets

    @property
    def is_perfect_information(self) -> bool:
        return all(
            len(information_set.node_ids) == 1
            for information_set in self.information_set_map.values()
        )

    def validate_strategy(self, strategy: BehavioralStrategy) -> None:
        expected_sets = set(self.information_set_map)
        observed_sets = set(strategy.probabilities)
        if observed_sets != expected_sets:
            raise ValueError(
                "strategy must cover every information set exactly; "
                f"missing={sorted(expected_sets - observed_sets)}, "
                f"extra={sorted(observed_sets - expected_sets)}"
            )
        for set_id, information_set in self.information_set_map.items():
            observed_actions = set(strategy[set_id])
            expected_actions = set(information_set.action_ids)
            if observed_actions != expected_actions:
                raise ValueError(
                    f"strategy actions at {set_id!r} do not match the game; "
                    f"missing={sorted(expected_actions - observed_actions)}, "
                    f"extra={sorted(observed_actions - expected_actions)}"
                )

    def realization_probabilities(
        self, strategy: BehavioralStrategy
    ) -> dict[str, float]:
        self.validate_strategy(strategy)
        reach: dict[str, float] = {}
        stack = [(self.root_id, 1.0)]
        while stack:
            node_id, probability = stack.pop()
            reach[node_id] = probability
            node = self.nodes[node_id]
            if isinstance(node, DecisionNode):
                distribution = strategy[node.information_set_id]
                for action_id, child_id in reversed(tuple(node.actions.items())):
                    stack.append((child_id, probability * distribution[action_id]))
            elif isinstance(node, ChanceNode):
                for branch in reversed(node.branches):
                    stack.append((branch.child_id, probability * branch.probability))
        return dict(sorted(reach.items()))

    def beliefs(
        self,
        strategy: BehavioralStrategy,
        *,
        off_path_beliefs: Mapping[str, Mapping[str, float]] | None = None,
    ) -> BeliefSystem:
        reach = self.realization_probabilities(strategy)
        off_path_beliefs = off_path_beliefs or {}
        beliefs: dict[str, Belief] = {}
        for set_id, information_set in self.information_set_map.items():
            node_reach = {
                node_id: reach[node_id] for node_id in information_set.node_ids
            }
            try:
                posterior = posterior_from_reach(node_reach)
                source = "Bayes' rule"
            except OffPathBeliefRequired:
                if set_id not in off_path_beliefs:
                    raise OffPathBeliefRequired(
                        f"off-path belief required for information set {set_id!r}"
                    ) from None
                supplied = off_path_beliefs[set_id]
                if set(supplied) != set(information_set.node_ids):
                    raise ValueError(
                        f"off-path belief for {set_id!r} must cover its nodes exactly"
                    ) from None
                validate_probability_distribution(
                    supplied, name=f"off-path belief {set_id!r}"
                )
                posterior = dict(supplied)
                source = "explicit off-path convention"
            beliefs[set_id] = Belief(set_id, posterior, source=source)
        return BeliefSystem(beliefs)

    def expected_utilities(self, strategy: BehavioralStrategy) -> ExpectedUtilities:
        reach = self.realization_probabilities(strategy)
        totals = {player: 0.0 for player in self.player_ids}
        terminal_mass = 0.0
        for node_id, node in self.nodes.items():
            if isinstance(node, TerminalNode):
                probability = reach[node_id]
                terminal_mass += probability
                for player in self.player_ids:
                    totals[player] += probability * node.terminal_utilities[player]
        if not math.isclose(terminal_mass, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError(
                f"terminal realization probability is {terminal_mass}, expected 1"
            )
        return ExpectedUtilities(totals)

    def terminal_ids(self) -> tuple[str, ...]:
        return tuple(
            node_id
            for node_id, node in self.nodes.items()
            if isinstance(node, TerminalNode)
        )

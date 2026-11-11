"""Learned play scored on an event's game tree (instant tree-family events).

The event's game is expanded into its tree at a scenario entry, and four joint policies
are compared: the learned networks, the engine's reference play, the true game's logit
QRE at the learner's temperature (the ``qre`` objective's target), and an exact
equilibrium (backward induction, or CFR+'s average). Scores are in goal units: the
game's payoffs divided by kappa, as the scenario's rewards.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from ..campaign.catalog.core import Catalog
from ..campaign.decisions.requests import MAX_OPTIONS
from ..campaign.engine.core import (
    QRE_TOLERANCE,
    CampaignEngine,
    EngineConfig,
    reference_tree_policy,
)
from ..campaign.environments.scenario import Entry, check_entries
from ..campaign.world.state import POWERS, identity
from ..families import build_family
from ..games import CHANCE, TERMINAL, GameState, build_tree
from ..games import outcome_distribution as tree_outcomes
from ..games.solvers import backward_induction, best_response, cfr_plus, logit_qre
from ..games.tree import (
    CHANCE_NODE,
    TERMINAL_NODE,
    GameTree,
    TreePolicy,
    expected_returns,
    node_values,
    reach_probabilities,
)
from ..records import Record
from .goalcheck import TREE_FAMILIES

TREMBLE = 0.01  # others' uniform tremble that defines beliefs off the played path


class OptionPolicy(Protocol):
    """Option probabilities for an identity at an observation (masked)."""

    def probabilities(self, ident: str, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Option probabilities for one observation of *ident*."""
        ...


class PolicyScore(Record):
    """One joint policy scored on the tree.

    **Exploitability** is the sum over agent roles of the gain from a best response while
    every other role keeps its policy; agents play the true game, so best responses use
    nature's probabilities, not a leader's mistaken belief. **Worst regret** (credibility) is
    each role's largest regret at any information set, conditional on reaching it, with
    beliefs formed under ``TREMBLE`` of uniform play, so every set is reached (the one-shot
    deviation principle). **Path regret** weights each role's regrets by the true-game QRE's
    reach: the regret it meets per episode on the QRE's paths.
    """

    def __init__(
        self,
        gains: dict[str, float],
        outcomes: dict[str, float],
        historical: float,
        regrets: dict[str, float] | None = None,
        qre_distance: float | None = None,
        path_regrets: dict[str, float] | None = None,
    ) -> None:
        self.gains = gains  # agent role -> gain from a best response (goal units)
        self.outcomes = outcomes
        self.historical = historical  # probability of the event's historical outcome
        self.regrets: dict[str, float] = (
            {} if regrets is None else regrets
        )  # role -> worst sequential regret
        self.qre_distance = qre_distance  # reach-weighted total variation from the true-game QRE
        self.path_regrets: dict[str, float] = (
            {} if path_regrets is None else path_regrets
        )  # role -> QRE-reach-weighted

    @property
    def exploitability(self) -> float:
        """The sum of the agent roles' best-response gains."""

        return sum(self.gains.values())

    @property
    def credibility(self) -> float:
        """The largest regret at any agent information set (0: sequentially rational)."""

        return max(self.regrets.values(), default=0.0)

    @property
    def path_regret(self) -> float:
        """The regrets met along the true-game QRE's paths, summed over agent roles."""

        return sum(self.path_regrets.values())


class TreeReport(Record):
    """The scores of an entry's four joint policies.

    **Distance** to the QRE is the mean total variation over agent information sets,
    weighted by the QRE's own reach. **Aliasing** counts information sets against distinct
    observations; **consistency** requires every history in a set to give one
    observation, and ``inconsistent`` counts the sets that break it. The QRE's fixed-point
    **residual** is flagged above ``QRE_TOLERANCE``.
    """

    def __init__(
        self,
        event_id: str,
        entry: str,
        nodes: int,
        agent_roles: tuple[str, ...],
        historical_label: str | None,
        scores: dict[str, PolicyScore],
        aliasing: dict[str, tuple[int, int]] | None = None,
        inconsistent: dict[str, int] | None = None,
        qre_residual: float = 0.0,
    ) -> None:
        self.event_id = event_id
        self.entry = entry
        self.nodes = nodes
        self.agent_roles = agent_roles
        self.historical_label = historical_label
        self.scores = scores  # "learned", "reference", "qre", "exact"
        self.aliasing: dict[str, tuple[int, int]] = (
            {} if aliasing is None else aliasing
        )  # role -> (sets, obs)
        self.inconsistent: dict[str, int] = (
            {} if inconsistent is None else inconsistent
        )  # role -> sets seen two ways
        self.qre_residual = qre_residual  # the QRE benchmark's fixed-point residual

    def text(self) -> list[str]:
        """The report, one block per policy."""

        out = [
            f"{self.event_id} from {self.entry}: {self.nodes} nodes; agent roles "
            f"{', '.join(self.agent_roles)}; historical outcome {self.historical_label}",
        ]
        flag = " (NOT CONVERGED)" if self.qre_residual > QRE_TOLERANCE else ""
        out.append(f"  QRE benchmark residual {self.qre_residual:.1e}{flag}")
        for name, score in self.scores.items():
            gains = ", ".join(f"{r} {g:.4f}" for r, g in score.gains.items())
            dist = ", ".join(
                f"{k} {v:.2f}" for k, v in sorted(score.outcomes.items(), key=lambda kv: -kv[1])
            )
            regrets = ", ".join(f"{r} {g:.4f}" for r, g in score.regrets.items())
            paths = ", ".join(f"{r} {g:.2e}" for r, g in score.path_regrets.items())
            out.append(
                f"  {name:<9} exploitability {score.exploitability:.4f} ({gains}); "
                f"P(historical) {score.historical:.2f}; {dist}"
            )
            distance = (
                "" if score.qre_distance is None else f"; distance to QRE {score.qre_distance:.3f}"
            )
            out.append(f"            worst regret at an information set: {regrets}{distance}")
            out.append(f"            regret along the QRE's paths: {paths}")
        if self.aliasing:
            text = ", ".join(f"{r} {s} sets/{o} obs" for r, (s, o) in self.aliasing.items())
            out.append(f"  aliasing: {text}")
        if self.inconsistent:
            text = ", ".join(f"{r} {n}" for r, n in self.inconsistent.items())
            out.append(f"  sets with more than one observation: {text}")
        return out


def tree_root(game: Any) -> GameState:
    """The root the tree is built from (alignment: its solver state)."""

    root: GameState = (
        game.solver_state() if hasattr(game, "solver_state") else game.new_initial_state()
    )
    return root


def decision_states(root: GameState, roles: Sequence[str]) -> Iterator[tuple[str, GameState]]:
    """Every decision state of *roles* with its role, by depth-first expansion."""

    stack = [root.clone()]
    while stack:
        state = stack.pop()
        player = state.current_player()
        if player == TERMINAL:
            continue
        if player == CHANCE:
            actions = [a for a, p in state.chance_outcomes() if p > 0.0]
        else:
            actions = list(state.legal_actions())
            if player in roles:
                yield player, state
        for action in actions:
            child = state.clone()
            child.apply_action(action)
            stack.append(child)


def representatives(root: GameState, roles: Sequence[str]) -> dict[Hashable, GameState]:
    """One state per information set of *roles*."""

    found: dict[Hashable, GameState] = {}
    for player, state in decision_states(root, roles):
        found.setdefault(state.information_state(player), state)
    return found


def inconsistent_sets(
    engine: CampaignEngine, event_id: str, root: GameState, roles: Sequence[str]
) -> dict[str, int]:
    """Per role, the information sets whose histories give it more than one observation."""

    seen: dict[tuple[str, Hashable], set[bytes]] = {}
    for player, state in decision_states(root, roles):
        obs = np.asarray(engine.observe_subgame(event_id, state, player), dtype=np.float32)
        seen.setdefault((player, state.information_state(player)), set()).add(obs.tobytes())
    counts = dict.fromkeys(roles, 0)
    for (role, _), observations in seen.items():
        counts[role] += len(observations) > 1
    return counts


def tree_report(
    catalog: Catalog,
    entry: Entry,
    policy: OptionPolicy | None,
    *,
    agents: Sequence[str] = POWERS,
    temperature: float | None = None,
    qre_temperature: float | None = None,
    iterations: int = 300,
    cfr_iterations: int = 2000,
) -> TreeReport:
    """Score *policy* (None: reference play only) on the entry's event tree. *temperature*
    sets reference play's; *qre_temperature* the true-game QRE's (default: the same)."""

    check_entries({entry.event_id: [entry]}, catalog)
    event_id = entry.event_id
    major = catalog.events[event_id].major
    assert major is not None
    if major.family not in TREE_FAMILIES:
        raise ValueError(f"{event_id} is a {major.family} event; tree scores need {TREE_FAMILIES}")
    config = EngineConfig(npc_policy="equilibrium", playable_tier=1, scope=event_id)
    if temperature is not None:
        config = config.replace(baseline_temperature=temperature)
    engine = CampaignEngine(catalog, config)
    engine.restore(entry.state)
    active = engine.s.active[event_id]
    game = build_family(major.family, active.config)
    root = tree_root(game)
    tree = build_tree(root)
    kappa = catalog.goals.kappa
    reference = reference_tree_policy(game, tree, iterations, engine.config.baseline_temperature)
    exact = (
        backward_induction(tree).policy
        if tree.is_perfect_information
        else cfr_plus(tree, iterations=cfr_iterations).average_policy
    )
    tau = engine.config.baseline_temperature if qre_temperature is None else qre_temperature
    solved = logit_qre(tree, temperature=tau)
    qre = solved.policy
    agent_roles = tuple(r for r in tree.roles if r in POWERS and r in agents)
    world = engine.world
    label = major.historical_outcome
    policies: dict[str, TreePolicy] = {"reference": reference, "qre": qre, "exact": exact}
    aliasing: dict[str, tuple[int, int]] = {}
    inconsistent: dict[str, int] = {}
    if policy is not None:
        inconsistent = inconsistent_sets(engine, event_id, root, agent_roles)
        states = representatives(root, agent_roles)
        learned = [list(p) for p in reference]
        seen: dict[str, set[bytes]] = {r: set() for r in agent_roles}
        counts: dict[str, int] = dict.fromkeys(agent_roles, 0)
        for index, info in enumerate(tree.infosets):
            role = tree.roles[info.role]
            if role not in agent_roles:
                continue
            obs = engine.observe_subgame(event_id, states[info.key], role)
            mask = np.zeros(MAX_OPTIONS, dtype=np.int8)
            mask[: len(info.actions)] = 1
            probs = np.asarray(policy.probabilities(identity(world, role), obs, mask))
            probs = probs[: len(info.actions)]
            total = float(probs.sum())
            learned[index] = [float(p) / total for p in probs] if total > 0 else learned[index]
            seen[role].add(np.asarray(obs, dtype=np.float32).tobytes())
            counts[role] += 1
        aliasing = {r: (counts[r], len(seen[r])) for r in agent_roles}
        policies = {"learned": learned, **policies}
    qre_reach = reach_probabilities(tree, qre)
    scores = {
        name: _score(tree, joint, agent_roles, kappa, label, qre_reach)
        for name, joint in policies.items()
    }
    for name, joint in policies.items():
        scores[name].qre_distance = qre_distance(tree, joint, qre, agent_roles)
    return TreeReport(
        event_id=event_id,
        entry=entry.label,
        nodes=len(tree.nodes),
        agent_roles=agent_roles,
        historical_label=label,
        scores=scores,
        aliasing=aliasing,
        inconsistent=inconsistent,
        qre_residual=solved.residual,
    )


def _score(
    tree: GameTree,
    joint: TreePolicy,
    roles: Sequence[str],
    kappa: float,
    label: str | None,
    qre_reach: Sequence[float],
) -> PolicyScore:
    """Score *joint*: gains, outcome shares, the historical outcome's probability and regrets."""

    on_policy = expected_returns(tree, joint)
    gains = {
        role: max(0.0, best_response(tree, joint, role).value - on_policy[role]) / kappa
        for role in roles
    }
    outcomes = tree_outcomes(tree, joint)
    by_set = information_set_regrets(tree, joint, roles, kappa)
    regrets = {role: max(sets.values(), default=0.0) for role, sets in by_set.items()}
    paths = {
        role: sum(
            regret * sum(qre_reach[n] for n in tree.infosets[index].nodes)
            for index, regret in sets.items()
        )
        for role, sets in by_set.items()
    }
    historical = outcomes.get(label, 0.0) if label else 0.0
    return PolicyScore(gains, outcomes, historical, regrets, path_regrets=paths)


def qre_distance(
    tree: GameTree, policy: TreePolicy, qre: TreePolicy, roles: Sequence[str]
) -> float:
    """Mean total variation between *policy* and *qre* over the information sets of
    *roles*, weighted by the QRE's reach (full support: every set is reached)."""

    reach = reach_probabilities(tree, qre)
    indices = {tree.role_index(r) for r in roles}
    total = weight_sum = 0.0
    for index, info in enumerate(tree.infosets):
        if info.role not in indices:
            continue
        weight = sum(reach[n] for n in info.nodes)
        if weight <= 0.0:
            continue
        gap = 0.5 * sum(abs(p - q) for p, q in zip(policy[index], qre[index], strict=True))
        total += weight * gap
        weight_sum += weight
    return total / weight_sum if weight_sum > 0.0 else 0.0


def sequential_regrets(
    tree: GameTree,
    policy: TreePolicy,
    roles: Sequence[str],
    kappa: float,
    tremble: float = TREMBLE,
) -> dict[str, float]:
    """Each role's largest one-shot-deviation regret at any of its information sets,
    with beliefs from chance and the others' trembled play (goal units)."""

    by_set = information_set_regrets(tree, policy, roles, kappa, tremble)
    return {role: max(sets.values(), default=0.0) for role, sets in by_set.items()}


def information_set_regrets(
    tree: GameTree,
    policy: TreePolicy,
    roles: Sequence[str],
    kappa: float,
    tremble: float = TREMBLE,
) -> dict[str, dict[int, float]]:
    """Each role's one-shot-deviation regret at each of its information sets (by index),
    with beliefs from chance and the others' trembled play (goal units)."""

    values = node_values(tree, policy)
    trembled = [[(1.0 - tremble) * p + tremble / len(probs) for p in probs] for probs in policy]
    result: dict[str, dict[int, float]] = {}
    for role in roles:
        r = tree.role_index(role)
        reach = [0.0] * len(tree.nodes)
        reach[0] = 1.0
        for idx, node in enumerate(tree.nodes):
            if node.kind == TERMINAL_NODE or reach[idx] == 0.0:
                continue
            if node.kind == CHANCE_NODE:
                weights: Sequence[float] = node.probs
            elif node.player == r:
                weights = [1.0] * len(node.children)
            else:
                weights = trembled[node.infoset]
            for weight, child in zip(weights, node.children, strict=True):
                reach[child] = reach[idx] * weight
        sets: dict[int, float] = {}
        for index, info in enumerate(tree.infosets):
            if info.role != r:
                continue
            total = sum(reach[n] for n in info.nodes)
            if total <= 0.0:
                continue
            q = [
                sum(reach[n] * values[tree.nodes[n].children[a]][r] for n in info.nodes) / total
                for a in range(len(info.actions))
            ]
            own = sum(p * value for p, value in zip(policy[index], q, strict=True))
            sets[index] = max(0.0, max(q) - own) / kappa
        result[role] = sets
    return result


def tree_entries(catalog: Catalog, entries: Mapping[str, Sequence[Entry]]) -> list[Entry]:
    """The entries of instant tree-family events, which :func:`tree_report` can score."""

    result: list[Entry] = []
    for event_id, items in sorted(entries.items()):
        major = catalog.events[event_id].major
        if major is not None and major.family in TREE_FAMILIES and major.cadence == "instant":
            result.extend(items)
    return result


__all__ = [
    "QRE_TOLERANCE",
    "TREE_FAMILIES",
    "OptionPolicy",
    "PolicyScore",
    "TreeReport",
    "decision_states",
    "inconsistent_sets",
    "information_set_regrets",
    "qre_distance",
    "representatives",
    "sequential_regrets",
    "tree_entries",
    "tree_report",
]

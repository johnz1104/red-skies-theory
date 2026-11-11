"""Patrons compete for clients by offering aid and security guarantees.

Offers are revealed together. Controlling patrons may block a client's move
to a rival; each client then chooses a patron or stays nonaligned. A client
may choose as a player or through a logit chance rule over its payoffs.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from typing import Any

from ..games.protocol import CHANCE, TERMINAL, ImmutableParams, LoggedState, Outcome, check_legal
from ..records import Record
from .common import ConfigError, Section, as_mapping, finite, identifier, nonnegative

NAME = "alignment"
LEVELS = ("none", "aid", "aid_guarantee")
FORBID = ("allow", "forbid")


def _by_level(s: Section, key: str, default: tuple[float, float, float]) -> dict[str, float]:
    """A number for each offer level under *key*."""

    with s.section(key, {}) as levels:
        return {
            level: levels.number(level, fallback)
            for level, fallback in zip(LEVELS, default, strict=True)
        }


class Patron(ImmutableParams, Record):
    """A patron: what its offers are worth to clients and what clients are worth to it.

    ``offer_value``   a client's utility from each offer level [0, 0.5, 1].
    ``offer_cost``    the patron's cost of a level, paid per client that aligns with it
                      [0, 0.2, 0.5].
    ``gain``          ``{client: x}``: its gain when the client aligns with it.
    ``loss``          ``{client: x}``: its loss when the client aligns with a rival.
    ``offer_return``  what a level is worth to it beyond the clients it wins, earned when
                      the first client takes the offer up [0].
    ``control``       ``{client: x}``: the utility a client loses by defying its order not
                      to align with a rival. A patron with control may ``forbid``.
    ``forbid_cost``   the patron's cost of forbidding.
    """

    def __init__(
        self,
        id: str,
        offer_value: dict[str, float],
        offer_cost: dict[str, float],
        gain: dict[str, float],
        loss: dict[str, float],
        control: dict[str, float],
        forbid_cost: float,
        offer_return: dict[str, float],
    ) -> None:
        self.id = id
        self.offer_value = offer_value  # level -> value to a client
        self.offer_cost = offer_cost  # level -> cost per aligned client
        self.gain = gain  # client -> gain if it aligns with this patron
        self.loss = loss  # client -> loss if it aligns with a rival
        self.control = control  # client -> its penalty for defying a forbid
        self.forbid_cost = forbid_cost
        self.offer_return = offer_return  # level -> return if the first client takes it

    @classmethod
    def from_config(cls, patron_id: str, data: Any, client_ids: Collection[str]) -> Patron:
        """Read a patron's offers, their costs and the value of each client's alignment."""

        with Section(f"alignment.patrons.{patron_id}", data) as s:
            gain = as_mapping(s.at("gain"), s.get("gain", {}))
            loss = as_mapping(s.at("loss"), s.get("loss", {}))
            control = as_mapping(s.at("control"), s.get("control", {}))
            if not set(gain) | set(loss) | set(control) <= set(client_ids):
                raise ConfigError(f"{s.name}: gain/loss/control refer to unknown clients")
            return cls(
                id=identifier(s.name, patron_id),
                offer_value=_by_level(s, "offer_value", (0.0, 0.5, 1.0)),
                offer_cost=_by_level(s, "offer_cost", (0.0, 0.2, 0.5)),
                gain={k: finite(f"{s.at('gain')}.{k}", v) for k, v in gain.items()},
                loss={k: nonnegative(f"{s.at('loss')}.{k}", v) for k, v in loss.items()},
                control={k: nonnegative(f"{s.at('control')}.{k}", v) for k, v in control.items()},
                forbid_cost=s.number("forbid_cost", 0.0),
                offer_return=_by_level(s, "offer_return", (0.0, 0.0, 0.0)),
            )


class Client(ImmutableParams, Record):
    """A client: a player, or a chance move with logit choice probabilities.

    ``mode``         ``player`` or ``chance`` (the default).
    ``affinity``     ``{patron: x}``: its utility from aligning with the patron, before
                     the patron's offer.
    ``nonaligned``   its utility from staying non-aligned [0].
    ``temperature``  a chance client aligns with probability proportional to
                     ``exp(utility / temperature)`` [0.5].
    """

    def __init__(
        self,
        id: str,
        mode: str,
        affinity: dict[str, float],
        nonaligned: float,
        temperature: float,
    ) -> None:
        self.id = id
        self.mode = mode
        self.affinity = affinity
        self.nonaligned = nonaligned
        self.temperature = temperature

    @classmethod
    def from_config(cls, client_id: str, data: Any) -> Client:
        """Read a client's affinities and whether a player or a chance draw chooses its alignment."""

        with Section(f"alignment.clients.{client_id}", data) as s:
            mode = s.get("mode", "chance")
            if mode not in ("player", "chance"):
                raise ConfigError(f"{s.at('mode')} must be 'player' or 'chance'")
            temperature = s.nonnegative("temperature", 0.5)
            if temperature <= 0.0:
                raise ConfigError(f"{s.at('temperature')} must be positive")
            affinity = as_mapping(s.at("affinity"), s.get("affinity", {}))
            return cls(
                id=identifier(s.name, client_id),
                mode=mode,
                affinity={k: finite(f"{s.at('affinity')}.{k}", v) for k, v in affinity.items()},
                nonaligned=s.number("nonaligned", 0.0),
                temperature=temperature,
            )


class AlignmentParams(ImmutableParams, Record):
    """The patrons, in the order they make offers, and the clients, in the order they choose."""

    def __init__(self, patrons: tuple[Patron, ...], clients: tuple[Client, ...]) -> None:
        self.patrons = patrons
        self.clients = clients
        self._patron = {p.id: p for p in patrons}

    def patron(self, patron_id: str) -> Patron:
        return self._patron[patron_id]

    @property
    def client_options(self) -> tuple[str, ...]:
        """A client's choices: align with one of the patrons, or with none."""

        return (*(f"aligned_{p.id}" for p in self.patrons), "nonaligned")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> AlignmentParams:
        """Read patrons and clients, checking that their roles and references agree."""

        with Section("alignment", config) as s:
            client_blocks = as_mapping(s.at("clients"), s.get("clients"))
            clients = [Client.from_config(cid, raw) for cid, raw in client_blocks.items()]
            client_ids = {c.id for c in clients}
            patron_blocks = as_mapping(s.at("patrons"), s.get("patrons"))
            patrons = [
                Patron.from_config(pid, raw, client_ids) for pid, raw in patron_blocks.items()
            ]
        if len(patrons) < 1 or not clients:
            raise ConfigError("alignment needs at least one patron and one client")
        patron_ids = {p.id for p in patrons}
        if patron_ids & client_ids:
            raise ConfigError("a role cannot be both patron and client")
        for client in clients:
            if not set(client.affinity) <= patron_ids:
                raise ConfigError(f"client {client.id} has affinity for unknown patrons")
        return cls(tuple(patrons), tuple(clients))


Draws = tuple[tuple[str, float], ...]  # (chance outcome, probability)


class AlignmentGame:
    """An alignment contest with fixed parameters and fresh states for play."""

    name = NAME

    def __init__(self, params: AlignmentParams) -> None:
        self.params = params

    @property
    def outcome_labels(self) -> tuple[str, ...]:
        return self.params.client_options

    @property
    def roles(self) -> tuple[str, ...]:
        """The patrons, then the player clients."""

        player_clients = tuple(c.id for c in self.params.clients if c.mode == "player")
        return (*(p.id for p in self.params.patrons), *player_clients)

    def new_initial_state(self) -> AlignmentState:
        return AlignmentState(self.params)

    def solver_state(self) -> AlignmentState:
        """A root for tree solvers with the trailing chance clients folded into
        expected returns.

        Chance clients that choose after every player decide nothing anyone reacts to, so
        their choices only add expected gains and losses; folding them keeps the tree
        small. Players' information states are unchanged, so a policy solved on this tree
        plays the full game.
        """

        return AlignmentState(self.params, fold_tail=True)


def _tail_start(params: AlignmentParams) -> int:
    """The index of the first client in the trailing run of chance clients (never 0)."""

    start = len(params.clients)
    while start > 1 and params.clients[start - 1].mode == "chance":
        start -= 1
    return start


class Phase:
    """A stage of the game: who moves, with what actions, and what follows."""

    def player(self, state: AlignmentState) -> str:
        raise NotImplementedError

    def legal_actions(self, state: AlignmentState) -> tuple[str, ...]:
        raise NotImplementedError

    def apply(self, state: AlignmentState, player: str, action: str) -> None:
        raise NotImplementedError


class Offers(Phase):
    """Patrons choose offer levels simultaneously: each sees only its own until all are in."""

    def player(self, state: AlignmentState) -> str:
        return state.params.patrons[state.patron_index].id

    def legal_actions(self, state: AlignmentState) -> tuple[str, ...]:
        return LEVELS

    def apply(self, state: AlignmentState, player: str, action: str) -> None:
        state.offers[player] = action
        state.log(f"offer:self={action}", visible_to=[player])
        state.patron_index += 1
        if state.patron_index == len(state.params.patrons):
            for patron_id, level in state.offers.items():
                state.log(f"offer:{patron_id}={level}")


class Forbids(Phase):
    """Each controlling patron, in turn and in public, allows or forbids defection."""

    def player(self, state: AlignmentState) -> str:
        return state.controllers[state.forbid_index].id

    def legal_actions(self, state: AlignmentState) -> tuple[str, ...]:
        return FORBID

    def apply(self, state: AlignmentState, player: str, action: str) -> None:
        state.forbids[player] = action == "forbid"
        state.log(f"{player}:{action}")
        state.forbid_index += 1


class Choices(Phase):
    """Each client, in turn and in public, aligns with a patron or with none."""

    def player(self, state: AlignmentState) -> str:
        client = state.client()
        return client.id if client.mode == "player" else CHANCE

    def legal_actions(self, state: AlignmentState) -> tuple[str, ...]:
        return state.params.client_options

    def apply(self, state: AlignmentState, player: str, action: str) -> None:
        client = state.client()
        state.choices[client.id] = action
        state.log(f"{client.id}={action}")
        state.client_index += 1
        if state.client_index >= state.stop:
            state.done = True


OFFERS, FORBIDS, CHOICES = Offers(), Forbids(), Choices()


class AlignmentState(LoggedState):
    """A history of the contest: the offers, the forbids and the clients' choices so far."""

    def __init__(self, params: AlignmentParams, *, fold_tail: bool = False) -> None:
        super().__init__(AlignmentGame(params).roles)
        self.params = params
        self.offers: dict[str, str] = {}  # patron -> offer level
        self.patron_index = 0
        self.controllers = tuple(p for p in params.patrons if p.control)
        self.forbid_index = 0
        self.forbids: dict[str, bool] = {}  # controlling patron -> whether it forbade
        self.choices: dict[str, str] = {}  # client -> its choice
        self.client_index = 0
        self.stop = _tail_start(params) if fold_tail else len(params.clients)
        self.done = False

    def _phase(self) -> Phase:
        if self.patron_index < len(self.params.patrons):
            return OFFERS
        if self.forbid_index < len(self.controllers):
            return FORBIDS
        return CHOICES

    def client(self) -> Client:
        """The client whose turn it is to choose."""

        return self.params.clients[self.client_index]

    def current_player(self) -> str:
        if self.done:
            return TERMINAL
        return self._phase().player(self)

    def is_terminal(self) -> bool:
        return self.done

    def legal_actions(self) -> tuple[str, ...]:
        if self.current_player() in (CHANCE, TERMINAL):
            return ()
        return self._phase().legal_actions(self)

    def client_utility(self, client: Client, choice: str) -> float:
        """*client*'s utility from *choice* given the revealed offers.

        Aligning with a patron is worth the client's affinity plus the value of the
        patron's offer, less each rival patron's control over it if that rival forbade.
        """

        if choice == "nonaligned":
            return client.nonaligned
        patron_id = choice.removeprefix("aligned_")
        level = self.offers.get(patron_id, "none")
        utility = (
            client.affinity.get(patron_id, 0.0) + self.params.patron(patron_id).offer_value[level]
        )
        for other in self.controllers:  # defying a rival patron's order
            if other.id != patron_id and self.forbids.get(other.id, False):
                utility -= other.control.get(client.id, 0.0)
        return utility

    def chance_outcomes(self) -> Draws:
        if self.current_player() != CHANCE:
            return ()
        return self._choice_probabilities(self.client())

    def _choice_probabilities(self, client: Client) -> Draws:
        """A chance client's logit choice: P(o) proportional to exp(u(o) / temperature)."""

        options = self.params.client_options
        utilities = [self.client_utility(client, o) / client.temperature for o in options]
        top = max(utilities)
        weights = [math.exp(u - top) for u in utilities]
        total = sum(weights)
        return tuple((o, w / total) for o, w in zip(options, weights, strict=True) if w > 0.0)

    def apply_action(self, action: str) -> None:
        player = self.current_player()
        if player == TERMINAL:
            raise ConfigError("cannot act after alignment is settled")
        legal = (
            tuple(a for a, _ in self.chance_outcomes())
            if player == CHANCE
            else self.legal_actions()
        )
        check_legal(action, legal)
        self._phase().apply(self, player, action)
        self._history.append(action)

    def returns(self) -> dict[str, float]:
        """Each patron's and each player client's payoff.

        A patron gains from each client that aligns with it, net of its offer's cost, and
        loses from each that aligns with a rival; folded chance clients enter in
        expectation. Forbidding costs it ``forbid_cost``, and it earns its offer's return
        if the first client took the offer. A player client gets its utility.
        """

        if not self.done:
            raise ConfigError("returns requested before alignment is settled")
        result: dict[str, float] = {}
        folded = [c for c in self.params.clients if c.id not in self.choices]
        odds = {c.id: dict(self._choice_probabilities(c)) for c in folded}
        for patron in self.params.patrons:
            level = self.offers[patron.id]
            total = 0.0
            for client in self.params.clients:
                if client.id in self.choices:
                    chances = {self.choices[client.id]: 1.0}
                else:
                    chances = odds[client.id]
                for choice, prob in chances.items():
                    if choice == f"aligned_{patron.id}":
                        gain = patron.gain.get(client.id, 0.0)
                        total += prob * (gain - patron.offer_cost[level])
                    elif choice != "nonaligned":
                        total -= prob * patron.loss.get(client.id, 0.0)
            if self.forbids.get(patron.id, False):
                total -= patron.forbid_cost
            if self.choices[self.params.clients[0].id] == f"aligned_{patron.id}":
                total += patron.offer_return[level]
            result[patron.id] = total
        for client in self.params.clients:
            if client.mode == "player":
                result[client.id] = self.client_utility(client, self.choices[client.id])
        return result

    def outcome(self) -> Outcome:
        """The first client's choice, tagged with every choice, offer and forbid.

        ``commit:<patron>`` is its offer level as a share of the highest.
        """

        if not self.done:
            raise ConfigError("outcome requested before alignment is settled")
        first = self.params.clients[0].id
        tags = [f"{client}:{choice}" for client, choice in self.choices.items()]
        tags += [f"offer:{patron}={level}" for patron, level in self.offers.items()]
        tags += [f"{patron}:forbade" for patron, forbade in self.forbids.items() if forbade]
        aligned = sum(1 for c in self.choices.values() if c != "nonaligned")
        metrics = {"aligned_clients": float(aligned)}
        for patron, forbade in self.forbids.items():
            metrics[f"forbade:{patron}"] = 1.0 if forbade else 0.0
        for patron, level in self.offers.items():
            metrics[f"commit:{patron}"] = LEVELS.index(level) / (len(LEVELS) - 1)
        return Outcome(self.choices[first], tuple(tags), metrics)

    def observation(self, role: str) -> tuple[float, ...]:
        """Nine features: patron or client, its own offer, the best rival offer once
        revealed, whether offers are revealed, the share of clients aligned with it, the
        share that have chosen (twice), and whether anyone forbade."""

        is_patron = any(p.id == role for p in self.params.patrons)
        revealed = self.patron_index >= len(self.params.patrons)
        own_level = LEVELS.index(self.offers[role]) / 2.0 if role in self.offers else 0.0
        best_other = 0.0
        if revealed:
            others = [LEVELS.index(v) for k, v in self.offers.items() if k != role]
            best_other = max(others, default=0) / 2.0
        mine = sum(1 for c in self.choices.values() if c == f"aligned_{role}")
        return (
            1.0 if is_patron else 0.0,
            0.0 if is_patron else 1.0,
            own_level,
            best_other,
            1.0 if revealed else 0.0,
            mine / len(self.params.clients),
            self.client_index / len(self.params.clients),
            len(self.choices) / len(self.params.clients),
            1.0 if any(self.forbids.values()) else 0.0,
        )


def build(config: Mapping[str, Any]) -> AlignmentGame:
    """The alignment game of *config* (``patrons`` and ``clients``; see :class:`Patron`
    and :class:`Client`)."""

    return AlignmentGame(AlignmentParams.from_config(config))


__all__ = [
    "FORBID",
    "LEVELS",
    "NAME",
    "AlignmentGame",
    "AlignmentParams",
    "AlignmentState",
    "Client",
    "Patron",
    "build",
]

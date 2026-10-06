"""Patron competition for clients' alignment.

Patrons secretly and simultaneously choose an offer level (``none``, ``aid``,
``aid_guarantee``); offers are then revealed. Each client in turn chooses to
align with one patron or stay non-aligned. A client is either a *player*
(a role that chooses after seeing the offers) or *chance* (its choice is a
logit over affinity plus offer value).

A patron pays its offer's cost only to a client that aligns with it, gains
from clients that align with it and loses from clients that align with a
rival. Outcome label: the first client's choice (``aligned_<patron>`` or
``nonaligned``); every client's choice is also a tag ``<client>:<choice>``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..games.protocol import (
    CHANCE,
    TERMINAL,
    ImmutableParams,
    LoggedState,
    Outcome,
    check_legal,
)
from .common import (
    ConfigError,
    as_mapping,
    finite,
    identifier,
    nonnegative,
    reject_unknown,
    require,
)

NAME = "alignment"
LEVELS = ("none", "aid", "aid_guarantee")


@dataclass(frozen=True, slots=True)
class Patron(ImmutableParams):
    id: str
    offer_value: tuple[float, float, float]
    offer_cost: tuple[float, float, float]
    gain: tuple[tuple[str, float], ...]  # per client
    loss: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class Client(ImmutableParams):
    id: str
    mode: str  # player | chance
    affinity: tuple[tuple[str, float], ...]
    nonaligned: float
    temperature: float


@dataclass(frozen=True, slots=True)
class AlignmentParams(ImmutableParams):
    patrons: tuple[Patron, ...]
    clients: tuple[Client, ...]


def _lookup(pairs: tuple[tuple[str, float], ...], key: str, default: float = 0.0) -> float:
    for k, v in pairs:
        if k == key:
            return v
    return default


class AlignmentGame:
    name = NAME

    def __init__(self, params: AlignmentParams) -> None:
        self.params = params

    @property
    def outcome_labels(self) -> tuple[str, ...]:
        return (*(f"aligned_{p.id}" for p in self.params.patrons), "nonaligned")

    @property
    def roles(self) -> tuple[str, ...]:
        player_clients = tuple(c.id for c in self.params.clients if c.mode == "player")
        return (*(p.id for p in self.params.patrons), *player_clients)

    def new_initial_state(self) -> AlignmentState:
        return AlignmentState(self.params)


class AlignmentState(LoggedState):
    def __init__(self, params: AlignmentParams) -> None:
        game = AlignmentGame(params)
        super().__init__(game.roles)
        self.params = params
        self.offers: dict[str, str] = {}
        self.patron_index = 0
        self.client_index = 0
        self.choices: dict[str, str] = {}
        self.done = False

    def _client(self) -> Client:
        return self.params.clients[self.client_index]

    def current_player(self) -> str:
        if self.done:
            return TERMINAL
        if self.patron_index < len(self.params.patrons):
            return self.params.patrons[self.patron_index].id
        client = self._client()
        return client.id if client.mode == "player" else CHANCE

    def is_terminal(self) -> bool:
        return self.done

    def _client_options(self) -> tuple[str, ...]:
        return (*(f"aligned_{p.id}" for p in self.params.patrons), "nonaligned")

    def legal_actions(self) -> tuple[str, ...]:
        player = self.current_player()
        if player in (CHANCE, TERMINAL):
            return ()
        if self.patron_index < len(self.params.patrons):
            return LEVELS
        return self._client_options()

    def client_utility(self, client: Client, choice: str) -> float:
        if choice == "nonaligned":
            return client.nonaligned
        patron_id = choice.removeprefix("aligned_")
        patron = next(p for p in self.params.patrons if p.id == patron_id)
        level = LEVELS.index(self.offers.get(patron_id, "none"))
        return _lookup(client.affinity, patron_id) + patron.offer_value[level]

    def chance_outcomes(self) -> tuple[tuple[str, float], ...]:
        if self.current_player() != CHANCE:
            return ()
        client = self._client()
        options = self._client_options()
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
        if self.patron_index < len(self.params.patrons):
            self.offers[player] = action
            self.log(f"offer:self={action}", visible_to=[player])
            self.patron_index += 1
            if self.patron_index == len(self.params.patrons):
                for patron_id, level in self.offers.items():
                    self.log(f"offer:{patron_id}={level}")
        else:
            client = self._client()
            self.choices[client.id] = action
            self.log(f"{client.id}={action}")
            self.client_index += 1
            if self.client_index == len(self.params.clients):
                self.done = True
        self._history.append(action)

    def returns(self) -> dict[str, float]:
        if not self.done:
            raise ConfigError("returns requested before alignment is settled")
        result: dict[str, float] = {}
        for patron in self.params.patrons:
            level = LEVELS.index(self.offers[patron.id])
            total = 0.0
            for client in self.params.clients:
                choice = self.choices[client.id]
                if choice == f"aligned_{patron.id}":
                    total += _lookup(patron.gain, client.id) - patron.offer_cost[level]
                elif choice != "nonaligned":
                    total -= _lookup(patron.loss, client.id)
            result[patron.id] = total
        for client in self.params.clients:
            if client.mode == "player":
                result[client.id] = self.client_utility(client, self.choices[client.id])
        return result

    def outcome(self) -> Outcome:
        if not self.done:
            raise ConfigError("outcome requested before alignment is settled")
        first = self.params.clients[0].id
        tags = [f"{client}:{choice}" for client, choice in self.choices.items()]
        tags += [f"offer:{patron}={level}" for patron, level in self.offers.items()]
        aligned = sum(1 for c in self.choices.values() if c != "nonaligned")
        return Outcome(self.choices[first], tuple(tags), {"aligned_clients": float(aligned)})

    def observation(self, role: str) -> tuple[float, ...]:
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
        )


def _triple(
    name: str, data: Mapping[str, Any], default: tuple[float, float, float]
) -> tuple[float, float, float]:
    reject_unknown(name, data, LEVELS)
    return (
        finite(f"{name}.none", data.get("none", default[0])),
        finite(f"{name}.aid", data.get("aid", default[1])),
        finite(f"{name}.aid_guarantee", data.get("aid_guarantee", default[2])),
    )


def build(config: Mapping[str, Any]) -> AlignmentGame:
    reject_unknown("alignment", config, ("patrons", "clients"))
    clients = []
    for client_id, raw in as_mapping(
        "alignment.clients", require("alignment", config, "clients")
    ).items():
        name = f"alignment.clients.{client_id}"
        item = as_mapping(name, raw)
        reject_unknown(name, item, ("mode", "affinity", "nonaligned", "temperature"))
        mode = item.get("mode", "chance")
        if mode not in ("player", "chance"):
            raise ConfigError(f"{name}.mode must be 'player' or 'chance'")
        temperature = nonnegative(f"{name}.temperature", item.get("temperature", 0.5))
        if temperature <= 0.0:
            raise ConfigError(f"{name}.temperature must be positive")
        affinity = as_mapping(f"{name}.affinity", item.get("affinity", {}))
        clients.append(
            Client(
                id=identifier(name, client_id),
                mode=mode,
                affinity=tuple((k, finite(f"{name}.affinity.{k}", v)) for k, v in affinity.items()),
                nonaligned=finite(f"{name}.nonaligned", item.get("nonaligned", 0.0)),
                temperature=temperature,
            )
        )
    client_ids = {c.id for c in clients}
    patrons = []
    for patron_id, raw in as_mapping(
        "alignment.patrons", require("alignment", config, "patrons")
    ).items():
        name = f"alignment.patrons.{patron_id}"
        item = as_mapping(name, raw)
        reject_unknown(name, item, ("offer_value", "offer_cost", "gain", "loss"))
        gain = as_mapping(f"{name}.gain", item.get("gain", {}))
        loss = as_mapping(f"{name}.loss", item.get("loss", {}))
        if not set(gain) <= client_ids or not set(loss) <= client_ids:
            raise ConfigError(f"{name}: gain/loss refer to unknown clients")
        patrons.append(
            Patron(
                id=identifier(name, patron_id),
                offer_value=_triple(
                    f"{name}.offer_value",
                    as_mapping(name, item.get("offer_value", {})),
                    (0.0, 0.5, 1.0),
                ),
                offer_cost=_triple(
                    f"{name}.offer_cost",
                    as_mapping(name, item.get("offer_cost", {})),
                    (0.0, 0.2, 0.5),
                ),
                gain=tuple((k, finite(f"{name}.gain.{k}", v)) for k, v in gain.items()),
                loss=tuple((k, nonnegative(f"{name}.loss.{k}", v)) for k, v in loss.items()),
            )
        )
    if len(patrons) < 1 or not clients:
        raise ConfigError("alignment needs at least one patron and one client")
    patron_ids = {p.id for p in patrons}
    if patron_ids & client_ids:
        raise ConfigError("a role cannot be both patron and client")
    for client in clients:
        if not {k for k, _ in client.affinity} <= patron_ids:
            raise ConfigError(f"client {client.id} has affinity for unknown patrons")
    return AlignmentGame(AlignmentParams(tuple(patrons), tuple(clients)))

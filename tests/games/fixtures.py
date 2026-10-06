"""Textbook games implemented on the kernel protocol, used to validate algorithms."""

from __future__ import annotations

from cold_war_sim.games import CHANCE, TERMINAL, LoggedState, Outcome
from cold_war_sim.games.protocol import check_legal

KUHN_ROLES = ("p0", "p1")


class KuhnState(LoggedState):
    """Kuhn poker: three cards, one ante, one bet. Player 0's value is -1/18."""

    def __init__(self) -> None:
        super().__init__(KUHN_ROLES)
        self.cards: list[str] = []
        self.moves: list[str] = []

    def current_player(self) -> str:
        if len(self.cards) < 2:
            return CHANCE
        if self.is_terminal():
            return TERMINAL
        return KUHN_ROLES[len(self.moves) % 2]

    def chance_outcomes(self) -> tuple[tuple[str, float], ...]:
        remaining = [card for card in "JQK" if card not in self.cards]
        return tuple((card, 1.0 / len(remaining)) for card in remaining)

    def legal_actions(self) -> tuple[str, ...]:
        if self.current_player() in (CHANCE, TERMINAL):
            return ()
        return ("p", "b")

    def apply_action(self, action: str) -> None:
        if len(self.cards) < 2:
            check_legal(action, tuple(a for a, _ in self.chance_outcomes()))
            self.cards.append(action)
            self.log(f"card:{action}", visible_to=[KUHN_ROLES[len(self.cards) - 1]])
        else:
            player = self.current_player()
            check_legal(action, self.legal_actions())
            self.moves.append(action)
            self.log(f"{player}:{action}")
        self._history.append(action)

    def is_terminal(self) -> bool:
        return "".join(self.moves) in {"pp", "bp", "bb", "pbp", "pbb"}

    def returns(self) -> dict[str, float]:
        moves = "".join(self.moves)
        rank = {"J": 0, "Q": 1, "K": 2}
        p0_high = rank[self.cards[0]] > rank[self.cards[1]]
        if moves == "bp":
            p0 = 1.0
        elif moves == "pbp":
            p0 = -1.0
        elif moves == "pp":
            p0 = 1.0 if p0_high else -1.0
        else:
            p0 = 2.0 if p0_high else -2.0
        return {"p0": p0, "p1": -p0}

    def observation(self, role: str) -> tuple[float, ...]:
        return ()

    def outcome(self) -> Outcome:
        return Outcome("fold" if self.moves[-1] == "p" and "b" in self.moves else "showdown")


class MatchingPenniesState(LoggedState):
    """Simultaneous matching pennies, serialized: p1 does not see p0's coin."""

    def __init__(self) -> None:
        super().__init__(KUHN_ROLES)
        self.coins: list[str] = []

    def current_player(self) -> str:
        return TERMINAL if len(self.coins) == 2 else KUHN_ROLES[len(self.coins)]

    def chance_outcomes(self) -> tuple[tuple[str, float], ...]:
        return ()

    def legal_actions(self) -> tuple[str, ...]:
        return () if len(self.coins) == 2 else ("H", "T")

    def apply_action(self, action: str) -> None:
        player = self.current_player()
        check_legal(action, self.legal_actions())
        self.coins.append(action)
        self.log(f"{player}:{action}", visible_to=[player])
        self._history.append(action)

    def is_terminal(self) -> bool:
        return len(self.coins) == 2

    def returns(self) -> dict[str, float]:
        p0 = 1.0 if self.coins[0] == self.coins[1] else -1.0
        return {"p0": p0, "p1": -p0}

    def observation(self, role: str) -> tuple[float, ...]:
        return ()

    def outcome(self) -> Outcome:
        return Outcome("match" if self.coins[0] == self.coins[1] else "mismatch")


class UltimatumState(LoggedState):
    """Proposer offers 0..3 of 3 units; responder accepts or rejects."""

    def __init__(self) -> None:
        super().__init__(("proposer", "responder"))
        self.offer: int | None = None
        self.accepted: bool | None = None

    def current_player(self) -> str:
        if self.offer is None:
            return "proposer"
        if self.accepted is None:
            return "responder"
        return TERMINAL

    def chance_outcomes(self) -> tuple[tuple[str, float], ...]:
        return ()

    def legal_actions(self) -> tuple[str, ...]:
        if self.offer is None:
            return ("offer0", "offer1", "offer2", "offer3")
        if self.accepted is None:
            return ("accept", "reject")
        return ()

    def apply_action(self, action: str) -> None:
        check_legal(action, self.legal_actions())
        if self.offer is None:
            self.offer = int(action[-1])
        else:
            self.accepted = action == "accept"
        self.log(action)
        self._history.append(action)

    def is_terminal(self) -> bool:
        return self.accepted is not None

    def returns(self) -> dict[str, float]:
        assert self.offer is not None
        if not self.accepted:
            return {"proposer": 0.0, "responder": 0.0}
        return {"proposer": float(3 - self.offer), "responder": float(self.offer)}

    def observation(self, role: str) -> tuple[float, ...]:
        return ()

    def outcome(self) -> Outcome:
        return Outcome("agreement" if self.accepted else "breakdown")

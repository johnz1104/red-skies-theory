"""Reference play for attrition wars that bargain while fighting.

:class:`AttritionBeliefPolicy` is an approximate, belief-based logit policy for
wars with ``approval``, ``offers``, ``types``, ``mobility`` or ``red_lines``. It
is not an equilibrium; its exploitability is measured by NashConv on small
trees. For plain wars it delegates to :class:`AttritionLookaheadPolicy`.

* **Beliefs, one per side.** Each side holds a joint distribution over every
  type holder (the sides and any sponsors with levels of their own; two kinds or
  more levels each, so that a side holding out shifts beliefs a little at a
  time). A side's prior uses nature's distribution for the holders it sees (its
  own) and their ``belief`` for the rest, which may be mistaken. A sponsor with a
  ``private`` type has a view of its own: its side's, plus its own type, which
  its side and the other side hold at its ``belief``. An ``unseen`` type is known
  to no one: its holder's side starts from its ``own_belief`` and the other side
  from its ``belief``, and both learn it from the battles. Each joint is
  updated by Bayes on every public action, using this policy's probabilities
  for each type of the acting side evaluated under that same joint, and on
  every battle result (type-dependent odds and holds), so others learn a
  private type from what its holder does. Effort choices count once the round
  reveals them. Logit play keeps every action possible, so the
  update is always defined; at temperature 0 a likelihood floor stands in. A
  sponsor's belief about the other side is its side's joint conditioned on its
  own side's types. **Projection:** each side models the others as seeing the
  world as it does (it does not know that the other side's priors differ), so
  a mistaken belief can survive until battle or a revealing choice corrects it.
* **Values.** Without offers, the lookahead's exact random walk of control to
  the end of the war (with holds and type-dependent costs), averaged over the
  other side's types. With offers, one round of fighting and then, at the new
  line of contact, a settlement at the nearest offer on the grid if every party
  with a say (the two leads and the consent roles) prefers it to fighting on;
  otherwise the walk from there. Other sponsors are assumed to repeat their
  last efforts or, in the first round, to play their best response to everyone
  else at low effort. A direct effort that has not yet deployed counts at the
  ``deployment`` share in the first round only. A side's lead values withdrawing
  as its side's defeat; another sponsor values it as the war going on without its
  effort or costs, while it still collects the outcome (it may free-ride on the
  lead), less its ``withdraw_cost``. Others are assumed to stay unless quitting
  beats fighting even as their side's defeat. A sponsor's ``proxy_gain`` counts against its cost in each round
  of fighting in which another sponsor on its side is assumed direct. Values ignore
  learning.
* **Offers.** The proposer weighs each offer's settlement value by the chance
  that the responder, and then each consent role, accepts. Each accepts by
  logit on the settlement against its own value of carrying on, computed with
  the belief from before the offer.
* **Approval.** The value of the war at the start against the value of refusing.
* **Warnings.** Honest by construction: a type warns by logit on its gain from
  going direct once the line is crossed, so warnings carry information and
  bluffs happen only by noise. A warner that would be bluffing counts the
  audience cost.
* **Commitments.** A settlement's value to a committed lead is net of its
  audience cost when the terms fall short. The settlement expected next round
  is the offer nearest the line that meets every commitment made (or the
  nearest offer if none does). A lead commits by logit on its value of
  carrying on with the commitment made against without it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

from ..games.protocol import CHANCE, GameState
from .attrition import (
    REFUSED,
    AttritionGame,
    AttritionLookaheadPolicy,
    AttritionState,
    SideTypes,
    Sponsor,
    other_side,
)
from .common import ConfigError

FLOOR = 0.02  # likelihood floor for exact (temperature 0) play
CACHE_LIMIT = 400000
KEY_CACHE_LIMIT = 20000

Cell = tuple[str, ...]  # one kind per type holder, in ``params.holders`` order
Joint = dict[Cell, float]
Own = tuple[tuple[str, str], ...]  # (holder, kind) pairs a side knows or assumes
Deferred = tuple[tuple[str, dict[str, dict[Own, float]]], ...]  # (side, {view: likes})
Entry = tuple[AttritionState, dict[str, Joint], Deferred]


def choose(
    scores: Mapping[str, float], legal: tuple[str, ...], temperature: float
) -> dict[str, float]:
    """Logit choice at *temperature*, or the best action (first on ties) at 0."""

    if temperature > 0.0:
        top = max(scores[a] for a in legal)
        weights = {a: math.exp((scores[a] - top) / temperature) for a in legal}
        total = sum(weights.values())
        return {a: w / total for a, w in weights.items()}
    best = max(legal, key=lambda a: (scores[a], -legal.index(a)))
    return {a: (1.0 if a == best else 0.0) for a in legal}


class AttritionBeliefPolicy:
    """Approximate reference play for attrition wars with offers or hidden types."""

    def __init__(self, game: AttritionGame, temperature: float = 0.0) -> None:
        if temperature < 0.0:
            raise ConfigError("temperature must be non-negative")
        self.game = game
        self.params = game.params
        self.temperature = temperature
        self._plain = None if game.params.extended else AttritionLookaheadPolicy(game, temperature)
        self._floor = FLOOR if temperature == 0.0 else 1e-9
        self._tracked: dict[tuple[str, ...], Entry] = {}
        self._memo: dict[Any, Any] = {}
        self._keys: dict[int, tuple[AttritionState, int, tuple[Any, ...]]] = {}
        self._holders = game.params.holders
        self._strength_holders = tuple(
            h for h in self._holders if len(set(self._types(h).strength)) > 1
        )
        private = tuple(h for h in self._holders if self._types(h).private)
        hidden = tuple(h for h in self._holders if self._types(h).unseen)  # seen by no one
        # the holders each role sees: its side's public ones, and its own private type
        self._knows: dict[str, tuple[str, ...]] = {
            s.id: tuple(
                h
                for h in self._holders
                if h not in hidden
                and ((self.params.holder_side(h) == s.side and h not in private) or h == s.id)
            )
            for s in self.params.sponsors
        }
        for side in ("a", "b"):
            self._knows[side] = tuple(
                h
                for h in self._holders
                if self.params.holder_side(h) == side and h not in private and h not in hidden
            )
        self._views = ("a", "b", *private)

    # ------------------------------------------------------------- interface
    def action_probabilities(self, state: GameState) -> dict[str, float]:
        if self._plain is not None:
            return self._plain.action_probabilities(state)
        if not isinstance(state, AttritionState):
            raise TypeError("AttritionBeliefPolicy needs an AttritionState")
        role = state.current_player()
        return self._probabilities(
            state, role, self._own(state, role), self.joint(state, self._view_of(role))
        )

    def joint(self, state: AttritionState, view: str = "a") -> Joint:
        """The belief over the type holders' kinds at *state* held by side *view*
        (``a`` or ``b``), or by a sponsor with a private type (its role)."""

        history = state.history()
        entry = self._tracked.get(history)
        if entry is None:
            entry = self._track(state, history)
        return entry[1][view]

    def _view_of(self, role: str) -> str:
        return role if role in self._views else self.params.sponsor(role).side

    def public_belief(self, state: AttritionState) -> dict[str, float]:
        """For each type holder, P(strongest kind) as the other side believes it."""

        result: dict[str, float] = {}
        for index, holder in enumerate(self._holders):
            holder_types = self.params.types_of(holder)
            assert holder_types is not None
            view = other_side(self.params.holder_side(holder))
            strongest = holder_types.kinds[0]
            result[holder] = sum(
                q for cell, q in self.joint(state, view).items() if cell[index] == strongest
            )
        return result

    def _types(self, holder: str) -> SideTypes:
        holder_types = self.params.types_of(holder)
        assert holder_types is not None
        return holder_types

    def _own(self, state: AttritionState, role: str) -> Own:
        """The kinds a role knows: those of the holders it sees, drawn so far."""

        return tuple((h, state.types[h]) for h in self._knows[role] if h in state.types)

    def _kinds(self, cell: Cell) -> dict[str, str]:
        return dict(zip(self._holders, cell, strict=True))

    # --------------------------------------------------------------- beliefs
    def _prior(self, view: str) -> Joint:
        """View *view*'s prior: nature's for the holders it sees, the holder side's own
        belief for an unseen type of its own side, and belief for the rest."""

        side = view if view in ("a", "b") else self.params.sponsor(view).side
        joint: Joint = {(): 1.0}
        for holder in self._holders:
            holder_types = self.params.types_of(holder)
            assert holder_types is not None
            if holder in self._knows[view]:
                dist = holder_types.distribution()
            elif holder_types.unseen and self.params.holder_side(holder) == side:
                dist = holder_types.own_believed()
            else:
                dist = holder_types.believed()
            joint = {(*cell, k): q * w for cell, q in joint.items() for k, w in dist.items()}
        return joint

    def _track(self, state: AttritionState, history: tuple[str, ...]) -> Entry:
        if len(self._tracked) > CACHE_LIMIT:
            self._tracked.clear()
        known = len(history)
        while known > 0 and history[:known] not in self._tracked:
            known -= 1
        if known == 0 and () not in self._tracked:
            root = self.game.new_initial_state()
            self._tracked[()] = (root, {v: self._prior(v) for v in self._views}, ())
        entry = self._tracked[history[:known]]
        for index in range(known, len(history)):
            entry = self._step(entry, history[index])
            self._tracked[history[: index + 1]] = entry
        return entry

    def _step(self, entry: Entry, action: str) -> Entry:
        state, joints, deferred = entry
        joints = dict(joints)
        player = state.current_player()
        if player == CHANCE:
            if state.stage == "battle":
                choices = dict(state.choices)

                def battle(cell: Cell) -> float:
                    return dict(state.battle_outcomes(choices, self._kinds(cell))).get(action, 0.0)

                joints = {v: self._update(j, battle) for v, j in joints.items()}
            # type draws are private: no public update
        else:
            combos = self._combos(player)
            if combos != [()]:
                likes = {
                    view: {
                        combo: max(
                            self._floor,
                            self._probabilities(state, player, combo, joint).get(action, 0.0),
                        )
                        for combo in combos
                    }
                    for view, joint in joints.items()
                }
                if state.stage == "effort":
                    deferred = (*deferred, (player, likes))
                else:
                    joints = {v: self._update_known(j, player, likes[v]) for v, j in joints.items()}
            if state.stage == "effort" and len(state.pending) == 1:  # the round reveals efforts
                for actor, item_likes in deferred:
                    joints = {
                        v: self._update_known(j, actor, item_likes[v]) for v, j in joints.items()
                    }
                deferred = ()
        child = state.clone()
        assert isinstance(child, AttritionState)
        child.apply_action(action)
        return child, joints, deferred

    def _known_holders(self, role: str) -> list[tuple[int, str]]:
        return [(i, h) for i, h in enumerate(self._holders) if h in self._knows[role]]

    def _combos(self, role: str) -> list[Own]:
        """Every combination of kinds for the holders a role sees."""

        combos: list[Own] = [()]
        for _, holder in self._known_holders(role):
            holder_types = self.params.types_of(holder)
            assert holder_types is not None
            combos = [(*c, (holder, k)) for c in combos for k in holder_types.kinds]
        return combos

    @staticmethod
    def _update(joint: Joint, like: Callable[[Cell], float]) -> Joint:
        new = {cell: q * like(cell) for cell, q in joint.items()}
        total = sum(new.values())
        return {cell: q / total for cell, q in new.items()} if total > 0.0 else joint

    def _update_known(self, joint: Joint, role: str, likes: Mapping[Own, float]) -> Joint:
        """Bayes on a choice by *role*, whose likelihood depends on the kinds it sees."""

        holders = self._known_holders(role)

        def like(cell: Cell) -> float:
            return likes.get(tuple((h, cell[i]) for i, h in holders), 1.0)

        return self._update(joint, like)

    def _about_other(self, joint: Joint, role: str, own: Own) -> dict[Own, float]:
        """Belief about the kinds a role does not see (the other side's, and others'
        private types), given the kinds it knows."""

        known = dict(own)
        mine = self._known_holders(role)
        theirs = [(i, h) for i, h in enumerate(self._holders) if h not in self._knows[role]]
        dist: dict[Own, float] = {}
        for cell, q in joint.items():
            if any(h in known and cell[i] != known[h] for i, h in mine):
                continue
            key = tuple((h, cell[i]) for i, h in theirs)
            dist[key] = dist.get(key, 0.0) + q
        total = sum(dist.values())
        if total <= 0.0:
            return {k: 1.0 / len(dist) for k in dist} if dist else {(): 1.0}
        return {k: q / total for k, q in dist.items()}

    @staticmethod
    def _cell_kinds(own: Own, other: Own) -> dict[str, str]:
        return {**dict(own), **dict(other)}

    def _own_in(self, kinds: Mapping[str, str], role: str) -> Own:
        """The part of *kinds* that *role* sees."""

        return tuple((h, kinds[h]) for h in self._knows[role] if h in kinds)

    # ---------------------------------------------------------------- values
    @staticmethod
    def _levels(sponsor: Sponsor) -> tuple[str, ...]:
        return ("low", "high", "direct") if sponsor.direct else ("low", "high")

    @staticmethod
    def _withdraw_value(sponsor: Sponsor) -> float:
        """Its side's defeat by withdrawal, plus its own cost of withdrawing."""

        label = "a_withdrew" if sponsor.side == "a" else "b_withdrew"
        return sponsor.values[label] - sponsor.withdraw_cost

    def _state_key(self, state: AttritionState) -> tuple[Any, ...]:
        # computed once per state and move: the cache holds the state itself, so its id
        # cannot be reused while cached, and a move changes the history's length
        moves = len(state.history())
        cached = self._keys.get(id(state))
        if cached is not None and cached[0] is state and cached[1] == moves:
            return cached[2]
        key = (
            state.round,
            round(state.control, 9),
            tuple(sorted(state.withdrawn)),
            tuple(sorted(state.last_choices.items())),
            tuple(sorted((r, round(w, 9)) for r, w in state.weariness.items())),
            tuple(sorted(state.deployed)),
            tuple(sorted(state.committed)),
            tuple(sorted(state.leads.items())),
        )
        if len(self._keys) > KEY_CACHE_LIMIT:
            self._keys.clear()
        self._keys[id(state)] = (state, moves, key)
        return key

    def _remember(self, key: Any, compute: Callable[[], Any]) -> Any:
        if key in self._memo:
            return self._memo[key]
        if len(self._memo) > CACHE_LIMIT:
            self._memo.clear()
        value = compute()
        self._memo[key] = value
        return value

    def _walk(
        self,
        state: AttritionState,
        role: str,
        level: str,
        kinds: Mapping[str, str],
        assumed: Mapping[str, str],
        control: float | None = None,
        later: bool = False,
    ) -> float:
        """Exact value to *role* of fighting on at *level* to the end, both types fixed;
        from the next round (every force arrived) if *later*."""

        # a walk depends on the state only through the round, the withdrawn sponsors, the
        # role's weariness and (this round) the deployed ones, and on the kinds only
        # through strengths and this role's own costs. Its values at every position of
        # the front are remembered together.
        key = (
            "walk",
            state.round,
            tuple(sorted(state.withdrawn)),
            None if later else tuple(sorted(state.deployed)),
            round(state.weariness[role], 9),
            role,
            level,
            tuple((h, kinds[h]) for h in self._strength_holders if h in kinds),
            round(state.role_cost_multiplier(role, kinds), 9),
            round(state.proxy_multiplier(role, kinds), 9),
            tuple(sorted(assumed.items())),
            later,
        )
        values: tuple[float, ...] = self._remember(
            key, lambda: self._walk_values(state, role, level, kinds, assumed, later)
        )
        p = self.params
        low = 1.0 - p.victory_threshold
        where = state.control if control is None else control
        position = min(len(values) - 2, max(1, round((where - low) / p.step)))
        return values[position]

    def _walk_values(
        self,
        state: AttritionState,
        role: str,
        level: str,
        kinds: Mapping[str, str],
        assumed: Mapping[str, str],
        later: bool = False,
    ) -> tuple[float, ...]:
        """The walk's value at every position of the front."""

        p = self.params
        sponsor = p.sponsor(role)
        trial = dict(assumed)
        trial[role] = level
        other = other_side(sponsor.side)
        low, high = 1.0 - p.victory_threshold, p.victory_threshold
        cells = max(2, round((high - low) / p.step))
        if all(
            s.id in state.withdrawn or trial.get(s.id) == "withdraw"
            for s in p.sponsors
            if s.side == other
        ):  # the other side is expected to quit
            return (sponsor.values["b_withdrew" if other == "b" else "a_withdrew"],) * (cells + 1)
        everyone = [s.id for s in p.sponsors]
        if p.momentum > 0.0:  # the local balance, and so the odds, differ along the front
            fronts = [low + i * p.step for i in range(cells + 1)]
            arrived = [self._odds(state, trial, kinds, everyone, c) for c in fronts]
            first = [self._odds(state, trial, kinds, None, c) for c in fronts]
        else:
            arrived = [self._odds(state, trial, kinds, everyone)] * (cells + 1)  # all arrived
            first = [self._odds(state, trial, kinds, None)] * (cells + 1)  # this round, any lag
        index = p.level_index(level)
        multiplier = state.role_cost_multiplier(role, kinds)
        cost = sponsor.cost_scale * p.costs[index] * multiplier if index >= 0 else 0.0
        cost -= state.proxy_gain(role, trial, kinds)
        if level == "direct":
            cost += sponsor.direct_cost
        rounds = max(0, p.max_rounds - state.round - int(later))
        end_value = sponsor.values["stalemate"]
        per_round = p.weariness[index] if index >= 0 else 0.0
        if per_round > 0.0:
            room = p.weariness_limit - state.weariness[role] - per_round * int(later)
            until_out = max(1, math.ceil(room / per_round - 1e-12))
            if until_out < rounds:
                rounds = until_out
                end_value = self._withdraw_value(sponsor)
        values = [end_value] * (cells + 1)
        values[0], values[cells] = sponsor.values["b_victory"], sponsor.values["a_victory"]
        for step in range(rounds):
            last = step == rounds - 1 and not later  # the last update is this round's battle
            table = first if last else arrived
            values = [
                values[0],
                *(
                    table[i][1]
                    * (table[i][0] * values[i + 1] + (1.0 - table[i][0]) * values[i - 1])
                    + (1.0 - table[i][1]) * values[i]
                    - cost
                    for i in range(1, cells)
                ),
                values[cells],
            ]
        return tuple(values)

    @staticmethod
    def _odds(
        state: AttritionState,
        choices: Mapping[str, str],
        kinds: Mapping[str, str],
        deployed: list[str] | None,
        control: float | None = None,
    ) -> tuple[float, float]:
        """P(A wins a battle) and P(the front moves), from one pass over the efforts,
        with the front at *control* (the state's if none)."""

        e_a = state.side_effort("a", choices, kinds, deployed, control)
        e_b = state.side_effort("b", choices, kinds, deployed, control)
        g = state.params.gamma
        w_a, w_b = e_a**g, e_b**g
        p_a = w_a / (w_a + w_b) if w_a + w_b > 0.0 else 0.5
        mobility = state.params.mobility
        move = 1.0 if mobility is None else mobility.moves(e_a + e_b)
        return float(p_a), float(move)

    def _assumed(
        self, state: AttritionState, role: str, kinds: Mapping[str, str]
    ) -> dict[str, str]:
        """Others' efforts: their last choices, or in the first round their best
        response (withdrawing included) to everyone else at low effort."""

        p = self.params
        if state.last_choices:
            return {s.id: state.last_choices.get(s.id, "low") for s in p.sponsors}
        key = ("assumed", self._state_key(state), role, tuple(sorted(kinds.items())))

        def compute() -> dict[str, str]:
            base = {s.id: "low" for s in p.sponsors}
            result = dict(base)
            for sponsor in p.sponsors:
                if sponsor.id == role or sponsor.id in state.withdrawn:
                    continue
                levels = self._levels(sponsor)
                best = max(
                    levels,
                    key=lambda lv: (
                        self._walk(state, sponsor.id, lv, kinds, base),
                        -levels.index(lv),
                    ),
                )
                # others are assumed to stay unless quitting beats fighting even as their
                # side's defeat: assuming that each free-rides on the others would cascade
                if self._withdraw_value(sponsor) > self._walk(state, sponsor.id, best, kinds, base):
                    best = "withdraw"
                result[sponsor.id] = best
            return result

        return dict(self._remember(key, compute))

    def _fight(
        self,
        state: AttritionState,
        role: str,
        own: Own,
        joint: Joint,
        control: float | None = None,
    ) -> dict[str, float]:
        """Value of each fighting level, averaged over the other side's types."""

        sponsor = self.params.sponsor(role)
        dist = self._about_other(joint, role, own)
        value = self._walk if self.params.offers is None else self._continue
        result = {}
        for level in (*self._levels(sponsor), "withdraw"):
            total = 0.0
            for other, q in dist.items():
                kinds = self._cell_kinds(own, other)
                assumed = self._assumed(state, role, kinds)
                if level == "withdraw":
                    total += q * self._quit(state, role, kinds, assumed, control, offers=True)
                else:
                    total += q * value(state, role, level, kinds, assumed, control)
            result[level] = total
        return result

    def _quit(
        self,
        state: AttritionState,
        role: str,
        kinds: Mapping[str, str],
        assumed: Mapping[str, str],
        control: float | None = None,
        later: bool = False,
        offers: bool = False,
    ) -> float:
        """Value to *role* of withdrawing: the side's defeat if it leads its side or is
        the last sponsor left, otherwise the war going on without its effort or costs
        (the lead carries the war; others may free-ride on it)."""

        p = self.params
        sponsor = p.sponsor(role)
        trial = {**assumed, role: "withdraw"}
        if role == state.lead(sponsor.side) or all(
            s.id in state.withdrawn or trial.get(s.id) == "withdraw"
            for s in p.sponsors
            if s.side == sponsor.side
        ):
            return self._withdraw_value(sponsor)
        # free-riding on the lead still costs the sponsor its withdraw cost, once
        if offers and p.offers is not None and not later:
            value = self._continue(state, role, "withdraw", kinds, assumed, control)
        else:
            value = self._walk(state, role, "withdraw", kinds, trial, control, later)
        return value - sponsor.withdraw_cost

    def _continue(
        self,
        state: AttritionState,
        role: str,
        level: str,
        kinds: Mapping[str, str],
        assumed: Mapping[str, str],
        control: float | None = None,
    ) -> float:
        """Value to *role* of fighting this round at *level*, then settling near the new
        line of contact or fighting on (module docstring), both types fixed."""

        key = (
            "continue",
            self._state_key(state),
            role,
            level,
            tuple(sorted(kinds.items())),
            tuple(sorted(assumed.items())),
            control,
        )

        def compute() -> float:
            p = self.params
            sponsor = p.sponsor(role)
            trial = dict(assumed)
            trial[role] = level
            other = other_side(sponsor.side)
            if all(
                s.id in state.withdrawn or trial.get(s.id) == "withdraw"
                for s in p.sponsors
                if s.side == other
            ):
                return sponsor.values["b_withdrew" if other == "b" else "a_withdrew"]
            where = state.control if control is None else control
            odds = state.win_probability(trial, kinds, control=where)
            move = state.move_probability(trial, kinds, control=where)
            index = p.level_index(level)
            multiplier = state.role_cost_multiplier(role, kinds)
            cost = sponsor.cost_scale * p.costs[index] * multiplier if index >= 0 else 0.0
            cost -= state.proxy_gain(role, trial, kinds)
            if level == "direct":
                cost += sponsor.direct_cost
            total = 0.0
            for shift, q in (
                (p.step, move * odds),
                (-p.step, move * (1.0 - odds)),
                (0.0, 1.0 - move),
            ):
                if q > 0.0:
                    total += q * self._next_round(state, role, kinds, trial, where + shift)
            return total - cost

        return float(self._remember(key, compute))

    def _next_round(
        self,
        state: AttritionState,
        role: str,
        kinds: Mapping[str, str],
        trial: Mapping[str, str],
        control: float,
    ) -> float:
        """Value to *role* of the war from the next round at *control*."""

        p = self.params
        offers = p.offers
        assert offers is not None
        sponsor = p.sponsor(role)
        if control >= p.victory_threshold - 1e-12:
            return sponsor.values["a_victory"]
        if control <= 1.0 - p.victory_threshold + 1e-12:
            return sponsor.values["b_victory"]
        if state.round + 1 >= p.max_rounds:
            return sponsor.values["stalemate"]

        state_key = self._state_key(state)
        kinds_key = tuple(sorted(kinds.items()))
        trial_key = tuple(sorted(trial.items()))

        def fight_on(party: str) -> float:
            def compute() -> float:
                if trial.get(party) == "withdraw":  # a withdrawn sponsor does not come back
                    return self._walk(state, party, "withdraw", kinds, trial, control, later=True)
                levels = self._levels(p.sponsor(party))
                best = max(
                    self._walk(state, party, lv, kinds, {**trial, party: lv}, control, later=True)
                    for lv in levels
                )
                return max(best, self._quit(state, party, kinds, trial, control, later=True))

            key = ("fight_on", state_key, party, kinds_key, trial_key, control)
            return float(self._remember(key, compute))

        def settlement() -> float | None:
            share = self._line_settlement(state, control)
            parties = [
                (state.lead(side), 0.0)
                for side in ("a", "b")
                if state.lead(side) not in state.withdrawn
            ]
            parties += [
                (c.role, c.veto_cost) for c in offers.consent if c.role not in state.withdrawn
            ]
            agreed = all(
                self._settle_value(state, party, share) >= fight_on(party) - cost
                for party, cost in parties
            )
            return share if agreed else None

        share = self._remember(("settle", state_key, kinds_key, trial_key, control), settlement)
        if share is not None:
            return self._settle_value(state, role, share)
        return fight_on(role)

    def _line_settlement(self, state: AttritionState, control: float) -> float:
        """The offer nearest the line of contact that meets every commitment made, or
        the nearest offer if none does."""

        offers = self.params.offers
        assert offers is not None
        made = [c for c in self.params.commitments if c.role in state.committed]
        meeting = [g for g in offers.grid if not any(c.breached(g) for c in made)]
        return min(meeting or offers.grid, key=lambda g: (abs(g - control), g))

    def _settle_value(self, state: AttritionState, role: str, share: float) -> float:
        """A settlement's value to *role*, net of the audience cost of a breached commitment."""

        value = self.params.sponsor(role).partition_value(share)
        item = self.params.commitment(role)
        if item is not None and role in state.committed and item.breached(share):
            value -= item.cost
        return value

    def _carry_on(self, state: AttritionState, role: str, own: Own, joint: Joint) -> float:
        """Value of carrying on (the best fighting level, or withdrawing)."""

        key = (
            "carry",
            self._state_key(state),
            role,
            own,
            tuple(sorted((c, round(q, 9)) for c, q in joint.items())),
        )

        def compute() -> float:
            return max(self._fight(state, role, own, joint).values())

        return float(self._remember(key, compute))

    # ---------------------------------------------------------------- stages
    def _probabilities(
        self, state: AttritionState, role: str, own: Own, joint: Joint
    ) -> dict[str, float]:
        scores = self._scores(state, role, own, joint)
        return choose(scores, state.legal_actions(), self.temperature)

    def _scores(self, state: AttritionState, role: str, own: Own, joint: Joint) -> dict[str, float]:
        p = self.params
        sponsor = p.sponsor(role)
        stage = state.stage
        if stage == "approval":
            return {
                "approve": self._carry_on(state, role, own, joint),
                "refuse": sponsor.values[REFUSED],
            }
        if stage == "warn":
            return {"warn": self._entry_gain(state, role, own, joint), "silent": 0.0}
        if stage == "commit":
            pledged = state.clone()
            assert isinstance(pledged, AttritionState)
            pledged.committed.add(role)
            return {
                "commit": self._carry_on(pledged, role, own, joint),
                "hold_off": self._carry_on(state, role, own, joint),
            }
        if stage == "offer":
            return self._offer_scores(state, role, own, joint)
        if stage == "respond":
            assert p.offers is not None and state.offer is not None
            return self._respond_scores(state, role, own, joint, p.offers.grid[state.offer])
        if stage == "ratify":
            terms = state.terms
            assert p.offers is not None and terms is not None
            return self._ratify_scores(state, role, own, joint, terms)
        fight = self._fight(state, role, own, joint)
        scores: dict[str, float] = {}
        for action in state.legal_actions():
            if action == "negotiate":  # without offers: a settlement at the line
                scores[action] = self._settle_value(state, role, state.control)
            else:
                scores[action] = fight[action]
        line = p.red_line(role)
        if (
            line is not None
            and role in state.warned
            and role not in state.bluffed | state.honored
            and line.crossed(state.control)
        ):
            for action in scores:
                if action != "direct":
                    scores[action] -= line.audience_cost
        return scores

    def _entry_gain(self, state: AttritionState, role: str, own: Own, joint: Joint) -> float:
        """Gain from going direct, rather than the best alternative, once the line is crossed."""

        line = self.params.red_line(role)
        assert line is not None
        fight = self._fight(state, role, own, joint, control=line.beyond(self.params.step))
        others = [v for level, v in fight.items() if level != "direct"]
        return fight["direct"] - max(others)

    def _offer_scores(
        self, state: AttritionState, role: str, own: Own, joint: Joint
    ) -> dict[str, float]:
        p = self.params
        offers = p.offers
        assert offers is not None
        carry = self._carry_on(state, role, own, joint)
        dist = self._about_other(joint, role, own)
        scores = {"no_offer": carry}
        for index, share in enumerate(offers.grid):
            deal = 0.0
            for other, q in dist.items():
                kinds = self._cell_kinds(own, other)
                deal += q * self._deal_probability(state, share, kinds, joint)
            scores[offers.offer_id(index)] = (
                deal * self._settle_value(state, role, share) + (1.0 - deal) * carry
            )
        return scores

    def _deal_probability(
        self, state: AttritionState, share: float, kinds: Mapping[str, str], joint: Joint
    ) -> float:
        responder = state.responder()
        assert responder is not None
        scores = self._respond_scores(
            state, responder, self._own_in(kinds, responder), joint, share
        )
        accept = choose(scores, ("accept", "reject"), self.temperature)["accept"]
        return accept * self._ratify_probability(state, share, kinds, joint)

    def _ratify_probability(
        self, state: AttritionState, share: float, kinds: Mapping[str, str], joint: Joint
    ) -> float:
        offers = self.params.offers
        assert offers is not None
        result = 1.0
        for item in offers.consent:
            if item.role in state.withdrawn:
                continue
            own = self._own_in(kinds, item.role)
            scores = self._ratify_scores(state, item.role, own, joint, share)
            result *= choose(scores, ("ratify", "veto"), self.temperature)["ratify"]
        return result

    def _respond_scores(
        self, state: AttritionState, role: str, own: Own, joint: Joint, share: float
    ) -> dict[str, float]:
        carry = self._carry_on(state, role, own, joint)
        ratified = 0.0
        for other, q in self._about_other(joint, role, own).items():
            kinds = self._cell_kinds(own, other)
            ratified += q * self._ratify_probability(state, share, kinds, joint)
        settle = self._settle_value(state, role, share)
        return {"accept": ratified * settle + (1.0 - ratified) * carry, "reject": carry}

    def _ratify_scores(
        self, state: AttritionState, role: str, own: Own, joint: Joint, share: float
    ) -> dict[str, float]:
        offers = self.params.offers
        assert offers is not None
        sponsor = self.params.sponsor(role)
        veto = self._carry_on(state, role, own, joint) - offers.consent_of(role).veto_cost
        return {"ratify": sponsor.partition_value(share), "veto": veto}

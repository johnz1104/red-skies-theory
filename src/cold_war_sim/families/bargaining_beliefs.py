"""Perfect Bayesian equilibrium of bargaining with hidden resolve and a contest.

Used by :class:`~.bargaining.BargainingEquilibriumPolicy` for bargaining games with
``types`` or a ``contest``. Each typed side is ``strong`` (low per-period cost of
carrying on) or ``weak`` (high cost), known only to itself. Public beliefs are
mu_A = P(A strong) and mu_B = P(B strong); an untyped side has a single type and
belief 1. The solution is a backward induction over (commitments, period,
position, mu_A, mu_B), with beliefs on uniform grids and linear interpolation.

At every node:

* **the responder screens.** Given the belief about the proposer after the offer,
  each responder type accepts when its agreement value is at least its value of
  rejecting. When the weak (high-cost) type would accept at the current belief but
  not if a rejection marked it as strong, it mixes so that its indifference pins
  the belief after a rejection (Fudenberg and Tirole 1991, ch. 10). An unexpected
  rejection is read as strength.
* **the proposer signals** (when typed). Pure offer profiles (strong type's offer,
  weak type's offer) are enumerated, with beliefs set by Bayes' rule on the path.
  *Stated rule:* an unexpected offer is read as weakness (belief 0), and among the
  profiles no type wants to leave, the one best for the proposer at the current
  belief is played (``selection: proposer_best``), or the best pooling profile
  when one exists (``pooling_first``). If no pure profile survives, the weak type
  mixes between the strong type's offer and its own best offer (semi-separating),
  with the mixture set by its indifference.

Commitment choices by a typed side are signals and use the same rule. Nodes that
needed the semi-separating mixture, or a last-resort fallback, are counted in
:attr:`BeliefSolution.report`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .bargaining import (
    STRONG,
    TOLERANCE,
    BargainingParams,
    BargainingState,
    consent_ratifies,
    side_payoff,
)

EPS = 1e-9
SEMI_ITERATIONS = 24
EDGE_TOLERANCE = 1e-6
Array = np.ndarray


@dataclass(frozen=True)
class SideTypes:
    """A side's types, their delay costs, its prior and its belief grid."""

    kinds: tuple[str, ...]
    costs: tuple[float, ...]
    prior: float
    grid: Array

    @property
    def typed(self) -> bool:
        return len(self.kinds) > 1

    def weights(self, mu: Array) -> Array:
        """P(each type) given belief *mu* (P(strong)); shape (types, *mu.shape)."""

        if not self.typed:
            return np.ones((1, *mu.shape))
        return np.stack([mu, 1.0 - mu])


@dataclass
class Signal:
    """A proposer's (or committer's) behaviour at a vector of belief points."""

    probs: Array  # (types, n, options): each type's distribution over options
    post: Array  # (n, options): the observer's belief about the sender after each option
    semi: Array  # (n,) bool: semi-separating mixture used
    fallback: Array  # (n,) bool: no equilibrium profile found; best guess played
    # (n, 3): for a semi-separating offer at which the observer's response jumps, the
    # bracket (lo, hi) around the jump and the weight on the response just above it
    edge: Array | None = None


@dataclass
class Response:
    accept: Array  # (responder types, n): probability of accepting
    post: Array  # (n,): belief about the responder after a rejection
    values: Array  # (2, kA, kB, n): both sides' values for each true type pair
    proposer: Array  # (proposer types, n): proposer's expected value by its type
    fallback: Array  # (n,) bool


@dataclass
class Stage:
    signal: Signal
    responses: list[Response]  # one per offer, at that offer's posterior
    values: Array  # (2, kA, kB, n)


def _grid_index(grid: Array, mu: Array) -> tuple[Array, Array, Array]:
    """Neighbouring grid indices and the interpolation weight of the upper one."""

    if grid.size == 1:
        zeros = np.zeros(mu.shape, dtype=int)
        return zeros, zeros, np.zeros(mu.shape)
    m = np.clip(mu, 0.0, 1.0)
    i1 = np.clip(np.searchsorted(grid, m, side="right"), 1, grid.size - 1)
    i0 = i1 - 1
    return i0, i1, (m - grid[i0]) / (grid[i1] - grid[i0])


@dataclass
class BeliefSolution:
    """Solved value tables and the stage rules that produce live behaviour."""

    params: BargainingParams
    proposer_grid: int = 21
    responder_grid: int = 51
    sides: dict[str, SideTypes] = field(init=False)
    values: dict[tuple[tuple[bool, bool], int, int], Array] = field(init=False)
    report: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        p = self.params
        fixed = p.proposers if p.proposers != "alternate" else None
        self.sides = {key: self._side_types(key, fixed) for key in ("A", "B")}
        self.values = {}
        self.report = {"semi_separating": 0, "fallback": 0, "screen_fallback": 0}
        self._terminal_cache: dict[tuple[object, ...], Array] = {}
        a_options = (False, True) if p.a.commitment is not None else (False,)
        b_options = (False, True) if p.b.commitment is not None else (False,)
        self.combos = [(ca, cb) for ca in a_options for cb in b_options]
        for combo in self.combos:
            for t in range(p.horizon - 1, -1, -1):
                for i in self.positions(t):
                    self.values[(combo, t, i)] = self._node_table(combo, t, i)

    # ------------------------------------------------------------------ set-up
    def _side_types(self, key: str, fixed: str | None) -> SideTypes:
        p = self.params
        types = p.types_of(key)
        if types is None:
            return SideTypes((STRONG,), (p.side(key).delay_cost,), 1.0, np.array([1.0]))
        size = self.proposer_grid if key == fixed else self.responder_grid
        grid = np.unique(np.concatenate([np.linspace(0.0, 1.0, size), [types.prior]]))
        return SideTypes((STRONG, "weak"), (types.strong_cost, types.weak_cost), types.prior, grid)

    def positions(self, t: int) -> list[int]:
        contest = self.params.contest
        if contest is None:
            return [0]
        return [i for i in range(-t, t + 1) if -contest.steps_down < i < contest.steps_up]

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.sides["A"].kinds), len(self.sides["B"].kinds)

    # --------------------------------------------------------------- terminals
    def _terminal(
        self, label: str, share: float | None, period: int, agreed: int | None, combo: tuple
    ) -> Array:
        key = (label, share, period, agreed, combo)
        cached = self._terminal_cache.get(key)
        if cached is not None:
            return cached
        p = self.params
        ka, kb = self.shape
        out = np.zeros((2, ka, kb))
        for s, side in enumerate(("A", "B")):
            for a in range(ka):
                for b in range(kb):
                    own = a if side == "A" else b
                    out[s, a, b] = side_payoff(
                        p,
                        side,
                        label,
                        share=share,
                        period=period,
                        agreed_period=agreed,
                        committed=combo[s],
                        cost=self.sides[side].costs[own],
                    )
        self._terminal_cache[key] = out
        return out

    def accept_values(self, combo: tuple, t: int, x: int) -> Array:
        p = self.params
        share = p.grid[x]
        ratified = all(consent_ratifies(c, share) for c in p.consent)
        return self._terminal("agreement" if ratified else "vetoed", share, t, t, combo)

    # ------------------------------------------------------------ continuation
    def _interpolate(self, combo: tuple, t: int, i: int, mu_a: Array, mu_b: Array) -> Array:
        table = self.values[(combo, t, i)]
        a0, a1, fa = _grid_index(self.sides["A"].grid, mu_a)
        b0, b1, fb = _grid_index(self.sides["B"].grid, mu_b)
        return np.asarray(
            table[:, :, :, a0, b0] * (1 - fa) * (1 - fb)
            + table[:, :, :, a1, b0] * fa * (1 - fb)
            + table[:, :, :, a0, b1] * (1 - fa) * fb
            + table[:, :, :, a1, b1] * fa * fb
        )

    def _next(self, combo: tuple, t1: int, i: int, mu_a: Array, mu_b: Array) -> Array:
        p = self.params
        if t1 >= p.horizon:
            values = self._terminal("impasse", None, p.horizon, None, combo)
            return np.broadcast_to(values[..., *([None] * mu_a.ndim)], (*values.shape, *mu_a.shape))
        return self._interpolate(combo, t1, i, mu_a, mu_b)

    def _moved(self, combo: tuple, t: int, j: int, mu_a: Array, mu_b: Array) -> Array:
        contest = self.params.contest
        assert contest is not None
        if j >= contest.steps_up or j <= -contest.steps_down:
            share = 1.0 if j >= contest.steps_up else 0.0
            values = self._terminal("agreement", share, t + 1, t + 1, combo)
            return np.broadcast_to(values[..., *([None] * mu_a.ndim)], (*values.shape, *mu_a.shape))
        return self._next(combo, t + 1, j, mu_a, mu_b)

    def after_rejection(self, combo: tuple, t: int, i: int, mu_a: Array, mu_b: Array) -> Array:
        """Both sides' values for each type pair after a rejection at period *t*."""

        p = self.params
        risk = p.breakdown_risk(t)
        breakdown = self._terminal("breakdown", None, t, None, combo)
        extra = (None,) * mu_a.ndim
        if risk >= 1.0:
            return np.broadcast_to(breakdown[..., *extra], (*breakdown.shape, *mu_a.shape))
        contest = p.contest
        cont: Array
        if contest is None:
            cont = self._next(combo, t + 1, i, mu_a, mu_b)
        else:
            q = contest.win_probability
            cont = np.zeros((2, *self.shape, *mu_a.shape))
            if q > 0.0:
                cont = cont + q * self._moved(combo, t, i + 1, mu_a, mu_b)
            if q < 1.0:
                cont = cont + (1.0 - q) * self._moved(combo, t, i - 1, mu_a, mu_b)
        return risk * breakdown[..., *extra] + (1.0 - risk) * cont

    # ------------------------------------------------------------------ stages
    def _roles(self, t: int) -> tuple[str, str]:
        proposer = self.params.proposer(t)
        return proposer, ("B" if proposer == "A" else "A")

    def _beliefs_for(self, proposer: str, mu_p: Array, mu_r: Array) -> tuple[Array, Array]:
        return (mu_p, mu_r) if proposer == "A" else (mu_r, mu_p)

    def rejection_grid(self, combo: tuple, t: int, i: int, post_p: Array) -> Array:
        """after_rejection at (proposer belief post_p, every responder grid belief)."""

        proposer, responder = self._roles(t)
        grid = self.sides[responder].grid
        n = post_p.shape[0]
        mu_p = np.repeat(post_p[:, None], grid.size, axis=1)
        mu_r = np.repeat(grid[None, :], n, axis=0)
        mu_a, mu_b = self._beliefs_for(proposer, mu_p, mu_r)
        return self.after_rejection(combo, t, i, mu_a, mu_b)  # (2, kA, kB, n, g)

    def prepare(self, combo: tuple, t: int, i: int, post_p: Array) -> tuple[Array, Array]:
        """Rejection values over the responder's grid, for both sides and by the
        responder's own type; they do not depend on the offer."""

        proposer, responder = self._roles(t)
        rv = self.rejection_grid(combo, t, i, post_p)
        w_p = self.sides[proposer].weights(post_p)  # (kP, n)
        if responder == "B":
            r_grid = (w_p[:, None, :, None] * rv[1]).sum(axis=0)  # (kB, n, g)
        else:
            r_grid = (w_p[None, :, :, None] * rv[0]).sum(axis=1)  # (kA, n, g)
        return rv, r_grid

    def respond(
        self,
        combo: tuple,
        t: int,
        i: int,
        x: int,
        post_p: Array,
        mu_r: Array,
        prepared: tuple[Array, Array] | None = None,
    ) -> Response:
        """Responder behaviour to offer *x* when the proposer is believed strong w.p. post_p."""

        _, responder = self._roles(t)
        rv, r_grid = self.prepare(combo, t, i, post_p) if prepared is None else prepared
        accept = self.accept_values(combo, t, x)
        r_types = self.sides[responder]
        u_r = accept[1, 0, :] if responder == "B" else accept[0, :, 0]
        grid = r_types.grid
        n = post_p.shape[0]

        def along(table: Array, m: Array) -> Array:  # interpolate the last axis at m (n,)
            j0, j1, f = _grid_index(grid, m)
            rows = np.arange(n)
            return np.asarray(table[..., rows, j0] * (1 - f) + table[..., rows, j1] * f)

        accept_p = np.zeros((len(r_types.kinds), n))
        post = mu_r.copy()
        fallback = np.zeros(n, dtype=bool)
        r_now = along(r_grid, mu_r)
        if not r_types.typed:
            accept_p[0] = u_r[0] >= r_now[0] - TOLERANCE
        else:
            s, w = 0, 1
            r_one, r_zero = r_grid[:, :, -1], r_grid[:, :, 0]
            done = (mu_r <= EPS) | (mu_r >= 1.0 - EPS)
            for k in (s, w):  # a known type decides at a belief that cannot move
                accept_p[k] = np.where(done, u_r[k] >= r_now[k] - TOLERANCE, 0.0)

            def take(mask: Array, acc_s: Array | float, acc_w: Array | float, m: Array) -> None:
                nonlocal done
                accept_p[s] = np.where(mask, acc_s, accept_p[s])
                accept_p[w] = np.where(mask, acc_w, accept_p[w])
                post[:] = np.where(mask, m, post)
                done = done | mask

            ones, zeros = np.ones(n), np.zeros(n)
            take(
                ~done & (u_r[s] >= r_one[s] - TOLERANCE) & (u_r[w] >= r_one[w] - TOLERANCE),
                1.0,
                1.0,
                ones,
            )
            take(
                ~done & (u_r[w] >= r_one[w] - TOLERANCE) & (u_r[s] <= r_one[s] + TOLERANCE),
                0.0,
                1.0,
                ones,
            )
            mixing = ~done & (u_r[w] - r_now[w] > TOLERANCE) & (u_r[w] - r_one[w] < -TOLERANCE)
            if mixing.any():
                m_star = self._crossing(u_r[w], r_grid[w], mu_r, r_now[w], grid)
                strong_rejects = along(r_grid[s], m_star) >= u_r[s] - TOLERANCE
                ok = mixing & strong_rejects
                with np.errstate(divide="ignore", invalid="ignore"):
                    alpha = 1.0 - mu_r * (1.0 - m_star) / (m_star * (1.0 - mu_r))
                take(ok, 0.0, np.clip(alpha, 0.0, 1.0), m_star)
            take(
                ~done & (u_r[s] <= r_now[s] + TOLERANCE) & (u_r[w] <= r_now[w] + TOLERANCE),
                0.0,
                0.0,
                mu_r,
            )
            take(
                ~done & (u_r[s] >= r_zero[s] - TOLERANCE) & (u_r[w] <= r_zero[w] + TOLERANCE),
                1.0,
                0.0,
                zeros,
            )
            fallback = ~done
            if fallback.any():
                take(
                    fallback,
                    (u_r[s] >= r_now[s] - TOLERANCE).astype(float),
                    (u_r[w] >= r_now[w] - TOLERANCE).astype(float),
                    mu_r,
                )
        response = self._assemble(combo, t, x, accept_p, post, rv, mu_r)
        response.fallback = fallback
        return response

    def _assemble(
        self, combo: tuple, t: int, x: int, accept_p: Array, post: Array, rv: Array, mu_r: Array
    ) -> Response:
        """Values by type pair, and the proposer's by its type, for given responses."""

        proposer, responder = self._roles(t)
        accept = self.accept_values(combo, t, x)
        grid = self.sides[responder].grid
        n = post.shape[0]
        j0, j1, f = _grid_index(grid, post)
        rows = np.arange(n)
        rv_post = rv[..., rows, j0] * (1 - f) + rv[..., rows, j1] * f  # (2, kA, kB, n)
        values = np.empty_like(rv_post)
        ka, kb = self.shape
        for a in range(ka):
            for b in range(kb):
                acc = accept_p[b if responder == "B" else a]
                values[:, a, b] = acc * accept[:, a, b, None] + (1 - acc) * rv_post[:, a, b]
        w_r = self.sides[responder].weights(mu_r)  # the proposer's expectation: prior belief
        p_side = 0 if proposer == "A" else 1
        if proposer == "A":
            prop = (w_r[None, :, :] * values[p_side]).sum(axis=1)
        else:
            prop = (w_r[:, None, :] * values[p_side]).sum(axis=0)
        return Response(accept_p, post, values, prop, np.zeros(n, dtype=bool))

    @staticmethod
    def _crossing(u: Array, r: Array, mu: Array, r_mu: Array, grid: Array) -> Array:
        """First belief m >= mu where u - r(m) crosses from positive to <= 0."""

        n = r.shape[0]
        d_grid = u - r  # (n, g)
        d_mu = u - r_mu  # (n,)
        prev_m = np.repeat(grid[None, :-1], n, axis=0)
        prev_d = d_grid[:, :-1]
        next_m = np.repeat(grid[None, 1:], n, axis=0)
        next_d = d_grid[:, 1:]
        before = prev_m < mu[:, None]  # segment starts left of mu: start it at mu
        prev_m = np.where(before, mu[:, None], prev_m)
        prev_d = np.where(before, d_mu[:, None], prev_d)
        valid = next_m > mu[:, None]
        cross = valid & (prev_d > 0) & (next_d <= 0)
        first = np.argmax(cross, axis=1)
        rows = np.arange(n)
        m0, m1 = prev_m[rows, first], next_m[rows, first]
        d0, d1 = prev_d[rows, first], next_d[rows, first]
        with np.errstate(divide="ignore", invalid="ignore"):
            m = m0 + (m1 - m0) * d0 / (d0 - d1)
        found = cross[rows, first]
        return np.where(found, np.clip(m, mu, 1.0), 1.0)

    def stage(self, combo: tuple, t: int, i: int, mu_a: Array, mu_b: Array) -> Stage:
        """Proposer and responder behaviour, and node values, at belief points."""

        p = self.params
        proposer, _ = self._roles(t)
        mu_p, mu_r = (mu_a, mu_b) if proposer == "A" else (mu_b, mu_a)
        n = mu_p.shape[0]
        options = len(p.grid)
        p_types = self.sides[proposer]
        cache: dict[str, tuple[Array, list[Response]]] = {}

        def evaluate(key: str, post: Array) -> list[Response]:
            if key not in cache:
                prepared = self.prepare(combo, t, i, post)
                cache[key] = (
                    post,
                    [self.respond(combo, t, i, x, post, mu_r, prepared) for x in range(options)],
                )
            return cache[key][1]

        def value_at(post: Array, x: int, idx: Array) -> Array:
            return self.respond(combo, t, i, x, post, mu_r[idx]).proposer

        if p_types.typed:
            tables = {
                "p": [r.proposer for r in evaluate("p", mu_p)],
                "0": [r.proposer for r in evaluate("0", np.zeros(n))],
                "1": [r.proposer for r in evaluate("1", np.ones(n))],
            }
        else:
            tables = {"p": [r.proposer for r in evaluate("p", mu_p)]}
        signal = select_signal(tables, mu_p, p.selection, value_at, typed=p_types.typed)
        self.report["semi_separating"] += int(signal.semi.sum())
        self.report["fallback"] += int(signal.fallback.sum())

        # responses at each offer's actual posterior
        responses: list[Response] = []
        for x in range(options):
            post = signal.post[:, x]
            if p_types.typed:
                candidates = {"p": mu_p, "0": np.zeros(n), "1": np.ones(n)}
                resp = self._merge(post, candidates, evaluate, x, combo, t, i, mu_r)
            else:
                resp = evaluate("p", mu_p)[x]
            if signal.edge is not None:
                at_edge = ~np.isnan(signal.edge[:, 0]) & (signal.probs[0, :, x] > 0.5)
                if p_types.typed and at_edge.any():
                    resp = self._edge(combo, t, i, x, signal.edge, at_edge, mu_r, resp)
            responses.append(resp)
            self.report["screen_fallback"] += int(resp.fallback.sum())
        values = np.zeros((2, *self.shape, n))
        ka, kb = self.shape
        for x in range(options):
            for a in range(ka):
                for b in range(kb):
                    prob = signal.probs[a if proposer == "A" else b, :, x]
                    values[:, a, b] += prob * responses[x].values[:, a, b]
        return Stage(signal, responses, values)

    def _merge(
        self,
        post: Array,
        candidates: dict[str, Array],
        evaluate: Callable[[str, Array], list[Response]],
        x: int,
        combo: tuple,
        t: int,
        i: int,
        mu_r: Array,
    ) -> Response:
        """Response at a per-point posterior, reusing the standard evaluations."""

        merged: Response | None = None
        covered = np.zeros(post.shape, dtype=bool)
        for key, cand in candidates.items():
            hit = ~covered & np.isclose(post, cand, atol=1e-12)
            if not hit.any():
                continue
            resp = evaluate(key, cand)[x]
            merged = resp if merged is None else _blend(merged, resp, hit)
            covered |= hit
        if not covered.all():  # semi-separating posteriors
            resp = self.respond(combo, t, i, x, post, mu_r)
            merged = resp if merged is None else _blend(merged, resp, ~covered)
        assert merged is not None
        return merged

    def _edge(
        self,
        combo: tuple,
        t: int,
        i: int,
        x: int,
        edge: Array,
        mask: Array,
        mu_r: Array,
        base: Response,
    ) -> Response:
        """The responder's mixture at a jump: weight edge[:, 2] on its response just
        above the jump, the rest on its response just below; then Bayes' rule."""

        idx = np.nonzero(mask)[0]
        lo, hi, lam = edge[idx, 0], edge[idx, 1], edge[idx, 2]
        r_lo = self.respond(combo, t, i, x, lo, mu_r[idx])
        r_hi = self.respond(combo, t, i, x, hi, mu_r[idx])
        accept = lam * r_hi.accept + (1 - lam) * r_lo.accept
        post_p = 0.5 * (lo + hi)
        _, responder = self._roles(t)
        m = mu_r[idx]
        if self.sides[responder].typed:
            reject_s, reject_w = m * (1 - accept[0]), (1 - m) * (1 - accept[1])
            total = reject_s + reject_w
            post = np.where(total > 1e-12, reject_s / np.maximum(total, 1e-300), 1.0)
        else:
            post = m.copy()
        rv, _ = self.prepare(combo, t, i, post_p)
        mixed = self._assemble(combo, t, x, accept, post, rv, m)
        return _blend(base, _scatter(mixed, idx, base), mask)

    def _node_table(self, combo: tuple, t: int, i: int) -> Array:
        ga, gb = self.sides["A"].grid, self.sides["B"].grid
        mu_a, mu_b = np.meshgrid(ga, gb, indexing="ij")
        values = self.stage(combo, t, i, mu_a.ravel(), mu_b.ravel()).values
        return values.reshape(2, *self.shape, ga.size, gb.size)

    # ------------------------------------------------------------- commitments
    def node_values(self, combo: tuple, mu_a: float, mu_b: float) -> Array:
        """(2, kA, kB) values at the first proposal, given commitments and beliefs."""

        return self._interpolate(combo, 0, 0, np.array([mu_a]), np.array([mu_b]))[..., 0]

    def commit_b(self, ca: bool, mu_a: float, mu_b: float) -> tuple[Signal, Array]:
        """B's commitment choice, and the resulting values by type pair (2, kA, kB)."""

        p = self.params
        if p.b.commitment is None:
            return _trivial_signal(mu_b), self.node_values((ca, False), mu_a, mu_b)
        b_types = self.sides["B"]
        w_a = self.sides["A"].weights(np.array([mu_a]))[:, 0]

        def value(post: Array, option: int, idx: Array | None = None) -> Array:
            pairs = self.node_values((ca, bool(option)), mu_a, float(post[0]))
            return np.asarray(np.einsum("a,ab->b", w_a, pairs[1])[:, None])

        tables = {"p": [value(np.array([mu_b]), o) for o in (0, 1)]}
        if b_types.typed:
            tables["0"] = [value(np.zeros(1), o) for o in (0, 1)]
            tables["1"] = [value(np.ones(1), o) for o in (0, 1)]
        signal = select_signal(tables, np.array([mu_b]), p.selection, value, typed=b_types.typed)
        pairs = np.zeros((2, *self.shape))
        for option in (0, 1):
            pv = self.node_values((ca, bool(option)), mu_a, float(signal.post[0, option]))
            for b in range(self.shape[1]):
                pairs[:, :, b] += signal.probs[b, 0, option] * pv[:, :, b]
        return signal, pairs

    def commit_a(self) -> tuple[Signal, Array]:
        """A's commitment choice at the priors, and the resulting values."""

        p = self.params
        mu_a, mu_b = self.sides["A"].prior, self.sides["B"].prior
        if p.a.commitment is None:
            _, pairs = self.commit_b(False, mu_a, mu_b)
            return _trivial_signal(mu_a), pairs
        a_types = self.sides["A"]
        w_b = self.sides["B"].weights(np.array([mu_b]))[:, 0]

        def value(post: Array, option: int, idx: Array | None = None) -> Array:
            _, pairs = self.commit_b(bool(option), float(post[0]), mu_b)
            return np.asarray(np.einsum("b,ab->a", w_b, pairs[0])[:, None])

        tables = {"p": [value(np.array([mu_a]), o) for o in (0, 1)]}
        if a_types.typed:
            tables["0"] = [value(np.zeros(1), o) for o in (0, 1)]
            tables["1"] = [value(np.ones(1), o) for o in (0, 1)]
        signal = select_signal(tables, np.array([mu_a]), p.selection, value, typed=a_types.typed)
        pairs = np.zeros((2, *self.shape))
        for option in (0, 1):
            _, pv = self.commit_b(bool(option), float(signal.post[0, option]), mu_b)
            for a in range(self.shape[0]):
                pairs[:, a, :] += signal.probs[a, 0, option] * pv[:, a, :]
        return signal, pairs


def _blend(base: Response, other: Response, mask: Array) -> Response:
    return Response(
        np.where(mask, other.accept, base.accept),
        np.where(mask, other.post, base.post),
        np.where(mask, other.values, base.values),
        np.where(mask, other.proposer, base.proposer),
        np.where(mask, other.fallback, base.fallback),
    )


def _scatter(part: Response, idx: Array, like: Response) -> Response:
    """Embed a response computed for points *idx* into arrays shaped like *like*."""

    def put(full: Array, sub: Array) -> Array:
        out = full.copy()
        out[..., idx] = sub
        return out

    return Response(
        put(like.accept, part.accept),
        put(like.post, part.post),
        put(like.values, part.values),
        put(like.proposer, part.proposer),
        put(like.fallback, part.fallback),
    )


def _trivial_signal(mu: float) -> Signal:
    probs = np.zeros((2, 1, 2))
    probs[:, 0, 0] = 1.0
    return Signal(probs, np.full((1, 2), mu), np.zeros(1, bool), np.zeros(1, bool), None)


def _top3(table: Array) -> tuple[Array, Array]:
    """Top three values and their option indices along axis 0 (options, n)."""

    order = np.argsort(-table, axis=0, kind="stable")[:3]
    return np.take_along_axis(table, order, axis=0), order


def _best_excluding(top_v: Array, top_i: Array, *excluded: int) -> Array:
    out = np.full(top_v.shape[1], -np.inf)
    filled = np.zeros(top_v.shape[1], dtype=bool)
    for k in range(top_v.shape[0]):
        ok = ~filled
        for e in excluded:
            ok &= top_i[k] != e
        out = np.where(ok, top_v[k], out)
        filled |= ok
    return out


def select_signal(
    tables: dict[str, list[Array]],
    mu: Array,
    selection: str,
    value_at: Callable[[Array, int, Array], Array],
    *,
    typed: bool,
) -> Signal:
    """Choose a sender's options by the stated rule (module docstring).

    ``tables[key][x]`` is each type's value (types, n) of option x when the
    observer's belief after it is ``mu`` (key ``"p"``), 0 or 1;
    ``value_at(beliefs, x, idx)`` gives the same for the points *idx* at any beliefs.
    """

    options = len(tables["p"])
    n = mu.shape[0]
    probs = np.zeros((2 if typed else 1, n, options))
    post = np.zeros((n, options))
    semi = np.zeros(n, dtype=bool)
    fallback = np.zeros(n, dtype=bool)
    edge = np.full((n, 3), np.nan)
    if not typed:
        values = np.stack([v[0] for v in tables["p"]])  # (options, n)
        best = np.argmax(values >= values.max(axis=0) - TOLERANCE, axis=0)  # earliest maximum
        probs[0, np.arange(n), best] = 1.0
        post[:] = mu[:, None]
        return Signal(probs, post, semi, fallback, edge)
    s, w = 0, 1
    v_mu = {k: np.stack([tables["p"][x][k] for x in range(options)]) for k in (s, w)}
    v_0 = {k: np.stack([tables["0"][x][k] for x in range(options)]) for k in (s, w)}
    v_1 = {k: np.stack([tables["1"][x][k] for x in range(options)]) for k in (s, w)}
    top0 = {k: _top3(v_0[k]) for k in (s, w)}
    # a strong-only offer is on the path only if the strong type has positive probability;
    # at belief 0 it is unexpected, so it is read as weakness like any other surprise
    strong_possible = mu > EPS
    best_key = np.full(n, -np.inf)
    choice_s = np.zeros(n, dtype=int)
    choice_w = np.zeros(n, dtype=int)
    bonus = 1e6 if selection == "pooling_first" else 0.0
    for xs in range(options):
        for xw in range(options):
            if xs == xw:
                on_s, on_w = v_mu[s][xs], v_mu[w][xs]
                dev_s = _best_excluding(*top0[s], xs)
                dev_w = _best_excluding(*top0[w], xs)
                key = mu * on_s + (1 - mu) * on_w + bonus
            else:
                on_s = np.where(strong_possible, v_1[s][xs], v_0[s][xs])
                on_w = v_0[w][xw]
                mimic_w = np.where(strong_possible, v_1[w][xs], v_0[w][xs])
                dev_s = _best_excluding(*top0[s], xs)
                dev_w = np.maximum(mimic_w, _best_excluding(*top0[w], xs, xw))
                key = mu * on_s + (1 - mu) * on_w
            valid = (on_s >= dev_s - TOLERANCE) & (on_w >= dev_w - TOLERANCE)
            better = valid & (key > best_key + TOLERANCE)
            best_key = np.where(better, key, best_key)
            choice_s = np.where(better, xs, choice_s)
            choice_w = np.where(better, xw, choice_w)
    found = np.isfinite(best_key)
    rows = np.arange(n)
    pooled = found & (choice_s == choice_w)
    separated = found & (choice_s != choice_w)
    probs[s, rows, choice_s] = np.where(found, 1.0, 0.0)
    probs[w, rows, choice_w] = np.where(found, 1.0, 0.0)
    post[rows, choice_s] = np.where(pooled, mu, np.where(separated & strong_possible, 1.0, 0.0))
    # off-path options keep belief 0; the weak type's separate offer has belief 0 too
    missing = ~found
    if missing.any():
        idx = np.nonzero(missing)[0]
        m_mu = mu[idx]
        best = np.full(idx.size, -np.inf)
        semi_choice = np.zeros((idx.size, 3))  # xs, xw, posterior
        semi_edge = np.full((idx.size, 3), np.nan)
        for xs in range(options):
            alt = _best_excluding(top0[w][0][:, idx], top0[w][1][:, idx], xs)
            alt_x = _argbest_excluding(v_0[w][:, idx], xs)

            def gap(v: Array, xs: int = xs, alt: Array = alt) -> Array:
                return np.asarray(value_at(v, xs, idx)[w] - alt)

            # the weak type copies the strong type's offer xs just often enough that
            # the belief after xs leaves it indifferent: gap(v*) = 0 for v* in [mu, 1]
            lo, hi = m_mu.copy(), np.ones(idx.size)
            g_lo, g_hi = gap(lo), gap(hi)
            ok = g_lo * g_hi <= TOLERANCE
            rising = g_hi >= g_lo
            for _ in range(SEMI_ITERATIONS):
                mid = 0.5 * (lo + hi)
                g_mid = gap(mid)
                below = np.where(rising, g_mid < 0, g_mid > 0)  # root lies above mid
                lo = np.where(below, mid, lo)
                hi = np.where(below, hi, mid)
            v_star = 0.5 * (lo + hi)
            strong_val = value_at(v_star, xs, idx)[s]
            # if the weak type's value jumps across its alternative (the observer's response
            # switches at v*), the observer, indifferent there, mixes its two responses with
            # the weight that leaves the weak type indifferent
            g_star = gap(v_star)
            jump = ok & (np.abs(g_star) > EDGE_TOLERANCE)
            lam = np.zeros(idx.size)
            if jump.any():
                v_lo, v_hi = value_at(lo, xs, idx), value_at(hi, xs, idx)
                g_left, g_right = v_lo[w] - alt, v_hi[w] - alt
                with np.errstate(divide="ignore", invalid="ignore"):
                    lam = np.clip(-g_left / (g_right - g_left), 0.0, 1.0)
                lam = np.where(jump, lam, 0.0)
                strong_val = np.where(jump, lam * v_hi[s] + (1 - lam) * v_lo[s], strong_val)
            dev_s = _best_excluding(top0[s][0][:, idx], top0[s][1][:, idx], xs)
            ok &= strong_val >= dev_s - TOLERANCE
            key = m_mu * strong_val + (1 - m_mu) * alt
            better = ok & (key > best + TOLERANCE)
            best = np.where(better, key, best)
            semi_choice[better] = np.stack([np.full(idx.size, xs), alt_x, v_star], axis=1)[better]
            fresh = np.stack([lo, hi, lam], axis=1)
            fresh[~jump] = np.nan
            semi_edge[better] = fresh[better]
        done = np.isfinite(best)
        for j, e in enumerate(idx):
            if done[j]:
                xs, xw, v_star = int(semi_choice[j, 0]), int(semi_choice[j, 1]), semi_choice[j, 2]
                m = mu[e]
                rho = 1.0 if v_star <= 0 else min(1.0, m * (1 - v_star) / (v_star * (1 - m)))
                probs[s, e, xs] = 1.0
                probs[w, e, xs] = rho
                probs[w, e, xw] = 1.0 - rho
                post[e, xs] = v_star
                semi[e] = True
                edge[e] = semi_edge[j]
            else:  # last resort: pool on the best ex-ante offer at belief mu
                ex_ante = m_mu[j] * v_mu[s][:, e] + (1 - m_mu[j]) * v_mu[w][:, e]
                x = int(np.argmax(ex_ante))
                probs[:, e, x] = 1.0
                post[e, x] = mu[e]
                fallback[e] = True
    return Signal(probs, post, semi, fallback, edge)


def _argbest_excluding(table: Array, excluded: int) -> Array:
    masked = table.copy()
    masked[excluded] = -np.inf
    return np.asarray(np.argmax(masked, axis=0), dtype=float)


class BeliefPolicy:
    """Plays a :class:`BeliefSolution` in live states.

    Public beliefs are rebuilt from the state's public record (commitments,
    offers, responses, contest results); each side then plays its own type's rule.
    """

    def __init__(self, params: BargainingParams) -> None:
        self.solution = BeliefSolution(params)
        self._stages: dict[tuple[object, ...], Stage] = {}
        self._responses: dict[tuple[object, ...], Response] = {}
        self._commit_b: dict[tuple[object, ...], Signal] = {}
        self._commit_a: Signal | None = None

    # ------------------------------------------------------------- memoized rules
    def commit_a(self) -> Signal:
        if self._commit_a is None:
            self._commit_a = self.solution.commit_a()[0]
        return self._commit_a

    def commit_b(self, ca: bool, mu_a: float, mu_b: float) -> Signal:
        key = (ca, round(mu_a, 12), round(mu_b, 12))
        if key not in self._commit_b:
            self._commit_b[key] = self.solution.commit_b(ca, mu_a, mu_b)[0]
        return self._commit_b[key]

    def stage(self, combo: tuple, t: int, i: int, mu_a: float, mu_b: float) -> Stage:
        key = (combo, t, i, round(mu_a, 12), round(mu_b, 12))
        if key not in self._stages:
            self._stages[key] = self.solution.stage(combo, t, i, np.array([mu_a]), np.array([mu_b]))
        return self._stages[key]

    def _corners(self, key: str, mu: float) -> list[tuple[float, float]]:
        grid = self.solution.sides[key].grid
        if grid.size == 1:
            return [(float(grid[0]), 1.0)]
        i0, i1, f = _grid_index(grid, np.array([mu]))
        corners = [(float(grid[i0[0]]), 1.0 - float(f[0])), (float(grid[i1[0]]), float(f[0]))]
        return [(g, w) for g, w in corners if w > 1e-12]

    def offers(self, combo: tuple, t: int, i: int, mu_a: float, mu_b: float) -> Array:
        """Each proposer type's offer distribution (types, options) at any beliefs.

        Off the grid, the proposer mixes the behaviour of the neighbouring grid
        points with the interpolation weights, which delivers the interpolated
        continuation values the solution was built on. After a screening rejection
        the belief lands where the proposer's choice changes, and there the
        equilibrium needs the proposer to randomize (Fudenberg and Tirole 1991,
        section 10.3). Each grid contains its side's prior, so play starts on it.
        """

        mixture: Array | None = None
        for ga, wa in self._corners("A", mu_a):
            for gb, wb in self._corners("B", mu_b):
                probs = self.stage(combo, t, i, ga, gb).signal.probs[:, 0, :]
                mixture = wa * wb * probs if mixture is None else mixture + wa * wb * probs
        assert mixture is not None
        return mixture

    def posterior(self, combo: tuple, t: int, i: int, mu_a: float, mu_b: float, x: int) -> float:
        """Belief that the proposer is strong after it offers *x* (Bayes; surprises: 0)."""

        proposer = self.solution.params.proposer(t)
        mu_p = mu_a if proposer == "A" else mu_b
        if not self.solution.sides[proposer].typed:
            return mu_p
        probs = self.offers(combo, t, i, mu_a, mu_b)
        strong, weak = mu_p * probs[0, x], (1.0 - mu_p) * probs[1, x]
        if strong + weak <= 1e-12:
            return 0.0
        return float(strong / (strong + weak))

    def response(
        self,
        combo: tuple,
        t: int,
        i: int,
        x: int,
        pre: tuple[float, float],
        mu_a: float,
        mu_b: float,
    ) -> Response:
        """The responder's rule for offer *x*; *pre* are the beliefs before the offer."""

        key = (combo, t, i, x, *(round(v, 12) for v in (*pre, mu_a, mu_b)))
        if key not in self._responses:
            proposer = self.solution.params.proposer(t)
            mu_p, mu_r = (mu_a, mu_b) if proposer == "A" else (mu_b, mu_a)
            on_grid = all(
                len(self._corners(side, v)) == 1 for side, v in (("A", pre[0]), ("B", pre[1]))
            )
            response = None
            if on_grid:  # reuse the stage's response (it includes mixtures at jumps)
                stage = self.stage(combo, t, i, *pre)
                if abs(float(stage.signal.post[0, x]) - mu_p) < 1e-9:
                    response = stage.responses[x]
            if response is None:
                response = self.solution.respond(combo, t, i, x, np.array([mu_p]), np.array([mu_r]))
            self._responses[key] = response
        return self._responses[key]

    # ------------------------------------------------------------------ tracking
    def track(
        self, state: BargainingState
    ) -> tuple[tuple[bool, bool], int, float, float, tuple[float, float]]:
        """(commitments, position, mu_A, mu_B, beliefs before the last offer)."""

        p = self.solution.params
        mu = {"A": self.solution.sides["A"].prior, "B": self.solution.sides["B"].prior}
        committed = {"A": False, "B": False}
        position = 0
        offer: int | None = None
        pre = (mu["A"], mu["B"])
        for event in state.events:
            kind = event[0]
            if kind == "commit":
                _, key, choice = event
                if key == "A":
                    signal = self.commit_a()
                else:
                    signal = self.commit_b(committed["A"], mu["A"], mu["B"])
                mu[key] = float(signal.post[0, int(choice)])
                committed[key] = bool(choice)
                continue
            combo = (committed["A"], committed["B"])
            if kind == "offer":
                _, t, offer = event
                pre = (mu["A"], mu["B"])
                mu[p.proposer(t)] = self.posterior(combo, t, position, mu["A"], mu["B"], offer)
            elif kind == "respond":
                _, t, accepted = event
                if not accepted:
                    assert offer is not None
                    response = self.response(combo, t, position, offer, pre, mu["A"], mu["B"])
                    responder = "B" if p.proposer(t) == "A" else "A"
                    mu[responder] = float(response.post[0])
            elif kind == "contest":
                position += int(event[2])
        return (committed["A"], committed["B"]), position, mu["A"], mu["B"], pre

    def action_probabilities(self, state: BargainingState) -> dict[str, float]:
        p = self.solution.params
        legal = state.legal_actions()
        if state.phase == "consent":
            assert state.offer is not None
            consent = p.consent[state.consent_index]
            choice = "ratify" if consent_ratifies(consent, p.grid[state.offer]) else "veto"
            return {a: (1.0 if a == choice else 0.0) for a in legal}
        combo, position, mu_a, mu_b, pre = self.track(state)
        key = {"commit_a": "A", "commit_b": "B"}.get(state.phase)
        if key is None:
            proposer = p.proposer(state.period)
            key = proposer if state.phase == "propose" else ("B" if proposer == "A" else "A")
        kind = 1 if state.types[key] == "weak" else 0
        if state.phase == "commit_a":
            dist = self.commit_a().probs[kind, 0]
        elif state.phase == "commit_b":
            dist = self.commit_b(combo[0], mu_a, mu_b).probs[kind, 0]
        elif state.phase == "propose":
            dist = self.offers(combo, state.period, position, mu_a, mu_b)[kind]
        else:
            assert state.offer is not None
            response = self.response(combo, state.period, position, state.offer, pre, mu_a, mu_b)
            accept = float(response.accept[kind, 0])
            return {"accept": accept, "reject": 1.0 - accept}
        total = float(dist.sum())
        return {a: float(dist[j]) / total for j, a in enumerate(legal)}


__all__ = [
    "BeliefPolicy",
    "BeliefSolution",
    "Response",
    "SideTypes",
    "Signal",
    "Stage",
    "select_signal",
]

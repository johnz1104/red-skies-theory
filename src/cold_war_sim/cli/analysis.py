"""Commands for checking goal values, uncertain inputs and historical fit."""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any

from ..campaign import assumptions
from ..campaign.catalog.core import Catalog, load_catalog
from ..campaign.decisions.policies import POLICIES, make_policy
from ..campaign.engine.core import CampaignEngine, EngineConfig, run_episode
from ..campaign.payoffs import derive, goals
from ..campaign.world import calendar
from ..campaign.world.state import POWERS, identity
from ..evaluation import goalcheck, robustness
from ..evaluation.robustness import distribution_text
from .main import _sampling_arguments

SHORT = {
    "security": "sec",
    "influence": "inf",
    "economy": "eco",
    "regime": "reg",
    "empire": "emp",
    "autonomy": "aut",
    "credibility": "cred",
}


def cmd_goals_show(args: argparse.Namespace) -> int:
    """Print each identity's goal components, weight row and score every N years."""

    catalog = load_catalog()
    config = EngineConfig(chance_mode=args.chance, npc_policy=args.npc_policy)
    engine = CampaignEngine(catalog, config)
    shown: set[int] = set()

    def snapshot(eng: CampaignEngine, _request: Any) -> None:
        world = eng.world
        year = calendar.year(world.quarter)
        if year in shown or (year - 1946) % args.every:
            return
        shown.add(year)
        print(f"{calendar.fmt(world.quarter)}")
        for power in POWERS:
            if world.powers[power].status == "eliminated":
                continue
            ident = identity(world, power)
            parts = goals.components(world, ident, catalog)
            weight_row = goals.row(world, ident, catalog)
            score = goals.combine(parts, weight_row.weights)
            values = "  ".join(f"{SHORT[k]} {v:+.2f}" for k, v in parts.items())
            print(f"  {ident:<4} {score:+.3f}  {values}  [{weight_row.label}]")

    run_episode(engine, make_policy(args.policy), args.seed, on_decision=snapshot)
    return 0


def _flatten(value: Any, path: tuple[Any, ...]) -> list[tuple[str, Any]]:
    """The leaves of a nested mapping as (dotted path, value)."""

    if isinstance(value, dict):
        return [item for k, v in value.items() for item in _flatten(v, (*path, k))]
    return [(".".join(str(k) for k in path), value)]


def _lookup(config: Any, path: tuple[Any, ...]) -> Any:
    """The value at *path* in a config, or ``None`` where it is missing."""

    for key in path:
        if isinstance(config, list) or key in config:
            config = config[key]
        else:
            return None
    return config


def _derived_event(catalog: Catalog, event_id: str) -> Any:
    """The major event *event_id*."""

    event = catalog.events.get(event_id)
    if event is None or event.major is None:
        raise LookupError(f"unknown major event {event_id!r}")
    return event


def cmd_goals_derive(args: argparse.Namespace) -> int:
    """Hand-set against goal-derived values at the event's historical start."""

    catalog = load_catalog()
    try:
        event = _derived_event(catalog, args.event)
        world, _ = goalcheck.capture_worlds(catalog, [args.event], args.seed)[args.event]
    except (LookupError, KeyError):
        print(f"{args.event} is not a major event of the historical campaign", file=sys.stderr)
        return 2
    major = event.major
    hand = major.bound_config(world)
    derived = derive.apply(catalog, event, world, copy.deepcopy(hand))
    print(
        f"{event.title} at {calendar.fmt(world.quarter)}: hand-set -> derived "
        f"(kappa {catalog.goals.kappa:g}; payoffs in the catalog: {major.payoffs})"
    )
    for slot in derive.slots(
        major.family, hand, major.tags, major.effort_costs, tuple(major.identities)
    ):
        for path, value in _flatten(_lookup(derived, slot), slot):
            keys = tuple(int(k) if k.isdigit() else k for k in path.split("."))
            before = _lookup(hand, keys)
            shown = "-" if before is None else f"{before:+.2f}"
            print(f"  {path:<44} {shown:>8}  ->  {value:+.2f}")
    return 0


def cmd_goals_sweep(args: argparse.Namespace) -> int:
    """Re-solve one event with goal settings varied; report outcome changes."""

    catalog = load_catalog()
    try:
        event = _derived_event(catalog, args.event)
        world, _ = goalcheck.capture_worlds(catalog, [args.event], args.seed)[args.event]
    except (LookupError, KeyError):
        print(f"{args.event} is not a major event of the historical campaign", file=sys.stderr)
        return 2
    major = event.major

    def solve(config: Any) -> dict[str, float]:
        bound = derive.apply(catalog, event, world, major.bound_config(world), config)
        return goalcheck.outcome_distribution(
            major.family,
            bound,
            iterations=args.iterations,
            rollouts=args.rollouts,
            temperature=args.temperature,
        )

    base = solve(catalog.goals)
    modal = max(base, key=lambda k: base[k])
    print(f"{event.title} at {calendar.fmt(world.quarter)}, derived payoffs")
    print(f"  base: {distribution_text(base)}")
    flips = 0
    for label, config in goalcheck.variations(catalog, event, world):
        dist = solve(config)
        top = max(dist, key=lambda k: dist[k])
        distance = goalcheck.total_variation(base, dist)
        mark = "FLIP " if top != modal else ("shift" if distance >= 0.2 else "     ")
        flips += top != modal
        if args.all or mark.strip():
            print(f"  {mark} {label:<34} {distribution_text(dist)}")
    print(f"  {flips} setting(s) change the most likely outcome")
    return 0


def cmd_goals_check(args: argparse.Namespace) -> int:
    """Equilibrium under derived payoffs against the historical outcome."""

    catalog = load_catalog()
    events = [
        e
        for e in sorted(catalog.events.values(), key=lambda e: (e.historical_date or 0, e.id))
        if e.major is not None
        and (e.major.payoffs == "derived" or (args.preview and e.major.tier == 1))
    ]
    if not events:
        print("no event uses derived payoffs yet; --preview checks every tier-1 event")
        return 0
    worlds = goalcheck.capture_worlds(catalog, [e.id for e in events], args.seed)
    for event in events:
        major = event.major
        assert major is not None
        if event.id not in worlds:
            print(f"  {event.id:<24} did not occur in the historical replay")
            continue
        world, _ = worlds[event.id]
        bound = derive.apply(catalog, event, world, major.bound_config(world))
        dist = goalcheck.outcome_distribution(
            major.family,
            bound,
            iterations=args.iterations,
            rollouts=args.rollouts,
            temperature=args.temperature,
        )
        history = major.historical_outcome
        top = max(dist, key=lambda k: dist[k])
        flag = " [hold-out]" if major.holdout else ""
        print(
            f"  {event.id:<24} history {history or '-':<12} p={dist.get(history or '', 0.0):.2f}"
            f"  most likely {top} ({dist[top]:.2f}){flag}"
        )
    return 0


def add_goals(sub: Any) -> None:
    """The ``goals`` commands."""

    goals_cmd = sub.add_parser("goals", help="inspect the goal functions")
    goals_sub = goals_cmd.add_subparsers(dest="goals_command", required=True)
    show = goals_sub.add_parser("show", help="goal components along one campaign")
    show.add_argument("--seed", type=int, default=0)
    show.add_argument("--policy", choices=sorted(POLICIES), default="historical")
    show.add_argument("--npc-policy", choices=sorted(POLICIES), default="historical")
    show.add_argument("--chance", choices=("sample", "historical"), default="historical")
    show.add_argument("--every", type=int, default=5, help="years between snapshots")
    show.set_defaults(func=cmd_goals_show)
    derive_cmd = goals_sub.add_parser("derive", help="hand-set against derived payoffs")
    derive_cmd.add_argument("--event", required=True)
    derive_cmd.add_argument("--seed", type=int, default=0)
    derive_cmd.set_defaults(func=cmd_goals_derive)
    sweep = goals_sub.add_parser("sweep", help="sensitivity of one event to goal settings")
    sweep.add_argument("--event", required=True)
    sweep.add_argument("--seed", type=int, default=0)
    sweep.add_argument("--iterations", type=int, default=300)
    sweep.add_argument("--rollouts", type=int, default=300)
    sweep.add_argument("--all", action="store_true", help="print every setting, not only changes")
    sweep.add_argument("--temperature", type=float, default=goalcheck.DEFAULT_TEMPERATURE)
    sweep.set_defaults(func=cmd_goals_sweep)
    check = goals_sub.add_parser("check", help="equilibrium against history for derived events")
    check.add_argument("--preview", action="store_true", help="also derive every tier-1 event")
    check.add_argument("--seed", type=int, default=0)
    check.add_argument("--iterations", type=int, default=300)
    check.add_argument("--rollouts", type=int, default=300)
    check.add_argument("--temperature", type=float, default=goalcheck.DEFAULT_TEMPERATURE)
    check.set_defaults(func=cmd_goals_check)


def cmd_assumptions_check(args: argparse.Namespace) -> int:
    """Coverage, entry values, ranges, and the catalog with every live number at an end."""

    catalog = load_catalog()
    problems = assumptions.check(catalog, extremes=not args.quick)
    current = assumptions.plan_for(catalog)
    live = current.live()
    print(
        f"register: {len(current.items)} numbers, {len(live)} live and drawn, "
        f"{sum(1 for a in current.items if a.curated)} curated; "
        + ("ok" if not problems else f"{len(problems)} problem(s)")
    )
    for problem in problems:
        print(f"  {problem}")
    return 1 if problems else 0


def cmd_assumptions_list(args: argparse.Namespace) -> int:
    """List the numbers with their basis, range and liveness (optionally one event's)."""

    catalog = load_catalog()
    current = assumptions.plan_for(catalog, args.playable_tier)
    for item in current.items:
        if args.event and not item.address.startswith(f"event:{args.event}/"):
            continue
        if args.group and item.group != args.group:
            continue
        if args.basis and item.basis != args.basis:
            continue
        if args.placeholder and not item.placeholder:
            continue
        if args.match and not assumptions.Sampling(select=(args.match,)).selects(item):
            continue
        status = current.status[item.address]
        if item.fixed:
            span = f"fixed:{item.fixed}"
        elif status != "live":
            span = status
        else:
            span = f"{item.low:.3g} .. {item.high:.3g}" if item.low is not None else "-"
        mark = "*" if item.curated else " "
        flag = " (placeholder)" if item.placeholder else ""
        print(f"{mark} {item.address}  {item.value:g}  [{span}]  {item.kind}, {item.basis}{flag}")
        if args.notes and (item.source or item.note):
            print(f"      {'; '.join(t for t in (item.source, item.note) if t)}")
    return 0


def cmd_assumptions_doc(args: argparse.Namespace) -> int:
    """Render the register as Markdown."""

    text = assumptions.render_markdown(load_catalog())
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


def cmd_assumptions_sweep(args: argparse.Namespace) -> int:
    """Move one event's numbers to their range ends and rank them by effect."""

    catalog = load_catalog()
    event = catalog.events.get(args.event)
    if event is None or event.major is None:
        print(f"unknown major event {args.event!r}", file=sys.stderr)
        return 2
    start = time.time()
    report = robustness.event_sweep(
        catalog,
        args.event,
        seed=args.seed,
        iterations=args.iterations,
        rollouts=args.rollouts,
        temperature=args.temperature,
        select=tuple(t for t in (args.select or "").split(",") if t),
        progress=lambda text: print(text, file=sys.stderr),
    )
    print("\n".join(report.text(args.top)))
    print(f"({time.time() - start:.0f} s)", file=sys.stderr)
    return 0


def _run_spec(args: argparse.Namespace) -> robustness.RunSpec:
    """The run spec of a robustness command's arguments."""

    return robustness.RunSpec(
        policy=args.policy,
        npc_policy=args.npc_policy,
        playable_tier=args.playable_tier,
        temperature=args.temperature,
        select=tuple(t for t in (args.select or "").split(",") if t),
        spread=args.spread,
    )


def cmd_assumptions_robust(args: argparse.Namespace) -> int:
    """Measure the findings over draws of campaigns (or over saved ones)."""

    findings = robustness.load_findings()
    if args.findings:
        wanted = set(args.findings.split(","))
        findings = [f for f in findings if f.id in wanted]
    start = time.time()
    spec, draws, seeds, runs = _run_spec(args), args.draws, args.seeds, None
    if args.load:
        spec, draws, seeds, runs = robustness.load_runs(Path(args.load))
    report = robustness.robust(
        load_catalog(),
        findings,
        draws=draws,
        seeds=seeds,
        spec=spec,
        workers=args.workers,
        confirm=not args.no_confirm,
        by_group=args.by_group,
        save=Path(args.save) if args.save else None,
        runs=runs,
        progress=lambda text: print(text, file=sys.stderr),
    )
    print("\n".join(report.text()))
    print(f"({time.time() - start:.0f} s)", file=sys.stderr)
    return 0


def cmd_assumptions_history(args: argparse.Namespace) -> int:
    """Replay history under draws."""

    start = time.time()
    report = robustness.history(
        load_catalog(),
        draws=args.draws,
        spec=_run_spec(args),
        workers=args.workers,
        by_group=not args.joint_only,
    )
    print("\n".join(report.text()))
    print(f"({time.time() - start:.0f} s)", file=sys.stderr)
    return 0


def add_assumptions(sub: Any) -> None:
    """The ``assumptions`` commands."""

    register = sub.add_parser("assumptions", help="the assumptions register and its sweeps")
    register_sub = register.add_subparsers(dest="assumptions_command", required=True)
    reg_check = register_sub.add_parser("check", help="validate the register against the data")
    reg_check.add_argument("--quick", action="store_true", help="skip the range-end compiles")
    reg_check.set_defaults(func=cmd_assumptions_check)
    reg_list = register_sub.add_parser("list", help="list numbers with their ranges")
    reg_list.add_argument("--event")
    reg_list.add_argument("--group")
    reg_list.add_argument("--basis", choices=assumptions.BASES)
    reg_list.add_argument("--match", help="an address glob")
    reg_list.add_argument("--placeholder", action="store_true")
    reg_list.add_argument("--notes", action="store_true", help="print sources and notes")
    reg_list.add_argument("--playable-tier", type=int)
    reg_list.set_defaults(func=cmd_assumptions_list)
    reg_doc = register_sub.add_parser("doc", help="render docs/ASSUMPTIONS.md")
    reg_doc.add_argument("--output")
    reg_doc.set_defaults(func=cmd_assumptions_doc)
    reg_sweep = register_sub.add_parser("sweep", help="one event's numbers at their range ends")
    reg_sweep.add_argument("--event", required=True)
    reg_sweep.add_argument("--seed", type=int, default=0)
    reg_sweep.add_argument("--iterations", type=int, default=300)
    reg_sweep.add_argument("--rollouts", type=int, default=300)
    reg_sweep.add_argument("--temperature", type=float, default=goalcheck.DEFAULT_TEMPERATURE)
    reg_sweep.add_argument("--select", help="restrict the candidates (groups, events, globs)")
    reg_sweep.add_argument("--top", type=int, default=25)
    reg_sweep.set_defaults(func=cmd_assumptions_sweep)
    reg_robust = register_sub.add_parser("robust", help="findings across draws: robust or fragile")
    reg_robust.add_argument("--draws", type=int, default=20)
    reg_robust.add_argument("--seeds", type=int, default=5, help="campaigns per draw")
    reg_robust.add_argument("--policy", choices=sorted(POLICIES), default="equilibrium")
    reg_robust.add_argument("--npc-policy", choices=sorted(POLICIES), default="equilibrium")
    reg_robust.add_argument("--findings", help="comma-separated finding ids (default: all)")
    reg_robust.add_argument("--no-confirm", action="store_true", help="skip the suspects' runs")
    reg_robust.add_argument(
        "--by-group",
        action="store_true",
        help="for fragile findings, also draw one group at a time (groups x draws x seeds)",
    )
    reg_robust.add_argument("--save", help="write every campaign's summary to this JSON file")
    reg_robust.add_argument(
        "--load", help="re-measure the campaigns saved by --save instead of playing them"
    )
    _sampling_arguments(reg_robust)
    reg_robust.set_defaults(func=cmd_assumptions_robust)
    reg_history = register_sub.add_parser("history", help="the historical replay under draws")
    reg_history.add_argument("--draws", type=int, default=50)
    reg_history.add_argument(
        "--joint-only", action="store_true", help="skip drawing each group alone"
    )
    reg_history.set_defaults(policy="historical", npc_policy="historical")
    _sampling_arguments(reg_history)
    reg_history.set_defaults(func=cmd_assumptions_history)

"""Command line: run campaigns, inspect the catalog and solve event games."""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ..campaign import assumptions
from ..campaign.catalog.core import Catalog, CatalogError, load_catalog
from ..campaign.catalog.markdown import render_markdown
from ..campaign.decisions.policies import POLICIES, make_policy
from ..campaign.engine.core import CampaignEngine, EngineConfig, reference_tree_policy, run_episode
from ..campaign.engine.replay import episode_record, write_json
from ..campaign.world import calendar
from ..campaign.world.state import POWERS
from ..evaluation import audit, branches, goalcheck
from ..families import build_family
from ..games import build_tree, outcome_distribution
from ..games.solvers import nash_conv


def _seeds(args: argparse.Namespace) -> list[int]:
    """The seeds of ``--seeds`` (an inclusive range such as ``1-50``), else ``--seed``."""

    if args.seeds:
        start, _, end = args.seeds.partition("-")
        return list(range(int(start), int(end or start) + 1))
    return [args.seed]


def _sampling(args: argparse.Namespace) -> assumptions.Sampling | None:
    """The sampling of ``--sample`` (all live numbers, or a selection), if given."""

    if args.sample is None:
        return None
    select = tuple(t for t in args.sample.split(",") if t)
    return assumptions.Sampling(select=select, spread=args.spread, draw=args.draw)


def capture_config(catalog: Catalog, event_id: str, seed: int) -> dict[str, Any]:
    """Replay the historical campaign until *event_id* starts; return its bound config."""

    engine = CampaignEngine(catalog, EngineConfig(chance_mode="historical"))
    policy = make_policy("historical")
    rng = np.random.default_rng(seed)
    engine.reset(seed)
    while not engine.done:
        active = engine.s.active.get(event_id)
        if active is not None:
            return active.config
        request = engine.pending()
        assert request is not None
        engine.act(policy.choose(engine, request, rng))
    raise LookupError(f"{event_id} did not occur in the historical campaign")


def _sampling_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the draw-selection arguments shared by the robustness commands."""

    parser.add_argument(
        "--select", help="comma-separated groups, events (event:id) or address globs to draw"
    )
    parser.add_argument("--spread", type=float, default=1.0, help="share of each range used")
    parser.add_argument("--playable-tier", type=int)
    parser.add_argument("--temperature", type=float, default=goalcheck.DEFAULT_TEMPERATURE)
    parser.add_argument("--workers", type=int)


def cmd_catalog(args: argparse.Namespace) -> int:
    """Validate the catalog and print its counts, or render it as Markdown."""

    try:
        catalog = load_catalog()
    except CatalogError as exc:
        print(f"catalog error: {exc}", file=sys.stderr)
        return 2
    if args.markdown:
        text = render_markdown(catalog)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
        else:
            print(text, end="")
        return 0
    counts = catalog.counts()
    print(
        f"catalog ok: {counts['major']} major, {counts['cutscene']} cutscenes, "
        f"{counts['leadership']} leadership, {counts['decision']} decision; "
        f"{len(catalog.leaders)} leaders; "
        f"{len(catalog.observed_flags)} observed flags"
    )
    return 0


def cmd_generator_fit(args: argparse.Namespace) -> int:
    """Fit the generator's rates so named plus generated crises match the ICB marginals."""

    import yaml

    from ..campaign.events import genfit

    catalog = load_catalog()
    assert catalog.raw is not None and catalog.raw.generator
    start, _, end = args.seeds.partition("-")
    seeds = list(range(int(start), int(end or start) + 1))

    def rates(doc: dict[str, Any]) -> str:
        fitted = {
            t: {k: spec[k] for k in ("rate", "group_factor", "decade_factor")}
            for t, spec in doc["templates"].items()
            if t in doc["targets"]
        }
        return str(yaml.safe_dump(fitted, sort_keys=False, default_flow_style=None))

    def keep(round_: int, doc: dict[str, Any]) -> None:  # each round's values, as they come
        print(f"== values after round {round_}\n{rates(doc)}", flush=True)
        if args.output:
            Path(args.output).write_text(yaml.safe_dump(doc, sort_keys=False))

    fitted = genfit.fit(
        catalog.raw.generator,
        seeds,
        args.rounds,
        args.workers,
        progress=lambda message: print(message, flush=True),
        power=args.power,
        on_round=keep,
    )
    print(rates(fitted))
    if args.output:
        Path(args.output).write_text(yaml.safe_dump(fitted, sort_keys=False))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Play campaigns with a policy and print the timeline (one seed) or a summary."""

    agents = tuple(args.agents.split(",")) if args.agents else POWERS
    config = EngineConfig(
        agents=agents,
        chance_mode=args.chance,
        npc_policy=args.npc_policy,
        playable_tier=args.playable_tier,
        reward=args.reward,
        baseline_temperature=args.temperature,
        sampling=_sampling(args),
        generator=args.generator,
    )
    engine = CampaignEngine(load_catalog(), config)
    policy = make_policy(args.policy)
    seeds = _seeds(args)
    reasons: Counter[str] = Counter()
    durations = []
    decisions = []
    for seed in seeds:
        started = time.perf_counter()
        result = run_episode(engine, policy, seed)
        durations.append(time.perf_counter() - started)
        decisions.append(result.decisions)
        reasons[result.end_reason] += 1
        if len(seeds) == 1 and not args.quiet:
            for date, event_id, text in result.timeline:
                print(f"{date}  {event_id:<32} {text}")
            print()
            print(
                f"end: {result.end_reason} at {result.final_quarter}; decisions {result.decisions}"
            )
            print("returns: " + ", ".join(f"{p} {v:+.3f}" for p, v in result.returns.items()))
        if args.output:
            path = Path(args.output)
            if len(seeds) > 1:
                path = path.with_name(f"{path.stem}-{seed}{path.suffix}")
            write_json(path, episode_record(engine, result))
    if len(seeds) > 1 or args.quiet:
        mean_time = sum(durations) / len(durations)
        print(
            f"episodes: {len(seeds)}; mean seconds/episode {mean_time:.3f}; "
            f"mean decisions {sum(decisions) / len(decisions):.0f}"
        )
        for reason, count in reasons.most_common():
            print(f"  {reason:<24} {count}")
    return 0


def cmd_solve(args: argparse.Namespace) -> int:
    """Solve one major event bound at its historical world and print its outcome distribution."""

    catalog = load_catalog()
    event = catalog.events.get(args.event)
    if event is None or event.major is None:
        print(f"unknown major event {args.event!r}", file=sys.stderr)
        return 2
    try:
        config = capture_config(catalog, args.event, args.seed)
    except LookupError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    family = event.major.family
    print(f"{event.title} ({family}) bound at the historical world")
    if family in ("bargaining", "attrition"):
        print(
            "baseline: dynamic programming or the belief solver (bargaining), lookahead "
            f"(attrition); temperature {args.temperature:g}; 300 rollouts"
        )
        dist = goalcheck.outcome_distribution(family, config, temperature=args.temperature)
        for label, prob in dist.items():
            print(f"  {label:<16} {prob:.3f}")
        return 0
    game = build_family(family, config)
    root = game.solver_state() if hasattr(game, "solver_state") else game.new_initial_state()
    tree = build_tree(root)
    policy = reference_tree_policy(game, tree, args.iterations, args.temperature)
    if args.temperature > 0.0:
        print(f"solver: logit quantal response (temperature {args.temperature:g})")
    elif tree.is_perfect_information:
        print("solver: backward induction")
    else:
        print(f"solver: CFR+ ({args.iterations} iterations, trembles 0.01)")
    beliefs = getattr(game, "has_beliefs", False)
    if beliefs:
        print("own beliefs: each role plays its part of its own view's solution")
    print(f"nodes {len(tree.nodes)}, information sets {len(tree.infosets)}")
    label = "NashConv in the true game (subjective play)" if beliefs else "NashConv"
    print(f"{label} {nash_conv(tree, policy).nash_conv:.6f}")
    for label, prob in outcome_distribution(tree, policy).items():
        print(f"  {label:<16} {prob:.3f}")
    return 0


def cmd_standing(args: argparse.Namespace) -> int:
    """The standing scoreboard along one campaign (never part of any reward)."""

    catalog = load_catalog()
    config = EngineConfig(chance_mode=args.chance, npc_policy=args.npc_policy)
    engine = CampaignEngine(catalog, config)
    result = run_episode(engine, make_policy(args.policy), args.seed)
    start = catalog.goals.standing_start
    for quarter, event_id, changes in engine.s.standing_log:
        moves = ", ".join(f"{who} {change:+.1f}" for who, change in sorted(changes.items()))
        print(f"{calendar.fmt(quarter)}  {event_id:<28} {moves}")
    print()
    print(f"end: {result.end_reason} at {result.final_quarter}; standing (start {start:g}):")
    for who, rating in sorted(result.standing.items(), key=lambda kv: -kv[1]):
        print(f"  {who:<5} {rating:7.1f}")
    return 0


def cmd_branches(args: argparse.Namespace) -> int:
    """Branch audit over many campaigns: fire rates, paths, stuck flags, dead events."""

    catalog = load_catalog()
    events = args.events.split(",") if args.events else None
    unknown = [e for e in events or () if e not in catalog.events]
    if unknown:
        print(f"unknown events: {unknown}", file=sys.stderr)
        return 2
    report = branches.audit(
        catalog,
        _seeds(args),
        policy=args.policy,
        npc_policy=args.npc_policy,
        temperature=args.temperature,
        events=events,
    )
    print("\n".join(report.lines(catalog)))
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Strategy-layer interest audit: scripted doctrines against random co-players."""

    seeds = _seeds(args)
    jobs = audit.job_count(seeds, args.profiles)
    print(f"{jobs} campaigns on {args.workers or 'all but one'} workers ...", file=sys.stderr)
    start = time.time()
    report = audit.audit_strategy(
        load_catalog(),
        seeds,
        profiles=args.profiles,
        workers=args.workers,
        temperature=args.temperature,
    )
    print("\n".join(report.lines()))
    print(f"({time.time() - start:.0f} s)", file=sys.stderr)
    return 0


def add_catalog(sub: Any) -> None:
    """The ``catalog`` command."""

    catalog = sub.add_parser("catalog", help="validate or render the event catalog")
    catalog.add_argument("--check", action="store_true", help="validate and print counts (default)")
    catalog.add_argument("--markdown", action="store_true", help="render the catalog as Markdown")
    catalog.add_argument("--output", help="write Markdown to this path")
    catalog.set_defaults(func=cmd_catalog)


def add_run(sub: Any) -> None:
    """The ``run`` command."""

    run = sub.add_parser("run", help="play campaigns with a baseline policy")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--seeds", help="inclusive range such as 1-50")
    run.add_argument("--policy", choices=sorted(POLICIES), default="historical")
    run.add_argument("--npc-policy", choices=sorted(POLICIES), default="historical")
    run.add_argument("--chance", choices=("sample", "historical"), default="sample")
    run.add_argument(
        "--generator", action="store_true", help="crises where the world makes them likely"
    )
    run.add_argument("--agents", help="comma-separated agent powers (default: all five)")
    run.add_argument(
        "--playable-tier",
        type=int,
        help="only major events up to this tier are playable; the rest run on rails",
    )
    run.add_argument(
        "--reward",
        choices=("goals", "score"),
        default="goals",
        help="goal functions (default) or the pre-goals world score",
    )
    run.add_argument(
        "--temperature",
        type=float,
        default=goalcheck.DEFAULT_TEMPERATURE,
        help="logit temperature of computer-run reference play (0 = equilibrium: "
        "backward induction, or CFR+'s average)",
    )
    run.add_argument(
        "--sample",
        nargs="?",
        const="",
        help="draw numbers within the assumptions register's ranges (optionally only these "
        "comma-separated groups, events or address globs)",
    )
    run.add_argument("--spread", type=float, default=1.0, help="share of each range used")
    run.add_argument("--draw", type=int, help="one fixed draw for every episode")
    run.add_argument("--output", help="write the episode record (JSON)")
    run.add_argument("--quiet", action="store_true")
    run.set_defaults(func=cmd_run)


def add_solve(sub: Any) -> None:
    """The ``solve`` command."""

    solve = sub.add_parser("solve", help="solve one major event bound at the historical world")
    solve.add_argument("--event", required=True)
    solve.add_argument("--seed", type=int, default=0)
    solve.add_argument("--iterations", type=int, default=1000)
    solve.add_argument("--temperature", type=float, default=goalcheck.DEFAULT_TEMPERATURE)
    solve.set_defaults(func=cmd_solve)


def add_standing(sub: Any) -> None:
    """The ``standing`` command."""

    board = sub.add_parser("standing", help="the standing scoreboard along one campaign")
    board.add_argument("--seed", type=int, default=0)
    board.add_argument("--policy", choices=sorted(POLICIES), default="historical")
    board.add_argument("--npc-policy", choices=sorted(POLICIES), default="historical")
    board.add_argument("--chance", choices=("sample", "historical"), default="historical")
    board.set_defaults(func=cmd_standing)


def add_branches(sub: Any) -> None:
    """The ``branches`` command."""

    audit = sub.add_parser("branches", help="branch audit: paths, stuck flags, dead events")
    audit.add_argument("--seed", type=int, default=0)
    audit.add_argument("--seeds", help="inclusive range such as 0-59")
    audit.add_argument("--policy", choices=sorted(POLICIES), default="equilibrium")
    audit.add_argument("--npc-policy", choices=sorted(POLICIES), default="equilibrium")
    audit.add_argument("--events", help="comma-separated events whose paths to list")
    audit.add_argument("--temperature", type=float, default=goalcheck.DEFAULT_TEMPERATURE)
    audit.set_defaults(func=cmd_branches)


def add_audit(sub: Any) -> None:
    """The ``audit`` command."""

    strategy = sub.add_parser("audit", help="interest audit: does a yearly strategy dominate?")
    strategy.add_argument("layer", choices=["strategy"])
    strategy.add_argument("--seed", type=int, default=0)
    strategy.add_argument("--seeds", default="0-1", help="inclusive range such as 0-1")
    strategy.add_argument("--profiles", type=int, default=20)
    strategy.add_argument("--workers", type=int)
    strategy.add_argument("--temperature", type=float, default=goalcheck.DEFAULT_TEMPERATURE)
    strategy.set_defaults(func=cmd_audit)


def add_generator(sub: Any) -> None:
    """The ``generator`` commands."""

    gen = sub.add_parser("generator", help="the crisis generator")
    gen_sub = gen.add_subparsers(dest="generator_command", required=True)
    gen_fit = gen_sub.add_parser("fit", help="fit the base rates to the ICB crisis data")
    gen_fit.add_argument("--seeds", default="0-27", help="campaign seeds, an inclusive range")
    gen_fit.add_argument("--rounds", type=int, default=2)
    gen_fit.add_argument("--workers", type=int, default=4)
    gen_fit.add_argument("--output", help="write the fitted generator.yaml here")
    gen_fit.add_argument(
        "--power", type=float, default=0.5, help="the step's exponent (0.5 damped, 1 full)"
    )
    gen_fit.set_defaults(func=cmd_generator_fit)


def build_parser() -> argparse.ArgumentParser:
    """The parser of every command; each sets ``func`` to its handler."""

    from . import analysis as cli_analysis
    from . import rl as cli_rl

    # Keep related command groups together in --help.
    commands = (
        add_catalog,
        add_run,
        add_solve,
        cli_analysis.add_goals,
        add_standing,
        add_branches,
        add_audit,
        cli_analysis.add_assumptions,
        add_generator,
        cli_rl.add_rl,
    )
    parser = argparse.ArgumentParser(prog="cold-war-campaign", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for add in commands:
        add(sub)
    return parser


_RL_MODULES = ("cold_war_sim.rl", "torch", "pettingzoo", "gymnasium")


def main(argv: list[str] | None = None) -> int:
    """Run the command line; the rl commands explain a missing RL install."""

    args = build_parser().parse_args(argv)
    try:
        status: int = args.func(args)
    except ModuleNotFoundError as error:
        missing = error.name or ""
        if args.command != "rl" or not missing.startswith(_RL_MODULES):
            raise
        if missing.startswith(_RL_MODULES[0]):
            reason = "the RL package (cold_war_sim/rl) is not in this checkout"
        else:
            reason = f"it needs {missing} (the RL dependencies: torch, pettingzoo, gymnasium)"
        raise SystemExit(f"rl {args.rl_command}: {reason}") from error
    return status


__all__ = ["build_parser", "capture_config", "main"]

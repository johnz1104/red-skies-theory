"""Command line: run campaigns, check or render the catalog, solve an event instance.

Examples::

    python -m cold_war_sim.campaign catalog --check
    python -m cold_war_sim.campaign run --seed 7 --policy historical --chance historical
    python -m cold_war_sim.campaign run --seeds 1-50 --policy random
    python -m cold_war_sim.campaign solve --event cuban_missile_crisis
    python -m cold_war_sim.campaign goals show --every 5
    python -m cold_war_sim.campaign goals derive --event cuban_missile_crisis
    python -m cold_war_sim.campaign goals sweep --event cuban_missile_crisis
    python -m cold_war_sim.campaign goals check --preview
    python -m cold_war_sim.campaign standing --policy historical --chance historical
    python -m cold_war_sim.campaign branches --seeds 0-59 --events korean_war,chinese_civil_war
    python -m cold_war_sim.campaign run --seeds 1-20 --policy random --sample
    python -m cold_war_sim.campaign assumptions check
    python -m cold_war_sim.campaign assumptions list --event suez_crisis
    python -m cold_war_sim.campaign assumptions doc --output docs/ASSUMPTIONS.md
    python -m cold_war_sim.campaign assumptions sweep --event suez_crisis
    python -m cold_war_sim.campaign assumptions robust --draws 20 --seeds 5
    python -m cold_war_sim.campaign assumptions history --draws 50
    python -m cold_war_sim.campaign rl entries --output runs/entries/tier1_historical.pkl
    python -m cold_war_sim.campaign rl train --entries runs/entries/tier1_historical.pkl --out runs/x
    python -m cold_war_sim.campaign rl eval --checkpoint runs/x/checkpoint.pt
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ..families import build_family
from ..games import build_tree, outcome_distribution
from ..games.algorithms import nash_conv
from . import assumptions, audit, branches, calendar, derive, goalcheck, goals, robustness
from .catalog import Catalog, CatalogError, load_catalog
from .catalog_doc import render_markdown
from .engine import CampaignEngine, EngineConfig, reference_tree_policy, run_episode
from .policies import POLICIES, make_policy
from .replay import episode_record, write_json
from .state import IDENTITIES, POWERS, identity


def _seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        start, _, end = args.seeds.partition("-")
        return list(range(int(start), int(end or start) + 1))
    return [args.seed]


def cmd_catalog(args: argparse.Namespace) -> int:
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

    from . import genfit

    catalog = load_catalog()
    assert catalog.raw is not None and catalog.raw.generator
    start, _, end = args.seeds.partition("-")
    seeds = list(range(int(start), int(end or start) + 1))
    fitted = genfit.fit(
        catalog.raw.generator,
        seeds,
        args.rounds,
        args.workers,
        progress=lambda message: print(message, flush=True),
        power=args.power,
    )
    rates = {
        t: {k: spec[k] for k in ("rate", "group_factor", "decade_factor")}
        for t, spec in fitted["templates"].items()
        if t in fitted["targets"]
    }
    print(yaml.safe_dump(rates, sort_keys=False, default_flow_style=None))
    if args.output:
        Path(args.output).write_text(yaml.safe_dump(fitted, sort_keys=False))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
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


def _sampling(args: argparse.Namespace) -> assumptions.Sampling | None:
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


def cmd_solve(args: argparse.Namespace) -> int:
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
    if isinstance(value, dict):
        return [item for k, v in value.items() for item in _flatten(v, (*path, k))]
    return [(".".join(str(k) for k in path), value)]


def _lookup(config: Any, path: tuple[Any, ...]) -> Any:
    for key in path:
        if isinstance(config, list) or key in config:
            config = config[key]
        else:
            return None
    return config


def _derived_event(catalog: Catalog, event_id: str) -> Any:
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


def _distribution_text(dist: dict[str, float]) -> str:
    return ", ".join(
        f"{k} {v:.2f}" for k, v in sorted(dist.items(), key=lambda kv: -kv[1]) if v >= 0.005
    )


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
    print(f"  base: {_distribution_text(base)}")
    flips = 0
    for label, config in goalcheck.variations(catalog, event, world):
        dist = solve(config)
        top = max(dist, key=lambda k: dist[k])
        distance = goalcheck.total_variation(base, dist)
        mark = "FLIP " if top != modal else ("shift" if distance >= 0.2 else "     ")
        flips += top != modal
        if args.all or mark.strip():
            print(f"  {mark} {label:<34} {_distribution_text(dist)}")
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


def _rl_events(catalog: Catalog, spec: str | None) -> list[str]:
    """``tier1`` (the default): every tier-1 major event; otherwise a comma-separated list."""

    if not spec or spec == "tier1":
        return [e for e in catalog.major_ids if (m := catalog.events[e].major) and m.tier == 1]
    events = [e.strip() for e in spec.split(",") if e.strip()]
    unknown = [e for e in events if e not in catalog.events or catalog.events[e].major is None]
    if unknown:
        raise SystemExit(f"not major events: {unknown}")
    return events


def cmd_rl_entries(args: argparse.Namespace) -> int:
    """Capture scenario entry states and save them for training."""

    from . import scenario

    catalog = load_catalog()
    events = _rl_events(catalog, args.events)
    start = time.time()
    if args.source == "historical":
        entries = scenario.capture_entries(catalog, events, playable_tier=args.playable_tier)
    else:
        entries = scenario.capture_campaign_entries(
            events, _seeds(args), workers=args.workers, playable_tier=args.playable_tier
        )
    if args.merge:
        entries = scenario.merge_entries(scenario.load_entries(Path(args.merge)), entries)
    scenario.save_entries(Path(args.output), entries)
    for event_id, items in sorted(entries.items()):
        quarters = Counter(calendar.fmt(e.quarter) for e in items)
        sources = Counter(e.source for e in items)
        print(f"  {event_id:<24} {len(items):4d} entries  {dict(sources)}  {dict(quarters)}")
    print(f"saved to {args.output} ({time.time() - start:.0f} s)", file=sys.stderr)
    return 0


def _tree_scores(
    catalog: Catalog, entries: Any, policy: Any, temperature: float | None = None
) -> dict[str, Any]:
    from .rl import evaluate as rl_evaluate

    scored: dict[str, Any] = {}
    for entry in rl_evaluate.tree_entries(catalog, entries):
        if entry.source != "historical" and any(
            e.source == "historical" for e in entries[entry.event_id]
        ):
            continue  # score at the historical entry when there is one
        if entry.event_id in scored:
            continue
        report = rl_evaluate.tree_report(catalog, entry, policy, qre_temperature=temperature)
        scored[entry.event_id] = {
            name: {
                "exploitability": round(score.exploitability, 5),
                "worst_regret": round(score.credibility, 5),
                "qre_distance": None
                if score.qre_distance is None
                else round(score.qre_distance, 4),
                "p_historical": round(score.historical, 3),
                "outcomes": {k: round(v, 3) for k, v in score.outcomes.items() if v >= 0.005},
            }
            for name, score in report.scores.items()
        }
    return scored


def cmd_rl_train(args: argparse.Namespace) -> int:
    """Masked-PPO self-play on single-event scenarios."""

    from .rl import ppo

    catalog = load_catalog()
    events = tuple(_rl_events(catalog, args.events)) if args.events else ()
    if args.mode == "scenario" and not args.entries:
        raise SystemExit("rl train: scenario mode needs --entries")
    config = ppo.PPOConfig(
        entries=args.entries or "",
        events=events,
        iterations=args.iterations,
        episodes=args.episodes,
        workers=args.workers,
        hidden=args.hidden,
        lr=args.lr,
        entropy=args.entropy,
        seed=args.seed,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        objective=args.objective,
        temperature=args.temperature,
        temperature_start=args.temperature_start,
        anneal=args.anneal,
        lr_final=args.lr_final,
        lam=args.lam,
        gamma=args.gamma,
        mode=args.mode,
        generator=args.generator,
        league=args.league,
        league_share=args.league_share,
    )
    evaluate = None
    if config.mode == "scenario":
        entries = ppo.scenario_entries(config)

        def evaluate(policy: Any) -> dict[str, Any]:
            return _tree_scores(catalog, entries, policy, config.temperature)

    ppo.train(
        config,
        Path(args.out),
        resume=Path(args.resume) if args.resume else None,
        progress=lambda message: print(message, file=sys.stderr),
        evaluate=evaluate,
    )
    return 0


def cmd_rl_br(args: argparse.Namespace) -> int:
    """An identity's approximate best response to a checkpoint, and its gain."""

    from .rl import br

    result = br.best_response(
        Path(args.checkpoint),
        args.identity,
        Path(args.out),
        iterations=args.iterations,
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        workers=args.workers,
        seed=args.seed,
        progress=lambda message: print(message, file=sys.stderr),
    )
    print(
        f"{result['identity']}: gain {result['gain']:+.4f} goal units "
        f"({result['gain_payoff']:+.2f} payoff units), standard error {result['se']:.4f}, "
        f"over {result['episodes']} episodes (learned {result['learned']:+.4f}, "
        f"best response {result['responder']:+.4f})"
    )
    return 0


def cmd_rl_eval(args: argparse.Namespace) -> int:
    """Score a checkpoint: exploitability on tree events; outcomes against reference play."""

    from . import scenario
    from .rl import evaluate as rl_evaluate
    from .rl import ppo

    learner = ppo.Learner.load(Path(args.checkpoint))
    policy = learner.policy()
    catalog = load_catalog()
    entries = scenario.load_entries(Path(args.entries or learner.config.entries))
    events = _rl_events(catalog, args.events) if args.events else sorted(entries)
    entries = {e: entries[e] for e in events if e in entries}
    for entry in rl_evaluate.tree_entries(catalog, entries):
        if args.all_entries or entry.source == "historical":
            report = rl_evaluate.tree_report(
                catalog, entry, policy, qre_temperature=learner.config.temperature
            )
            print("\n".join(report.text()))
    trees = {e.event_id for e in rl_evaluate.tree_entries(catalog, entries)}
    others = {e: v for e, v in entries.items() if e not in trees}
    if others:
        learned_env = ppo.make_env(replace(learner.config, events=tuple(sorted(others))), catalog)
        reference_env = scenario.ScenarioEnv(
            others, catalog, EngineConfig(npc_policy="equilibrium", playable_tier=1, agents=())
        )
        for event_id in sorted(others):
            label = catalog.events[event_id].major.historical_outcome  # type: ignore[union-attr]
            learned: Counter[str] = Counter()
            reference: Counter[str] = Counter()
            for seed in range(args.episodes):
                rng = np.random.default_rng(seed + 7919)
                episode = ppo.play_episode(learned_env, policy, seed, rng, event=event_id)
                learned[str(episode.outcome)] += 1
                reference_env.reset(seed, event=event_id)
                reference[str(reference_env.outcome())] += 1
            n = args.episodes
            for name, counts in (("learned", learned), ("reference", reference)):
                dist = ", ".join(f"{k} {v / n:.2f}" for k, v in counts.most_common())
                print(
                    f"{event_id} {name:<9} P(historical {label}) {counts[str(label)] / n:.2f}; {dist}"
                )
    return 0


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
    text = assumptions.render_markdown(load_catalog())
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


def cmd_assumptions_sweep(args: argparse.Namespace) -> int:
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
    return robustness.RunSpec(
        policy=args.policy,
        npc_policy=args.npc_policy,
        playable_tier=args.playable_tier,
        temperature=args.temperature,
        select=tuple(t for t in (args.select or "").split(",") if t),
        spread=args.spread,
    )


def cmd_assumptions_robust(args: argparse.Namespace) -> int:
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


def _sampling_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--select", help="comma-separated groups, events (event:id) or address globs to draw"
    )
    parser.add_argument("--spread", type=float, default=1.0, help="share of each range used")
    parser.add_argument("--playable-tier", type=int)
    parser.add_argument("--temperature", type=float, default=goalcheck.DEFAULT_TEMPERATURE)
    parser.add_argument("--workers", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cold-war-campaign", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    catalog = sub.add_parser("catalog", help="validate or render the event catalog")
    catalog.add_argument("--check", action="store_true", help="validate and print counts (default)")
    catalog.add_argument("--markdown", action="store_true", help="render the catalog as Markdown")
    catalog.add_argument("--output", help="write Markdown to this path")
    catalog.set_defaults(func=cmd_catalog)

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
        help="logit temperature of computer-run reference play (0 = exact)",
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

    solve = sub.add_parser("solve", help="solve one major event bound at the historical world")
    solve.add_argument("--event", required=True)
    solve.add_argument("--seed", type=int, default=0)
    solve.add_argument("--iterations", type=int, default=1000)
    solve.add_argument("--temperature", type=float, default=goalcheck.DEFAULT_TEMPERATURE)
    solve.set_defaults(func=cmd_solve)

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

    board = sub.add_parser("standing", help="the standing scoreboard along one campaign")
    board.add_argument("--seed", type=int, default=0)
    board.add_argument("--policy", choices=sorted(POLICIES), default="historical")
    board.add_argument("--npc-policy", choices=sorted(POLICIES), default="historical")
    board.add_argument("--chance", choices=("sample", "historical"), default="historical")
    board.set_defaults(func=cmd_standing)

    audit = sub.add_parser("branches", help="branch audit: paths, stuck flags, dead events")
    audit.add_argument("--seed", type=int, default=0)
    audit.add_argument("--seeds", help="inclusive range such as 0-59")
    audit.add_argument("--policy", choices=sorted(POLICIES), default="equilibrium")
    audit.add_argument("--npc-policy", choices=sorted(POLICIES), default="equilibrium")
    audit.add_argument("--events", help="comma-separated events whose paths to list")
    audit.add_argument("--temperature", type=float, default=goalcheck.DEFAULT_TEMPERATURE)
    audit.set_defaults(func=cmd_branches)

    strategy = sub.add_parser("audit", help="interest audit: does a yearly strategy dominate?")
    strategy.add_argument("layer", choices=["strategy"])
    strategy.add_argument("--seed", type=int, default=0)
    strategy.add_argument("--seeds", default="0-1", help="inclusive range such as 0-1")
    strategy.add_argument("--profiles", type=int, default=20)
    strategy.add_argument("--workers", type=int)
    strategy.add_argument("--temperature", type=float, default=goalcheck.DEFAULT_TEMPERATURE)
    strategy.set_defaults(func=cmd_audit)

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

    rl = sub.add_parser("rl", help="reinforcement learning on single-event scenarios")
    rl_sub = rl.add_subparsers(dest="rl_command", required=True)
    rl_entries = rl_sub.add_parser("entries", help="capture scenario entry states")
    rl_entries.add_argument("--events", help="tier1 (default) or comma-separated event ids")
    rl_entries.add_argument("--source", choices=("historical", "campaign"), default="historical")
    rl_entries.add_argument("--seed", type=int, default=0)
    rl_entries.add_argument("--seeds", help="campaign seeds, an inclusive range such as 0-49")
    rl_entries.add_argument("--workers", type=int, default=4)
    rl_entries.add_argument("--playable-tier", type=int, default=1)
    rl_entries.add_argument("--merge", help="add the entries of this file")
    rl_entries.add_argument("--output", required=True)
    rl_entries.set_defaults(func=cmd_rl_entries)
    rl_train = rl_sub.add_parser("train", help="masked-PPO self-play on scenarios")
    rl_train.add_argument("--entries", help="a file written by rl entries (scenario mode)")
    rl_train.add_argument(
        "--mode",
        choices=("scenario", "campaign"),
        default="scenario",
        help="scenario: single events from entries; campaign: freestyle from 1945",
    )
    rl_train.add_argument("--generator", action="store_true", help="campaign mode: crises")
    rl_train.add_argument("--gamma", type=float, default=1.0, help="discount per decision")
    rl_train.add_argument("--league", type=int, default=0, help="snapshot every K iterations")
    rl_train.add_argument("--league-share", type=float, default=0.5)
    rl_train.add_argument("--events", help="tier1 or comma-separated ids (default: all entries)")
    rl_train.add_argument("--iterations", type=int, default=50)
    rl_train.add_argument("--episodes", type=int, default=256, help="per iteration")
    rl_train.add_argument("--workers", type=int, default=4)
    rl_train.add_argument("--hidden", type=int, default=256)
    rl_train.add_argument("--lr", type=float, default=3e-4)
    rl_train.add_argument(
        "--objective",
        choices=("qre", "ppo"),
        default="qre",
        help="qre: the logit QRE at --temperature (payoff units); ppo: plain PPO (E45)",
    )
    rl_train.add_argument("--temperature", type=float, default=0.1, help="the qre objective's tau")
    rl_train.add_argument(
        "--temperature-start",
        type=float,
        help="qre: start at this tau and cool geometrically to --temperature (the QRE path)",
    )
    rl_train.add_argument(
        "--anneal", type=float, default=0.5, help="share of iterations spent cooling"
    )
    rl_train.add_argument(
        "--lr-final", type=float, help="after cooling, decay the learning rate linearly to this"
    )
    rl_train.add_argument("--entropy", type=float, default=0.01, help="the ppo objective's weight")
    rl_train.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    rl_train.add_argument("--seed", type=int, default=0)
    rl_train.add_argument("--eval-every", type=int, default=10)
    rl_train.add_argument("--checkpoint-every", type=int, default=10)
    rl_train.add_argument("--resume", help="a checkpoint to continue from")
    rl_train.add_argument("--out", required=True, help="the run directory")
    rl_train.set_defaults(func=cmd_rl_train)
    rl_br = rl_sub.add_parser("br", help="approximate best response to a checkpoint")
    rl_br.add_argument("--checkpoint", required=True)
    rl_br.add_argument("--identity", required=True, choices=IDENTITIES)
    rl_br.add_argument("--iterations", type=int, default=30)
    rl_br.add_argument("--episodes", type=int, default=256, help="per iteration")
    rl_br.add_argument("--eval-episodes", type=int, default=2000)
    rl_br.add_argument("--workers", type=int, default=4)
    rl_br.add_argument("--seed", type=int, default=0)
    rl_br.add_argument("--out", required=True, help="the best response's run directory")
    rl_br.set_defaults(func=cmd_rl_br)
    rl_eval = rl_sub.add_parser("eval", help="score a checkpoint")
    rl_eval.add_argument("--checkpoint", required=True)
    rl_eval.add_argument("--entries", help="default: the run's entries")
    rl_eval.add_argument("--events", help="tier1 or comma-separated ids (default: all entries)")
    rl_eval.add_argument("--episodes", type=int, default=200, help="for multi-quarter events")
    rl_eval.add_argument("--all-entries", action="store_true", help="score every entry")
    rl_eval.set_defaults(func=cmd_rl_eval)
    return parser


_RL_MODULES = ("cold_war_sim.campaign.rl", "torch", "pettingzoo", "gymnasium")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        status: int = args.func(args)
    except ModuleNotFoundError as error:
        missing = error.name or ""
        if args.command != "rl" or not missing.startswith(_RL_MODULES):
            raise
        if missing.startswith(_RL_MODULES[0]):
            reason = "the RL package (campaign/rl) is not in this checkout"
        else:
            reason = f"it needs {missing} (the RL dependencies: torch, pettingzoo, gymnasium)"
        raise SystemExit(f"rl {args.rl_command}: {reason}") from error
    return status

"""Commands for capturing scenarios, training policies and evaluating their play."""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ..campaign.catalog.core import Catalog, load_catalog
from ..campaign.engine.core import EngineConfig
from ..campaign.world import calendar
from ..campaign.world.state import IDENTITIES
from .main import _seeds


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

    from ..campaign.environments import scenario

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


def _scored_entries(catalog: Catalog, entries: Any) -> list[Any]:
    """One entry per tree event: the historical one, else the first."""

    from ..evaluation import learned_play as rl_evaluate

    chosen: dict[str, Any] = {}
    for entry in rl_evaluate.tree_entries(catalog, entries):
        current = chosen.get(entry.event_id)
        if current is None or (entry.source == "historical" and current.source != "historical"):
            chosen[entry.event_id] = entry
    return list(chosen.values())


def _tree_scores(
    catalog: Catalog, entries: Any, policy: Any, temperature: float | None = None
) -> dict[str, Any]:
    """The tree scores (exploitability and the rest) of *policy* on the scored entries."""

    from ..evaluation import learned_play as rl_evaluate

    scored: dict[str, Any] = {}
    for entry in _scored_entries(catalog, entries):
        report = rl_evaluate.tree_report(catalog, entry, policy, qre_temperature=temperature)
        scored[entry.event_id] = {
            name: {
                "exploitability": round(score.exploitability, 5),
                "worst_regret": round(score.credibility, 5),
                "path_regret": float(f"{score.path_regret:.3e}"),
                "qre_distance": None
                if score.qre_distance is None
                else round(score.qre_distance, 4),
                "p_historical": round(score.historical, 3),
                "outcomes": {k: round(v, 3) for k, v in score.outcomes.items() if v >= 0.005},
            }
            for name, score in report.scores.items()
        }
        if "learned" in scored[entry.event_id]:  # 0 unless observations leak
            scored[entry.event_id]["learned"]["inconsistent"] = sum(report.inconsistent.values())
        scored[entry.event_id]["qre"]["residual"] = float(f"{report.qre_residual:.2e}")
    return scored


def cmd_rl_train(args: argparse.Namespace) -> int:
    """Masked-PPO self-play: single-event scenarios or freestyle campaigns."""

    from ..rl import ppo

    catalog = load_catalog()
    events = tuple(_rl_events(catalog, args.events)) if args.events else ()
    if args.mode == "scenario" and not args.entries:
        raise SystemExit("rl train: scenario mode needs --entries")
    try:
        config = _ppo_config(ppo, args, events)
    except ValueError as error:
        raise SystemExit(f"rl train: {error}") from error
    evaluate = None
    if config.mode == "scenario":
        entries = ppo.scenario_entries(config)
        if config.chance_expected:  # only instant events count chance in expectation
            instant = [e for e in sorted(entries) if _cadence(catalog, e) == "instant"]
            other = [e for e in sorted(entries) if e not in instant]
            note = f"; sampled returns for {', '.join(other)} (over quarters)" if other else ""
            print(
                f"rl train: chance-expected returns for {', '.join(instant) or 'no event'}{note}",
                file=sys.stderr,
            )

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


def _cadence(catalog: Catalog, event_id: str) -> str | None:
    """The cadence of a major event."""

    major = catalog.events[event_id].major
    return None if major is None else major.cadence


def _ppo_config(ppo: Any, args: argparse.Namespace, events: tuple[str, ...]) -> Any:
    """The training configuration of ``rl train``'s arguments."""

    return ppo.PPOConfig(
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
        chance_expected=args.chance_expected,
    )


def cmd_rl_br(args: argparse.Namespace) -> int:
    """An identity's approximate best response to a checkpoint, and its gain."""

    from ..rl import br

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

    from ..campaign.environments import scenario
    from ..evaluation import learned_play as rl_evaluate
    from ..rl import ppo

    learner = ppo.Learner.load(Path(args.checkpoint))
    policy = learner.policy()
    catalog = load_catalog()
    entries_path = Path(args.entries or learner.config.entries)
    entries = scenario.load_entries(entries_path)
    events = _rl_events(catalog, args.events) if args.events else sorted(entries)
    entries = {e: entries[e] for e in events if e in entries}
    scored = (
        rl_evaluate.tree_entries(catalog, entries)
        if args.all_entries
        else _scored_entries(catalog, entries)
    )
    for entry in scored:
        report = rl_evaluate.tree_report(
            catalog, entry, policy, qre_temperature=learner.config.temperature
        )
        print("\n".join(report.text()))
    trees = {e.event_id for e in rl_evaluate.tree_entries(catalog, entries)}
    others = {e: v for e, v in entries.items() if e not in trees}
    if others:
        # play from the file given, not the one the checkpoint was trained on
        learned_env = ppo.make_env(
            learner.config.replace(entries=str(entries_path), events=tuple(sorted(others))),
            catalog,
        )
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


def add_rl(sub: Any) -> None:
    """The ``rl`` commands."""

    rl = sub.add_parser(
        "rl", help="reinforcement learning: single-event scenarios and freestyle campaigns"
    )
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
    rl_train.add_argument(
        "--chance-expected",
        action="store_true",
        help="scenario mode: average over chance outcomes and computer-run choices",
    )
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
        help="qre: the logit QRE at --temperature (payoff units); ppo: plain PPO",
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
    rl_eval.add_argument(
        "--all-entries", action="store_true", help="score every entry (default: one an event)"
    )
    rl_eval.set_defaults(func=cmd_rl_eval)

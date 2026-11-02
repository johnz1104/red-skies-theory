"""The command line: the rl commands where the RL package or its dependencies are missing."""

from __future__ import annotations

import sys
from typing import NoReturn

import pytest

import cold_war_sim.campaign as campaign
from cold_war_sim.campaign import cli

RL = "cold_war_sim.campaign.rl"


def _hide(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Make importing *name* fail as if it were not installed (restored afterwards)."""

    for module in [m for m in sys.modules if m == RL or m.startswith(RL + ".")]:
        monkeypatch.delitem(sys.modules, module)
    monkeypatch.delattr(campaign, "rl", raising=False)
    monkeypatch.setitem(sys.modules, name, None)


@pytest.mark.parametrize(
    ("hidden", "reason"),
    [(RL, "the RL package .* is not in this checkout"), ("torch", "it needs torch")],
)
def test_rl_commands_explain_what_is_missing(
    monkeypatch: pytest.MonkeyPatch, hidden: str, reason: str
) -> None:
    _hide(monkeypatch, hidden)
    with pytest.raises(SystemExit, match=f"rl eval: {reason}"):
        cli.main(["rl", "eval", "--checkpoint", "missing.pt"])


def _missing(name: str) -> NoReturn:
    raise ModuleNotFoundError(f"No module named {name!r}", name=name)


def test_other_missing_modules_still_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    # outside the rl commands, or a module the RL install does not provide
    monkeypatch.setattr(cli, "cmd_assumptions_check", lambda args: _missing("torch"))
    with pytest.raises(ModuleNotFoundError):
        cli.main(["assumptions", "check"])
    monkeypatch.setattr(cli, "cmd_rl_eval", lambda args: _missing("yaml"))
    with pytest.raises(ModuleNotFoundError):
        cli.main(["rl", "eval", "--checkpoint", "missing.pt"])

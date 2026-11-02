"""Reinforcement learning on the campaign's environments (user, 2026-10-20).

* :mod:`.pz`: PettingZoo AEC adapters for :class:`~..scenario.ScenarioEnv` and
  :class:`~..env.ColdWarEnv`.
* :mod:`.ppo`: masked-PPO self-play, one actor-critic per learning identity, with
  rollouts on worker processes.
* :mod:`.evaluate`: learned play scored on an event's game tree: exploitability (each
  agent role's gain from a best response), sequential regret, the distance to the
  true-game QRE and the outcome distribution, against reference play and an exact
  equilibrium.
* :mod:`.br`: an identity's approximate best response to a checkpoint, for events without
  a tree.

Needs the ``rl`` extra (``pip install -e ".[rl]"``: torch, pettingzoo, gymnasium); the
campaign itself does not import this package.
"""

"""Reinforcement learning on the campaign's environments.

* :mod:`.pz`: PettingZoo AEC adapters for the scenario and campaign environments.
* :mod:`.ppo`: masked-PPO self-play, one actor-critic per learning identity.
* :mod:`.evaluate`: learned play scored on an event's game tree.
* :mod:`.br`: an identity's approximate best response to a checkpoint.

Needs the ``rl`` extra (``pip install -e ".[rl]"``: torch, pettingzoo, gymnasium); the
campaign itself does not import this package.
"""

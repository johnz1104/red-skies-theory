# Cold War Simulator

A computational framework that analyzes major Cold War events through the
lens of formal game theory, modeling strategic interactions between
superpowers as structured games and solving for equilibria and optimal
strategies within those models.

## Game-theoretic frameworks

**Bayesian extensive-form and signaling games:** These games unfold through a
sequence of moves, and players do not know everything about one another. The
Cuban Missile Crisis model studies noisy U.S. intelligence, Soviet resolve,
public signals, and accident risk. The Korean War model asks when a modeled
warning from China is credible enough to change the receiver's advance
decision. Both models check every equilibrium in their supported
pure-strategy perfect Bayesian equilibrium class.

**Dynamic bargaining:** Players take turns making offers, and waiting can be
costly or risky. The Berlin model studies how patience, commitments, and
escalation risk affect the offers each side will accept. It works backward
from the final round to find a finite-horizon subgame-perfect equilibrium.

**Differential games:** Players make choices continuously rather than taking
separate turns. In the strategic arms-competition model, each player controls
how its modeled armament stock changes over time. The solver finds an
open-loop Nash equilibrium and checks it with a separate numerical method.
Richardson's classic arms-race equations are included separately as a baseline
without strategic choices.

**Counterfactual analysis:** The software can replace a policy, change
available information, restrict actions, or add a supported commitment and
then account for strategic responses. It distinguishes an outcome that merely
exists in a game from one that a feasible policy can reach and one supported
after the game is solved again. A model-feasible counterfactual is not a claim
about what should have happened in history.

## Implemented scenarios

| Scenario | Model | Supported solution |
| --- | --- | --- |
| Cuban Missile Crisis | Bayesian crisis bargaining with sequential signaling | Revealed-type backward induction and restricted pure-PBE enumeration |
| Korean War intervention | Warning signaling followed by advance and intervention decisions | Continuation optimization and restricted pure-PBE enumeration |
| Berlin confrontation | Finite-horizon alternating-offers bargaining with risky delay | Exact backward induction on a finite offer grid |
| Arms competition | Two-player linear-quadratic differential game | Numerically verified open-loop Nash equilibrium |
| Richardson baseline | Affine arms-race dynamics | RK4 trajectories, fixed points, and stability classification |

## Repository layout

```text
src/cold_war_sim/
  core/                 finite games, probabilities, beliefs, and utilities
  solvers/              backward induction, best responses, and pure PBE
  events/               Cuba, Korea, Berlin, and arms-competition models
  counterfactuals/      interventions, feasibility, responses, and policy search
  differential_games/   open-loop Nash solver and independent verification
```

# Hybrid Intelligence Schrodingers Cats Game

Implementation and evaluation of Theory-of-Mind (ToM) agents for the evidence-based **Schrödinger’s Cats** card game (Hybrid Intelligence course project, University of Groningen).

The project includes:
- An **evidence-based** game environment (optional evidence reveal when making claims)
- A **ToM0 agent with explicit memory** (`ToM0(mem)`)
- A **ToM1 agent** with interpretative + predictive ToM and a **behaviour-triggered fallback** to a ToM0-style policy
- An evaluation script that runs seat-swapped experiments and prints **win rates + 95% confidence intervals**

---

## Repository structure (key files)

- `src/schcats/env.py`  
  Game environment (`SchCatsEnv`) and action definitions (`MakeClaim`, `Doubt`).

- `src/schcats/agents/tom0_memory.py`  
  Zero-order ToM agent with explicit cross-round memory (ToM0(mem)).

- `src/schcats/agents/tom1.py`  
  First-order ToM agent (ToM1) using:
  - interpretative ToM (uses opponent revealed evidence to update belief about hidden cards),
  - predictive ToM (models opponent doubt behaviour),
  - fallback to a ToM0-style policy when the opponent’s behaviour matches a trigger rule.

- `src/schcats/experiments/run_matches.py`  
  Runs experiments (baseline + ToM1 vs ToM0(mem) with seat swap), prints results.

- `src/schcats/experiments/metrics.py`  
  Simple win-rate statistics (including 95% CI).

---

## Requirements

- Python 3.10+ (3.11 also fine)
- No external dependencies required beyond the standard library for the core experiments.

---

## How to run

From the repository root:

```bash
python3 -m src.schcats.experiments.run_matches
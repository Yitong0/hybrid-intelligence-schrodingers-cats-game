from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
from ..env import SchCatsEnv
from ..agents.tom0_memory import ToM0MemoryAgent
from ..agents.tom1 import ToM1Agent


@dataclass
class Result:
    p0_wins: int
    p1_wins: int
    winrate_p0: float
    winrate_p1: float


def play_match(env: SchCatsEnv, a0, a1) -> Tuple[int, int]:
    obs0, obs1 = env.reset_match()
    done = False

    while not done:
        pid = env.public.turn  # type: ignore
        obs = obs0 if pid == 0 else obs1
        agent = a0 if pid == 0 else a1
        action = agent.act(obs)

        winner, done = env.step(pid, action)

        # refresh observations (hands/public changed)
        obs0, obs1 = env._obs(0), env._obs(1)

        if winner is not None:
            # notify agents for explicit memory updates
            a0.observe_round_end(my_id=0, winner=winner, last_obs=env.last_round_obs(0))
            a1.observe_round_end(my_id=1, winner=winner, last_obs=env.last_round_obs(1))

    return env.scores[0], env.scores[1]


def eval_config(n_matches: int, rounds_per_match: int, seed: int, agent0, agent1) -> Result:
    p0_total = 0
    p1_total = 0
    for m in range(n_matches):
        env = SchCatsEnv(rounds_per_match=rounds_per_match, seed=seed + m)
        a0 = agent0(seed=seed + 1000 + m)
        a1 = agent1(seed=seed + 2000 + m)
        s0, s1 = play_match(env, a0, a1)
        p0_total += s0
        p1_total += s1

    total = p0_total + p1_total
    return Result(
        p0_wins=p0_total,
        p1_wins=p1_total,
        winrate_p0=p0_total / total if total else 0.0,
        winrate_p1=p1_total / total if total else 0.0,
    )


def main():
    N_MATCHES = 1000
    ROUNDS_PER_MATCH = 30
    SEED = 0

    # Baseline: ToM0(mem) vs ToM0(mem)
    r00 = eval_config(N_MATCHES, ROUNDS_PER_MATCH, SEED, ToM0MemoryAgent, ToM0MemoryAgent)

    # Main comparisons with seat swapping:
    # A) ToM1 is player 0
    r10_p0 = eval_config(N_MATCHES, ROUNDS_PER_MATCH, SEED, ToM1Agent, ToM0MemoryAgent)
    # B) ToM1 is player 1
    r10_p1 = eval_config(N_MATCHES, ROUNDS_PER_MATCH, SEED, ToM0MemoryAgent, ToM1Agent)

    # Extract ToM1 winrates in each seating
    tom1_wr_as_p0 = r10_p0.winrate_p0  # p0 is ToM1
    tom1_wr_as_p1 = r10_p1.winrate_p1  # p1 is ToM1

    tom1_wr_avg = 0.5 * (tom1_wr_as_p0 + tom1_wr_as_p1)

    print("== Baseline ==")
    print(f"ToM0(mem) vs ToM0(mem): p0={r00.winrate_p0:.3f}, p1={r00.winrate_p1:.3f}")

    print("\n== Main (seat swap) ==")
    print(f"ToM1 as P0 vs ToM0(mem): ToM1={r10_p0.winrate_p0:.3f}, ToM0={r10_p0.winrate_p1:.3f}")
    print(f"ToM0(mem) vs ToM1 as P1: ToM0={r10_p1.winrate_p0:.3f}, ToM1={r10_p1.winrate_p1:.3f}")

    print(f"\nToM1 winrate (avg over seats): {tom1_wr_avg:.4f}")


if __name__ == "__main__":
    main()
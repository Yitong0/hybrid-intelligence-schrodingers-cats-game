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
            a0.observe_round_end(my_id=0, winner=winner, last_obs=obs0)
            a1.observe_round_end(my_id=1, winner=winner, last_obs=obs1)

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
    N_MATCHES = 200
    ROUNDS_PER_MATCH = 30
    SEED = 0

    # Baseline: ToM0(mem) vs ToM0(mem)
    r00 = eval_config(N_MATCHES, ROUNDS_PER_MATCH, SEED, ToM0MemoryAgent, ToM0MemoryAgent)

    # Main: ToM1 vs ToM0(mem)
    r10 = eval_config(N_MATCHES, ROUNDS_PER_MATCH, SEED, ToM1Agent, ToM0MemoryAgent)

    print("ToM0(mem) vs ToM0(mem):", r00)
    print("ToM1 vs ToM0(mem):    ", r10)


if __name__ == "__main__":
    main()
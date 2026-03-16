from __future__ import annotations
import sys
import time
from dataclasses import dataclass
from typing import Tuple, Type
from ..env import SchCatsEnv
from ..agents.tom0_memory import ToM0MemoryAgent
from ..agents.tom1 import ToM1Agent
from .metrics import WinStats


@dataclass
class Result:
    p0_round_wins: int
    p1_round_wins: int

    @property
    def total_rounds(self) -> int:
        return self.p0_round_wins + self.p1_round_wins

    def stats_p0(self) -> WinStats:
        return WinStats(self.p0_round_wins, self.p1_round_wins)

    def stats_p1(self) -> WinStats:
        return WinStats(self.p1_round_wins, self.p0_round_wins)


def play_match(env: SchCatsEnv, a0, a1) -> Tuple[int, int]:
    obs0, obs1 = env.reset_match()
    done = False

    while not done:
        pid = env.public.turn
        obs = obs0 if pid == 0 else obs1
        agent = a0 if pid == 0 else a1
        action = agent.act(obs)

        winner, done = env.step(pid, action)

        obs0, obs1 = env._obs(0), env._obs(1)

        if winner is not None:
            a0.observe_round_end(my_id=0, winner=winner, last_obs=env.last_round_obs(0))
            a1.observe_round_end(my_id=1, winner=winner, last_obs=env.last_round_obs(1))

    return env.scores[0], env.scores[1]


def eval_config(n_matches: int, rounds_per_match: int, seed: int,
                agent0: Type, agent1: Type, label: str = "") -> tuple:
    p0_total = 0
    p1_total = 0
    start = time.time()
    log_every = max(1, n_matches // 20)
    last_a0 = None
    last_a1 = None

    for m in range(n_matches):
        env = SchCatsEnv(rounds_per_match=rounds_per_match, seed=seed + m)
        a0 = agent0(seed=seed + 1000 + m)
        a1 = agent1(seed=seed + 2000 + m)
        s0, s1 = play_match(env, a0, a1)
        p0_total += s0
        p1_total += s1
        last_a0, last_a1 = a0, a1

        if (m + 1) % log_every == 0 or m == n_matches - 1:
            elapsed = time.time() - start
            pct = (m + 1) / n_matches * 100
            rounds_done = (m + 1) * rounds_per_match
            p0_rate = p0_total / rounds_done if rounds_done > 0 else 0
            eta = elapsed / (m + 1) * (n_matches - m - 1)
            print(f"  [{label}] {m+1}/{n_matches} matches ({pct:.0f}%)  "
                  f"P0 winrate so far: {p0_rate:.3f}  "
                  f"elapsed: {elapsed:.0f}s  ETA: {eta:.0f}s",
                  flush=True)

    result = Result(p0_round_wins=p0_total, p1_round_wins=p1_total)
    fallback_rate = None
    for agent in [last_a0, last_a1]:
        if hasattr(agent, "fallback_rate"):
            fallback_rate = agent.fallback_rate
    return result, fallback_rate


def _print_result(title: str, r: Result):
    s0 = r.stats_p0()
    s1 = r.stats_p1()
    ci0 = s0.ci95()
    ci1 = s1.ci95()
    print(title)
    print(f"  P0 winrate: {s0.winrate:.3f}  (95% CI {ci0[0]:.3f}..{ci0[1]:.3f})   rounds={s0.n}")
    print(f"  P1 winrate: {s1.winrate:.3f}  (95% CI {ci1[0]:.3f}..{ci1[1]:.3f})   rounds={s1.n}")


def main():
    N_MATCHES = 1000
    ROUNDS_PER_MATCH = 30
    SEED = 0

    print("Starting baseline: ToM0(mem) vs ToM0(mem)...", flush=True)
    r00, _ = eval_config(N_MATCHES, ROUNDS_PER_MATCH, SEED,
                         ToM0MemoryAgent, ToM0MemoryAgent, label="ToM0 vs ToM0")

    print("\nStarting ToM1 as P0 vs ToM0(mem)...", flush=True)
    r10_p0, p0_fallback_rate = eval_config(N_MATCHES, ROUNDS_PER_MATCH, SEED,
                                           ToM1Agent, ToM0MemoryAgent, label="ToM1(P0) vs ToM0")

    print("\nStarting ToM1 as P1 vs ToM0(mem)...", flush=True)
    r10_p1, p1_fallback_rate = eval_config(N_MATCHES, ROUNDS_PER_MATCH, SEED,
                                           ToM0MemoryAgent, ToM1Agent, label="ToM0 vs ToM1(P1)")

    print()
    _print_result("== Baseline: ToM0(mem) vs ToM0(mem) ==", r00)
    _print_result("== ToM1 as P0 vs ToM0(mem) ==", r10_p0)
    _print_result("== ToM1 as P1 vs ToM0(mem) ==", r10_p1)

    print("\n== ToM0 fallback diagnostics ==")
    print("  (How often did ToM1 fall back to ToM0-style play?)")
    print(f"  P0 config fallback rate: {p0_fallback_rate:.1%}")
    print(f"  P1 config fallback rate: {p1_fallback_rate:.1%}")

    tom1_wins = r10_p0.p0_round_wins + r10_p1.p1_round_wins
    tom1_losses = r10_p0.p1_round_wins + r10_p1.p0_round_wins
    tom1 = WinStats(tom1_wins, tom1_losses)
    ci = tom1.ci95()

    print("\n== ToM1 overall (avg across seats) ==")
    print(f"  ToM1 winrate: {tom1.winrate:.3f}  (95% CI {ci[0]:.3f}..{ci[1]:.3f})   rounds={tom1.n}")


if __name__ == "__main__":
    main()
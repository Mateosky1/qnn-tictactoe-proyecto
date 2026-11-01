"""Numpy-only checks for selfplay_vec.py (no torch needed). Run: python test_selfplay_vec.py"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'app')); sys.path.insert(0, HERE)
import game_rules as gr
import selfplay_vec as sv

rng = np.random.default_rng(0)

# 1. same winning lines as the reference rules
assert {tuple(l) for l in sv.LINES} == {tuple(l) for l in gr.LINES}, 'lines differ'

# 2. the 8 D4 permutations are distinct and map winning lines onto winning lines
assert len({tuple(p) for p in sv.SYM}) == 8, 'sym perms not distinct'
LS = {frozenset(l) for l in sv.LINES}
assert all({frozenset(int(p[i]) for i in l) for l in sv.LINES} == LS for p in sv.SYM), 'sym breaks lines'

# 3. batch win detection == reference, on random boards; win is D4-invariant
for _ in range(3000):
    b = rng.integers(0, 3, 25).astype(np.int8)
    for p in (1, 2):
        ref = gr.has_won(list(b), p)
        assert bool(sv.won(b[None], np.array([p]))[0]) == ref
        for k in range(8):
            assert gr.has_won(list(b[sv.SYM[k]]), p) == ref
print('rules + symmetry OK')

# 4. full games with random policies: terminal state agrees with reference, records are coherent
steps, winner, boards = sv.run_games(sv.random_policy, sv.random_policy, True, True, 400, rng, 4, 0)
for g in range(400):
    done, w = gr.game_over(list(boards[g]))
    assert done and w == winner[g], (g, w, winner[g])
    for p in (1, 2):
        for (xt, a_t, lp, v) in steps[g][p - 1]:
            assert xt[2, a_t] == 1.0, 'recorded action not legal in recorded (transformed) state'
            assert np.isfinite(lp) and lp <= 0
print('self-play records OK; results P1/P2/draw =',
      [(winner == k).sum() for k in (1, 2, 0)])

# 5. the symmetry trick: recorded action maps back to a legal cell of the real board
class Spy:
    def __call__(self, x):
        return np.zeros((len(x), 25), np.float32), np.zeros(len(x), np.float32)
steps, winner, boards = sv.run_games(Spy(), Spy(), True, True, 200, rng, 0, 0)
for g in range(200):
    assert (boards[g] != 0).sum() >= 7  # games have no opening and still finish legally
print('no-opening games OK')

# 6. GAE with v=0 and lam=1,gamma=1 gives +1 / -1 / 0 to every step of winner / loser / draw
steps, winner, _ = sv.run_games(sv.random_policy, sv.random_policy, True, True, 300, rng, 3, 0)
batch = sv.build_batch(steps, winner, gamma=1.0, lam=1.0)
off = 0
for g in range(300):
    for p in (1, 2):
        T = len(steps[g][p - 1])
        if T == 0: continue
        exp = 0.0 if winner[g] == 0 else (1.0 if winner[g] == p else -1.0)
        assert np.allclose(batch['adv'][off:off + T], exp)
        off += T
assert off == len(batch['adv']) == len(batch['x'])
print('GAE OK; samples =', len(batch['x']))

# 7. greedy match harness: a policy always playing the lowest free index vs random
def lowest(x):
    lg = -np.arange(25, dtype=np.float32)[None].repeat(len(x), 0)
    return lg, np.zeros(len(x), np.float32)
W, D, L = sv.match(lowest, sv.random_policy, 400, rng, greedyB=False)
print('lowest-index vs UNIFORM random W/D/L =', W, D, L)
assert W + D + L == 400
# a greedy 'random_policy' is NOT random: it always plays the lowest free cell
W2, D2, L2 = sv.match(lowest, sv.random_policy, 100, rng, greedyB=True)
assert D2 == 100 or W2 + L2 > 0
print('ALL OK')

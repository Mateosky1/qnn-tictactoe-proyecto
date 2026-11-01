"""Vectorized self-play engine for 5x5 / 4-in-a-row (numpy only, no torch).

Pure neural self-play machinery:
- many games advance in parallel; every ply is ONE batched policy call
- both sides can be recorded (twice the data of v21)
- every recorded decision is taken in a RANDOM D4 SYMMETRY of the board, and the
  stored (state, action, old_logprob) are the transformed ones. The sample is
  therefore exactly on-policy for PPO and the data is symmetry-augmented for free.
- no lookahead of any kind: a move is sampled (or argmax'ed) from the policy output.
"""
import numpy as np

BOARD_N, N, WIN_LEN = 5, 25, 4


def _lines():
    out = []
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                rr, cc = r + dr * (WIN_LEN - 1), c + dc * (WIN_LEN - 1)
                if 0 <= rr < BOARD_N and 0 <= cc < BOARD_N:
                    out.append([(r + dr * k) * BOARD_N + c + dc * k for k in range(WIN_LEN)])
    return out


LINES = _lines()
LINE_MAT = np.zeros((N, len(LINES)), np.float32)
for _j, _l in enumerate(LINES):
    for _i in _l:
        LINE_MAT[_i, _j] = 1.0


def _sym_perms():
    idx = np.arange(N).reshape(BOARD_N, BOARD_N)
    perms = []
    for k in range(4):
        r = np.rot90(idx, k)
        perms.append(r.reshape(-1))
        perms.append(np.fliplr(r).reshape(-1))
    return np.stack(perms).astype(np.int64)


SYM = _sym_perms()  # (8,25): board_t[i] = board[SYM[k][i]]  ->  original cell = SYM[k][a_t]


def won(boards, players):
    """boards (n,25) int8, players (n,) -> bool (n,): did `players[i]` complete a line."""
    mine = (boards == players[:, None]).astype(np.float32)
    return (mine @ LINE_MAT >= WIN_LEN - 0.5).any(1)


def encode_batch(boards, players):
    pl = players[:, None]
    return np.stack([boards == pl, boards == (3 - pl), boards == 0], 1).astype(np.float32)  # (n,3,25)


def _gumbel(rng, shape):
    return -np.log(-np.log(rng.random(shape) + 1e-20) + 1e-20)


def pick(logits, legal, rng, greedy):
    """Masked categorical sample (Gumbel-max) or argmax. Returns action, logprob of action."""
    m = np.where(legal, logits.astype(np.float64), -1e9)
    z = m - m.max(1, keepdims=True)
    p = np.exp(z) * legal
    p /= p.sum(1, keepdims=True)
    a = m.argmax(1) if greedy else (m + _gumbel(rng, m.shape)).argmax(1)
    lp = np.log(np.maximum(p[np.arange(len(a)), a], 1e-12))
    return a, lp.astype(np.float32)


def run_games(pol1, pol2, rec1, rec2, n, rng, max_open=4, min_open=0, greedy1=False, greedy2=False,
              sym=True):
    """Play n games. pol(x[n,3,25]) -> (logits[n,25], values[n]) as numpy arrays.
    pol1 plays P1 (moves first after the random opening), pol2 plays P2.
    rec1/rec2: record decisions of that side. Random opening moves are never recorded.
    Returns steps (per game, per side lists), winner (n,), final boards (n,25)."""
    boards = np.zeros((n, N), np.int8)
    to_move = np.ones(n, np.int8)
    open_left = rng.integers(min_open, max_open + 1, size=n)
    alive = np.ones(n, bool)
    winner = np.zeros(n, np.int8)
    steps = [[[], []] for _ in range(n)]

    def apply(idx, mv):
        pl = to_move[idx]
        boards[idx, mv] = pl
        w = won(boards[idx], pl)
        full = (boards[idx] != 0).all(1)
        winner[idx[w]] = pl[w]
        alive[idx[w | full]] = False
        cont = idx[~(w | full)]
        to_move[cont] = 3 - to_move[cont]

    while alive.any():
        op = alive & (open_left > 0)
        if op.any():
            idx = np.nonzero(op)[0]
            legal = boards[idx] == 0
            mv = np.where(legal, _gumbel(rng, legal.shape), -1e18).argmax(1)
            open_left[idx] -= 1
            apply(idx, mv)
        pol_idx = np.nonzero(alive & (open_left <= 0) & ~op)[0]
        for p, pol, rec, greedy in ((1, pol1, rec1, greedy1), (2, pol2, rec2, greedy2)):
            sel = pol_idx[to_move[pol_idx] == p]
            if len(sel) == 0:
                continue
            x = encode_batch(boards[sel], to_move[sel])
            if sym and not greedy:
                k = rng.integers(0, 8, len(sel))
                perm = SYM[k]
            else:
                perm = np.broadcast_to(SYM[0], (len(sel), N))
            xt = np.take_along_axis(x, perm[:, None, :], axis=2)
            logits, v = pol(xt)
            at, lp = pick(logits, xt[:, 2] > 0.5, rng, greedy)
            a = perm[np.arange(len(sel)), at]
            if rec:
                for j, g in enumerate(sel):
                    steps[g][p - 1].append((xt[j], int(at[j]), float(lp[j]), float(v[j])))
            apply(sel, a)
    return steps, winner, boards


def build_batch(steps, winner, gamma=0.99, lam=0.95):
    """GAE per (game, side) trajectory in the side's own perspective.
    Reward only at the side's last decision: +1 win, -1 loss, 0 draw."""
    X, A, LP, V, ADV, RET = [], [], [], [], [], []
    for g, both in enumerate(steps):
        for p in (1, 2):
            seq = both[p - 1]
            if not seq:
                continue
            T = len(seq)
            w = int(winner[g])
            r = 0.0 if w == 0 else (1.0 if w == p else -1.0)
            v = np.array([s[3] for s in seq], np.float32)
            adv = np.zeros(T, np.float32)
            gae = 0.0
            for t in range(T - 1, -1, -1):
                if t == T - 1:
                    gae = r - v[t]
                else:
                    gae = gamma * v[t + 1] - v[t] + gamma * lam * gae
                adv[t] = gae
            X += [s[0] for s in seq]
            A += [s[1] for s in seq]
            LP += [s[2] for s in seq]
            V.append(v)
            ADV.append(adv)
            RET.append(adv + v)
    return {'x': np.stack(X).astype(np.float32), 'a': np.asarray(A, np.int64),
            'oldlp': np.asarray(LP, np.float32), 'v': np.concatenate(V),
            'adv': np.concatenate(ADV), 'ret': np.concatenate(RET)}


def match(polA, polB, games, rng, min_open=1, max_open=3, greedyB=True):
    """A (greedy) vs B, A plays each side half of the time. Returns (W, D, L) for A.
    greedyB=False makes B SAMPLE from its policy (use it for the uniform-random baseline)."""
    h = games // 2
    _, w1, _ = run_games(polA, polB, False, False, h, rng, max_open, min_open, True, greedyB, False)
    _, w2, _ = run_games(polB, polA, False, False, games - h, rng, max_open, min_open, greedyB, True, False)
    W = int((w1 == 1).sum() + (w2 == 2).sum())
    L = int((w1 == 2).sum() + (w2 == 1).sum())
    return W, games - W - L, L


def random_policy(x):
    return np.zeros((len(x), N), np.float32), np.zeros(len(x), np.float32)

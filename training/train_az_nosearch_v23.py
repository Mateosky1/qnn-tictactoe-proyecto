"""AZ-NoSearch v23 (v22 + speed engineering): pure neural self-play with PPO/GAE. No search of any kind.

Contract (unchanged from v21):
- random initialization; the only optimized policy is the current network
- opponents: the current network or snapshots taken in THIS run
- no MCTS / Oracle / Minimax / Alpha-Beta / teacher / external tactical reward or feature
- final inference: one forward pass + legal argmax

What changed vs v21 (all of it is neural-RL engineering, none of it is search):
1. Both sides of every self-play game are trained on (2x data per game).
2. Games run in parallel and every ply is one batched forward (~order of magnitude faster).
3. Each decision is taken in a random D4 symmetry; the transformed sample is stored, so
   PPO stays exactly on-policy while the data is symmetry-augmented (replaces the KL loss).
4. Fully convolutional policy head.
5. Bigger batches, higher LR with cosine decay, value clipping, KL early stop,
   entropy annealing, openings 0..max_open (empty board included).
6. v23: GPU auto-detect, presets (fast/full), wall-clock budget (--minutes), --benchmark.
7. Logging: draw rate in self-play and greedy win rate vs a UNIFORM RANDOM player
   (a sanity metric, never used for optimization).
"""
import argparse, copy, json, math, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'app')); sys.path.insert(0, HERE)
from az_resnet_model import AZResNet
import selfplay_vec as sv


def get_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def as_policy(model, dev):
    @torch.inference_mode()
    def f(x):
        model.eval()
        t = torch.from_numpy(np.ascontiguousarray(x)).reshape(-1, 3, 5, 5).to(dev)
        lg, v = model(t)
        return lg.cpu().numpy(), v.cpu().numpy()
    return f


def ppo_update(model, opt, b, a, ent_coef, dev):
    T = lambda k: torch.from_numpy(b[k]).to(dev)
    X = T('x').reshape(-1, 3, 5, 5)
    A, OLP, OV, RET = T('a'), T('oldlp'), T('v'), T('ret')
    ADV = T('adv'); ADV = (ADV - ADV.mean()) / (ADV.std(unbiased=False) + 1e-6)
    n = len(A); model.train(); stats = {}
    stop = False
    for _ in range(a.ppo_epochs):
        order = torch.randperm(n, device=dev)
        for s in range(0, n, a.minibatch):
            i = order[s:s + a.minibatch]
            logits, v = model(X[i])
            legal = X[i][:, 2].reshape(len(i), 25) > 0.5
            logp = F.log_softmax(logits.masked_fill(~legal, -1e9), 1)
            ch = logp.gather(1, A[i][:, None])[:, 0]
            ratio = torch.exp(ch - OLP[i])
            pol = -torch.min(ratio * ADV[i], torch.clamp(ratio, 1 - a.ppo_clip, 1 + a.ppo_clip) * ADV[i]).mean()
            vc = OV[i] + torch.clamp(v - OV[i], -a.vf_clip, a.vf_clip)
            val = 0.5 * torch.max((v - RET[i]) ** 2, (vc - RET[i]) ** 2).mean()
            ent = -(logp.exp() * logp).sum(1).mean()
            loss = pol + a.value_coef * val - ent_coef * ent
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            kl = float((OLP[i] - ch).detach().mean())
            stats = {'policy': pol.item(), 'value': val.item(), 'entropy': ent.item(), 'kl': kl,
                     'clipfrac': float(((ratio - 1).abs() > a.ppo_clip).float().mean())}
            if kl > 1.5 * a.target_kl:
                stop = True; break
        if stop: break
    model.eval()
    return stats


def collect(model, pool, a, rng, dev):
    cur = as_policy(model, dev)
    n_lg = int(a.games * a.pool_prob) if pool else 0
    n_self = a.games - n_lg
    steps, winner, selfw = [], [], None
    s, w, _ = sv.run_games(cur, cur, True, True, n_self, rng, a.max_open)
    steps += s; winner += list(w); selfw = w
    chunk = max(16, min(64, n_lg // 2)); chunk -= chunk % 2
    for _ in range(max(1, n_lg // chunk) if n_lg > 0 else 0):
        opp = as_policy(random.choice(pool), dev)
        s, w, _ = sv.run_games(cur, opp, True, False, chunk // 2, rng, a.max_open)
        steps += s; winner += list(w)
        s, w, _ = sv.run_games(opp, cur, False, True, chunk // 2, rng, a.max_open)
        steps += s; winner += list(w)
    return sv.build_batch(steps, np.asarray(winner), a.gamma, a.gae_lambda), selfw


def save(model, a, manifest, path, extra):
    torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'architecture': {**vars(a), 'type': 'residual_cnn_groupnorm'},
                'purity': manifest, **extra}, path)


PRESETS = {
    # ~2.2x less compute per sample than 'full'; receptive field still covers the whole 5x5.
    'fast': dict(channels=48, blocks=4, games=192, ppo_epochs=3, minibatch=1024),
    'full': dict(channels=64, blocks=5, games=256, ppo_epochs=4, minibatch=1024),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', default='fast', choices=list(PRESETS))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--updates', type=int, default=None, help='default 3000, or unlimited if --minutes is set')
    ap.add_argument('--minutes', type=float, default=0, help='wall-clock budget; LR/entropy schedules follow it')
    ap.add_argument('--benchmark', action='store_true', help='time 4 updates, print projected total time, exit')
    ap.add_argument('--device', default='auto'); ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--games', type=int, default=None); ap.add_argument('--channels', type=int, default=None)
    ap.add_argument('--blocks', type=int, default=None); ap.add_argument('--ppo-epochs', type=int, default=None)
    ap.add_argument('--minibatch', type=int, default=None)
    ap.add_argument('--lr', type=float, default=1.5e-4); ap.add_argument('--lr-final-frac', type=float, default=0.05)
    ap.add_argument('--policy-head', default='conv', choices=['conv'])
    ap.add_argument('--gamma', type=float, default=.99); ap.add_argument('--gae-lambda', type=float, default=.95)
    ap.add_argument('--ppo-clip', type=float, default=.2); ap.add_argument('--target-kl', type=float, default=.015)
    ap.add_argument('--vf-clip', type=float, default=.5); ap.add_argument('--value-coef', type=float, default=.5)
    ap.add_argument('--entropy-start', type=float, default=.03); ap.add_argument('--entropy-end', type=float, default=.003)
    ap.add_argument('--max-open', type=int, default=4)
    ap.add_argument('--pool-every', type=int, default=25); ap.add_argument('--pool-size', type=int, default=12)
    ap.add_argument('--pool-prob', type=float, default=.25)
    ap.add_argument('--eval-every', type=int, default=200); ap.add_argument('--eval-games', type=int, default=100)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    for k, v in PRESETS[a.preset].items():
        if getattr(a, k) is None: setattr(a, k, v)
    if a.updates is None: a.updates = 10 ** 7 if a.minutes > 0 else 3000
    a.out = a.out or f'az_nosearch_v24_{a.preset}_seed{a.seed}'
    if a.threads: torch.set_num_threads(a.threads)
    dev = get_device(a.device)
    print(f'device={dev} preset={a.preset} channels={a.channels} blocks={a.blocks} games={a.games} '
          f'epochs={a.ppo_epochs} minutes={a.minutes or "-"}', flush=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    root = os.path.abspath(os.path.join(HERE, '..'))
    md, od = os.path.join(root, 'modelos'), os.path.join(root, 'resultados', a.out)
    os.makedirs(md, exist_ok=True); os.makedirs(od, exist_ok=True)

    model = AZResNet(a.channels, a.blocks, a.policy_head).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    manifest = {'algorithm': 'AZ-NoSearch-v24-ResidualCNN-PPO', 'canonical_alphazero': False,
                'mcts_used': False, 'oracle_used_for_training': False, 'search_used_for_training': False,
                'minimax_used_for_training': False, 'teacher_used_for_training': False,
                'initialization': 'random', 'training_opponents': 'current_network_and_own_snapshots_only',
                'policy_training': 'on_policy_ppo_gae_both_sides_of_self_play',
                'symmetry_augmentation': 'random_D4_view_per_decision_stored_on_policy',
                'final_inference': 'one_forward_plus_legal_argmax', 'args': vars(a)}
    json.dump(manifest, open(os.path.join(od, 'purity_manifest.json'), 'w'), indent=2)

    pool, hist, t0, u, stamps = [], [], time.time(), 0, []
    budget = a.minutes * 60
    while u < a.updates:
        u += 1
        el = time.time() - t0
        frac = max((u - 1) / max(1, a.updates - 1), (el / budget) if budget else 0.0)
        frac = min(frac, 1.0)
        lr = a.lr * (a.lr_final_frac + (1 - a.lr_final_frac) * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups: g['lr'] = lr
        ent_coef = a.entropy_start + (a.entropy_end - a.entropy_start) * frac
        tu = time.time()
        batch, selfw = collect(model, pool, a, rng, dev)
        met = ppo_update(model, opt, batch, a, ent_coef, dev)
        stamps.append(time.time() - tu)
        if u % a.pool_every == 0:
            pool = (pool + [copy.deepcopy(model).eval()])[-a.pool_size:]
        if a.benchmark and u == 4:
            spu = float(np.mean(stamps[1:]))
            print(f'BENCHMARK {spu:.2f} s/update -> 1000 updates ~ {spu*1000/60:.0f} min, '
                  f'3000 updates ~ {spu*3000/3600:.1f} h', flush=True)
            return
        if u % 10 == 0:
            print(f'[{u}] n={len(batch["a"])} lr={lr:.2e} pol={met["policy"]:.4f} val={met["value"]:.4f} '
                  f'H={met["entropy"]:.3f} kl={met["kl"]:.4f} clip={met["clipfrac"]:.3f} '
                  f'selfplay_draw={float((selfw == 0).mean()):.2f} pool={len(pool)} t={time.time()-t0:.0f}s '
                  f'frac={frac:.2f}', flush=True)
        last = (budget and time.time() - t0 >= budget) or u >= a.updates
        if u % a.eval_every == 0 or last:
            cur = as_policy(model, dev); ev = np.random.default_rng(10_000 + u)
            wr, dr_, lr_ = sv.match(cur, sv.random_policy, a.eval_games, ev, greedyB=False)
            sc = [(lambda r: (r[0] + .5 * r[1]) / a.eval_games)(sv.match(cur, as_policy(sn, dev), a.eval_games, ev))
                  for sn in pool[-2:]]
            league = float(np.mean(sc)) if sc else 0.5
            hist.append({'update': u, 'vs_random': [wr, dr_, lr_], 'league_score': league})
            print(f'   EVAL u={u} vs_random W/D/L={wr}/{dr_}/{lr_}  league={league:.3f}', flush=True)
            save(model, a, manifest, os.path.join(md, f'{a.out}_gen{u}.pt'), {'generation': u})
            json.dump(hist, open(os.path.join(od, 'history.json'), 'w'), indent=2)
        if last: break
    save(model, a, manifest, os.path.join(md, f'{a.out}_final.pt'), {'generation': u})
    print('FINISHED', a.out, 'updates=', u, 'elapsed=', round(time.time() - t0), 's  -> usa', f'{a.out}_final.pt', flush=True)


if __name__ == '__main__':
    main()

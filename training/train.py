import sys, os, pickle, time, math, argparse, numpy as np, torch, torch.nn.functional as F
pass
from pocketnet import feat, pockets
from pocketnet.model import PocketNet
ap = argparse.ArgumentParser()
ap.add_argument('--meta', required=True, help='pickle with ids and folds[i]["role"] (train/val/test per protein)'); ap.add_argument('--cache', default='data/cache'); ap.add_argument('--fold', type=int, default=0); ap.add_argument('--epochs', type=int, default=30)
ap.add_argument('--h', type=int, default=128); ap.add_argument('--layers', type=int, default=6)
ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--bs', type=int, default=8)
ap.add_argument('--pos', type=float, default=4.0, help='point positive if within this Å of ligand atom')
ap.add_argument('--steps_per_epoch', type=int, default=1500); ap.add_argument('--name', default='v1')
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--max_atoms', type=int, default=9000)
ap.add_argument('--crop', type=float, default=30.0); ap.add_argument('--full', action='store_true', help='train on train+test, keep only val for model selection'); ap.add_argument('--init', default=''); ap.add_argument('--rank_w', type=float, default=0.0); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--npts', type=int, default=3000)
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed); dev = 'cuda'
M = pickle.load(open(a.meta, 'rb')); role = M['folds'][a.fold]['role']
cache = a.cache; have = set(f[:-3] for f in os.listdir(cache))
items = {r: [] for r in ('train', 'val', 'test')}
for p, r in zip(M['ids'], role):
    if p in have: items[r].append(p)
if a.limit: items = {k: v[:a.limit] for k, v in items.items()}
print({k: len(v) for k, v in items.items()}, flush=True)
def load(pid):
    d = torch.load(f'{cache}/{pid}.pt', weights_only=False); return d
# preload everything to RAM (CPU)
t0 = time.time(); data = {}
for r in items:
    for p in items[r]:
        try: d = load(p)
        except Exception as ex: print('bad cache', p, flush=True); continue
        if d is None: continue
        data[p] = d
for r in items: items[r] = [p for p in items[r] if p in data]
if a.full: items['train'] = items['train'] + items['test']; print('FULL: train', len(items['train']), flush=True)
print('loaded', len(data), time.time() - t0, flush=True)
cl = {p: M['meta'][p]['pcluster'] for p in M['ids']}
from collections import Counter
cc = Counter(cl[p] for p in items['train']); w = np.array([1 / math.sqrt(cc[cl[p]]) for p in items['train']]); w /= w.sum()

def to_dev(d):
    return (d['ty'].long().to(dev), d['xyz'].to(dev), d['pts'].to(dev), d['pf'].float().to(dev),
            d['dl'].float().to(dev), d['da'].float().to(dev))

model = PocketNet(a.h, a.layers).to(dev)
if a.init: model.load_state_dict(torch.load(a.init, map_location=dev)); print('init from', a.init, flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
total = a.epochs * a.steps_per_epoch
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05)

def predict(d):
    ty, xyz, pts, pf, dl, da = to_dev(d)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        s, _ = model(ty, xyz, pts, pf)
    return torch.sigmoid(s.float()).cpu().numpy()

def evaluate(ids, params=dict(), maxn=200):
    model.eval(); res = []
    for p in ids[:maxn]:
        d = data[p]
        if len(d['ty']) > 14000: continue
        try: s = predict(d)
        except torch.OutOfMemoryError: torch.cuda.empty_cache(); continue
        pk = pockets.extract(d['pts'].numpy(), s, **params)
        res.append(pockets.evaluate_one(pk, d['lig'].numpy()))
    model.train()
    return {k: float(np.mean([r[k] for r in res])) for k in ('top1', 'top3', 'topany', 'dcc1')} | {'n': len(res)}

step = 0; t0 = time.time(); model.train(); best = -1; log = open(f'runs/{a.name}_fold{a.fold}.log', 'a')
def P(*x):
    s = ' '.join(str(i) for i in x); print(s, flush=True); log.write(s + '\n'); log.flush()
P(vars(a))
for ep in range(a.epochs):
    tl = 0.0; ta = 0.0; nooom = 0
    for it in range(a.steps_per_epoch):
        try:
            ids = np.random.choice(len(items['train']), a.bs, p=w)
            opt.zero_grad(set_to_none=True); samples = []
            for i in ids:
                d = data[items['train'][i]]
                ty, xyz, pts, pf, dl, da = to_dev(d)
                if len(xyz) > a.max_atoms:   # crop to a sphere around a surface point (half the time near the ligand) instead of dropping
                    pos = (dl < a.pos).nonzero()[:, 0]
                    c = pts[pos[torch.randint(len(pos), (1,), device=dev)]][0] if len(pos) and np.random.rand() < 0.5 else pts[torch.randint(len(pts), (1,), device=dev)][0]
                    ka = (xyz - c).norm(dim=1) < a.crop; kp = (pts - c).norm(dim=1) < a.crop - 8
                    ty, xyz, da, pts, pf, dl = ty[ka], xyz[ka], da[ka], pts[kp], pf[kp], dl[kp]
                    if len(pts) < 20: continue
                if len(pts) > a.npts:
                    sel = torch.randperm(len(pts), device=dev)[:a.npts]; pts, pf, dl = pts[sel], pf[sel], dl[sel]
                samples.append((len(xyz), model.graph(ty, xyz, pts, pf), (dl < a.pos).float(), (da < 4.5).float()))
            if not samples: continue
            groups, cur, n = [], [], 0
            for s_i in samples:
                if cur and n + s_i[0] > a.max_atoms: groups.append(cur); cur, n = [], 0
                cur.append(s_i); n += s_i[0]
            groups.append(cur); tot = sum(len(s_i[2]) for s_i in samples)
            for gr in groups:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    s_, al = model.forward_graph(model.collate([g[1] for g in gr]))
                y = torch.cat([g[2] for g in gr]); ya = torch.cat([g[3] for g in gr]); fr = len(y) / tot
                l1 = F.binary_cross_entropy_with_logits(s_.float(), y, pos_weight=torch.tensor(3.0, device=dev))
                l2 = F.binary_cross_entropy_with_logits(al.float(), ya)
                lr_ = 0.0
                if a.rank_w > 0:   # first-pick loss: probability mass of a softmax over points that lands on ligand-contact points
                    o = 0
                    for g in gr:
                        n_ = len(g[2]); sl = s_[o:o + n_].float(); yy = g[2]; o += n_
                        if yy.sum() > 0: lr_ = lr_ - torch.logsumexp(sl[yy > 0], 0) + torch.logsumexp(sl, 0)
                    lr_ = lr_ / len(gr)
                ((l1 + 0.3 * l2 + a.rank_w * lr_) * fr).backward(); tl += l1.item() * fr; ta += l2.item() * fr
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step(); sched.step(); step += 1
        except torch.OutOfMemoryError:
            nooom += 1; opt.zero_grad(set_to_none=True); samples = groups = s_ = al = l1 = l2 = None; torch.cuda.empty_cache(); sched.step(); step += 1
    m = evaluate(items['val'])
    P(f'ep {ep} step {step} loss {tl/a.steps_per_epoch:.4f} atom {ta/a.steps_per_epoch:.4f} oom {nooom} val {m} t {time.time()-t0:.0f}s')
    score = m['top1'] + m['top3']
    if score > best: best = score; torch.save(model.state_dict(), f'runs/{a.name}_fold{a.fold}.pt')
P('best val', best)

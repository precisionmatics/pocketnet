"""Turn per-point scores into ranked pockets; DCA/DCC metrics (P2Rank convention)."""
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

def extract(pts, s, thr_q=0.97, link=5.0, topn=10, max_pts=600, method='average', top_k=12, min_pts=3):
    """pts [P,3] numpy, s [P] scores -> list of (score, center[3]) sorted desc."""
    if len(pts) < 3: return []
    k = max(min_pts, min(max_pts, int(round(len(pts) * (1 - thr_q)))))
    sel = np.argsort(-s)[:k]; p, v = pts[sel], s[sel]
    if len(p) < 2: return [(float(v.sum()), p[0])]
    lab = fcluster(linkage(p, method), link, 'distance'); out = []
    for c in np.unique(lab):
        m = lab == c
        if m.sum() < min_pts: continue
        vc, pc = v[m], p[m]; o = np.argsort(-vc)[:top_k]
        w = vc[o] - vc.min() + 1e-6 if False else np.maximum(vc[o], 1e-6)
        out.append((float(vc[o].sum()), (pc[o] * w[:, None]).sum(0) / w.sum()))
    out.sort(key=lambda t: -t[0]); return out[:topn]

def evaluate_one(pockets, lig, n_lig=1, thr=4.0):
    """DCA success@n, @n+2 and DCC; pockets list of (score, center); lig [A,3]."""
    if not pockets: return dict(top1=0, top3=0, topany=0, dcc1=0, dist=np.inf, found=0)
    c = np.stack([p[1] for p in pockets]); d = np.linalg.norm(c[:, None, :] - lig[None], axis=-1).min(1)
    dcc = np.linalg.norm(c - lig.mean(0), axis=-1)
    return dict(top1=float(d[:n_lig].min() <= thr), top3=float(d[:n_lig + 2].min() <= thr), topany=float(d.min() <= thr),
                dcc1=float(dcc[:n_lig].min() <= thr), dist=float(d[0]), found=len(pockets))


TUNED = dict(thr_q=0.97, link=6.5, max_pts=300, top_k=24)

def extract_with_members(pts, s, thr_q=0.97, link=6.5, topn=10, max_pts=300, method='average', top_k=24, min_pts=3):
    """Same clustering as `extract`, but also returns each pocket's member point indices (into pts).
    -> list of dict(score, center[3], idx[int array], top_idx[int array])."""
    if len(pts) < 3: return []
    k = max(min_pts, min(max_pts, int(round(len(pts) * (1 - thr_q)))))
    sel = np.argsort(-s)[:k]; p, v = pts[sel], s[sel]
    if len(p) < 2: return [dict(score=float(v.sum()), center=p[0], idx=sel, top_idx=sel)]
    lab = fcluster(linkage(p, method), link, 'distance'); out = []
    for c in np.unique(lab):
        m = lab == c
        if m.sum() < min_pts: continue
        vc, pc, ic = v[m], p[m], sel[m]; o = np.argsort(-vc)[:top_k]
        w = np.maximum(vc[o], 1e-6)
        out.append(dict(score=float(vc[o].sum()), center=(pc[o] * w[:, None]).sum(0) / w.sum(), idx=ic, top_idx=ic[o]))
    out.sort(key=lambda t: -t['score']); return out[:topn]

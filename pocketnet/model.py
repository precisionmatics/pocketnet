import math, torch, torch.nn as nn, torch.nn.functional as F
from . import feat

class RBF(nn.Module):
    def __init__(self, n, lo, hi):
        super().__init__(); self.register_buffer('mu', torch.linspace(lo, hi, n)); self.g = (hi - lo) / n
    def forward(self, d): return torch.exp(-((d[..., None] - self.mu) / self.g) ** 2)

class Layer(nn.Module):
    def __init__(self, h, ne):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(2 * h + ne, h), nn.SiLU(), nn.Linear(h, h))
        self.gate = nn.Linear(2 * h + ne, h)
        self.upd = nn.Sequential(nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, h))
        self.n1 = nn.LayerNorm(h)
    def forward(self, h, ei, ea):
        i, j = ei; z = torch.cat([h[i], h[j], ea], 1)
        m = self.msg(z) * torch.sigmoid(self.gate(z))
        agg = torch.zeros_like(h, dtype=m.dtype).index_add_(0, i, m) / 12.0
        return self.n1(h + self.upd(torch.cat([h, agg], 1)))

class PocketNet(nn.Module):
    def __init__(self, h=128, layers=6, k_atom=24, k_pt=32, r_atom=6.0, r_pt=10.0, drop=0.1):
        super().__init__()
        self.k_atom, self.k_pt, self.r_atom, self.r_pt = k_atom, k_pt, r_atom, r_pt
        self.emb = nn.Embedding(feat.N_TYPES, h); self.bur = nn.Linear(4, h)
        self.rbf = RBF(16, 0, r_atom); self.layers = nn.ModuleList([Layer(h, 16) for _ in range(layers)])
        self.atom_head = nn.Linear(h, 1)
        hp = 64
        self.proj = nn.Linear(h, hp)
        self.rax = RBF(12, -r_pt, r_pt); self.rrad = RBF(10, 0, r_pt)
        self.att = nn.Sequential(nn.Linear(22 + hp, 128), nn.SiLU(), nn.Linear(128, 4))     # 4 attention heads
        self.val = nn.Linear(hp + 22, hp)
        self.pfn = nn.Sequential(nn.Linear(feat.N_PF, 128), nn.SiLU(), nn.Linear(128, 128))
        self.out = nn.Sequential(nn.Linear(4 * hp + 128 + hp, 256), nn.SiLU(), nn.Dropout(drop), nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))

    @torch.no_grad()
    def graph(self, ty, xyz, pts, pf):
        """Per-protein geometry: everything that does not need gradients."""
        d, idx = feat.knn(xyz, xyz, self.k_atom + 1)
        d, idx = d[:, 1:], idx[:, 1:]; m = d < self.r_atom
        i = torch.arange(len(xyz), device=xyz.device)[:, None].expand_as(idx)
        cnt = torch.stack([(d < 4).sum(1), (d < 6).sum(1), (d < 3).sum(1), (d < 5).sum(1)], 1).float() / 24.0
        d1, i1 = feat.knn(pts, xyz, 1); nrm = pts - xyz[i1[:, 0]]; nrm = nrm / nrm.norm(dim=1, keepdim=True).clamp(min=1e-6)
        dp, ip = feat.knn(pts, xyz, self.k_pt)
        rel = xyz[ip] - pts[:, None, :]
        ax = (rel * nrm[:, None, :]).sum(-1); rad = (dp ** 2 - ax ** 2).clamp(min=0).sqrt()
        return dict(ty=ty, cnt=cnt, ei=torch.stack([i[m], idx[m]]), d=d[m], i1=i1[:, 0], ip=ip, ax=ax, rad=rad, dp=dp, pf=pf, na=len(xyz), npt=len(pts))

    def collate(self, gs):
        off = 0; ei = []; i1 = []; ip = []; o = 0
        for g in gs:
            ei.append(g['ei'] + off); i1.append(g['i1'] + off); ip.append(g['ip'] + off); off += g['na']
        cat = lambda k: torch.cat([g[k] for g in gs])
        return dict(ty=cat('ty'), cnt=cat('cnt'), ei=torch.cat(ei, 1), d=cat('d'), i1=torch.cat(i1), ip=torch.cat(ip), ax=cat('ax'), rad=cat('rad'),
                    dp=cat('dp'), pf=cat('pf'), sizes=[g['npt'] for g in gs])

    def forward_graph(self, g):
        h = self.emb(g['ty']) + self.bur(g['cnt']); ea = self.rbf(g['d'])
        for l in self.layers: h = l(h, g['ei'], ea)
        a_logit = self.atom_head(h)[:, 0]; hp = self.proj(h)
        ge = torch.cat([self.rax(g['ax']), self.rrad(g['rad'])], -1)
        m = g['dp'] < self.r_pt; hj = hp[g['ip']]
        a = self.att(torch.cat([ge, hj], -1)).masked_fill(~m[..., None], -1e4).softmax(1)
        v = self.val(torch.cat([hj, ge], -1))
        pooled = torch.einsum('pkh,pkv->phv', a, v).reshape(len(g['pf']), -1)
        z = torch.cat([pooled, self.pfn(g['pf']), hp[g['i1']]], 1)
        return self.out(z)[:, 0], a_logit

    def forward(self, ty, xyz, pts, pf):
        return self.forward_graph(self.collate([self.graph(ty, xyz, pts, pf)]))

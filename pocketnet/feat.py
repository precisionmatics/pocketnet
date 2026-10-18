"""Protein featurisation for pocket detection. Everything heavy runs on GPU (torch)."""
import math, numpy as np, torch

AA = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']
RMAP = {'HIE':'HIS','HID':'HIS','HIP':'HIS','HSD':'HIS','HSE':'HIS','HSP':'HIS','CYX':'CYS','CYM':'CYS','ASH':'ASP','GLH':'GLU','LYN':'LYS','MSE':'MET'}
ATOMS = {  # standard heavy-atom names per residue (PDB naming)
 'ALA':'N CA C O CB','ARG':'N CA C O CB CG CD NE CZ NH1 NH2','ASN':'N CA C O CB CG OD1 ND2','ASP':'N CA C O CB CG OD1 OD2',
 'CYS':'N CA C O CB SG','GLN':'N CA C O CB CG CD OE1 NE2','GLU':'N CA C O CB CG CD OE1 OE2','GLY':'N CA C O',
 'HIS':'N CA C O CB CG ND1 CD2 CE1 NE2','ILE':'N CA C O CB CG1 CG2 CD1','LEU':'N CA C O CB CG CD1 CD2',
 'LYS':'N CA C O CB CG CD CE NZ','MET':'N CA C O CB CG SD CE','PHE':'N CA C O CB CG CD1 CD2 CE1 CE2 CZ',
 'PRO':'N CA C O CB CG CD','SER':'N CA C O CB OG','THR':'N CA C O CB OG1 CG2','TRP':'N CA C O CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2',
 'TYR':'N CA C O CB CG CD1 CD2 CE1 CE2 CZ OH','VAL':'N CA C O CB CG1 CG2'}
VOCAB = {}
for r in AA:
    for a in ATOMS[r].split(): VOCAB[(r, a)] = len(VOCAB) + 1       # 0 = unknown
for r in AA: VOCAB[(r, 'OXT')] = len(VOCAB) + 1
N_TYPES = len(VOCAB) + 2                                           # +unknown(0) +other-residue(last)
OTHER = N_TYPES - 1
ALIAS = {'OC1':'O','OC2':'OXT','O1':'O','O2':'OXT'}
EL = {'C':0,'N':1,'O':2,'S':3,'P':4}
VDW = torch.tensor([1.70, 1.55, 1.52, 1.80, 1.80, 1.80])          # C N O S P other

def atom_types(res, name):
    out = np.zeros(len(res), np.int64); el = np.zeros(len(res), np.int64)
    for i, (r, a) in enumerate(zip(res, name)):
        r = RMAP.get(r, r); a = ALIAS.get(a, a)
        out[i] = VOCAB.get((r, a), 0) if r in ATOMS else OTHER
        el[i] = EL.get(a[0], 5)
    return out, el

def fib_sphere(n, device):
    i = torch.arange(n, device=device, dtype=torch.float32) + 0.5
    phi = torch.acos(1 - 2 * i / n); th = math.pi * (1 + 5 ** 0.5) * i
    return torch.stack([torch.cos(th) * torch.sin(phi), torch.sin(th) * torch.sin(phi), torch.cos(phi)], 1)

@torch.no_grad()
def sas_points(xyz, el, probe=1.4, n_sphere=48, chunk=4096):
    """Shrake-Rupley points on the solvent-accessible surface. Returns pts [P,3], normals [P,3] (pointing out)."""
    dev = xyz.device; r = VDW.to(dev)[el] + probe
    sph = fib_sphere(n_sphere, dev)
    pts = (xyz[:, None, :] + r[:, None, None] * sph[None]).reshape(-1, 3)
    own = torch.arange(len(xyz), device=dev).repeat_interleave(n_sphere)
    keep = torch.ones(len(pts), dtype=torch.bool, device=dev)
    for s in range(0, len(pts), chunk):
        d = torch.cdist(pts[s:s+chunk], xyz)                                # [c,N]
        inside = d < (r[None] - 1e-3)
        inside[torch.arange(inside.shape[0], device=dev), own[s:s+chunk]] = False
        keep[s:s+chunk] = ~inside.any(1)
    pts = pts[keep]; nrm = (pts - xyz[own[keep]]); nrm = nrm / nrm.norm(dim=1, keepdim=True)
    return pts, nrm

@torch.no_grad()
def occupancy_grid(xyz, el, cell=0.8, pad=18.0):
    dev = xyz.device; lo = xyz.min(0).values - pad; hi = xyz.max(0).values + pad
    dims = ((hi - lo) / cell).ceil().long() + 1
    grid = torch.zeros(tuple(dims.tolist()), dtype=torch.bool, device=dev)
    rr = (VDW.to(dev)[el] + 0.2)                                          # slightly inflated
    m = int(math.ceil(2.1 / cell)); rng = torch.arange(-m, m + 1, device=dev)
    off = torch.stack(torch.meshgrid(rng, rng, rng, indexing='ij'), -1).reshape(-1, 3)   # [K,3]
    for s in range(0, len(xyz), 2048):
        a = xyz[s:s+2048]; c = ((a - lo) / cell).round().long()            # [n,3]
        idx = c[:, None, :] + off[None]; pos = lo + idx.float() * cell
        ok = ((pos - a[:, None, :]).norm(dim=-1) <= rr[s:s+2048, None])
        idx = idx[ok]; grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return grid, lo, cell

@torch.no_grad()
def point_feats(pts, nrm, xyz, el, aa_sel=None, n_rays=64, step=0.8, rmax=16.0):
    """Rotation-invariant per-point descriptors: ray-cast burial (all dirs + cones about the normal), shell counts."""
    dev = pts.device
    grid, lo, cell = occupancy_grid(xyz, el)
    rays = fib_sphere(n_rays, dev)                                         # [R,3]
    ns = int(rmax / step); t = (torch.arange(1, ns + 1, device=dev, dtype=torch.float32) * step)
    P = len(pts); fh = torch.empty(P, n_rays, device=dev)                  # first-hit distance
    for s in range(0, P, 2048):
        p = pts[s:s+2048]                                                  # [c,3]
        q = p[:, None, None, :] + rays[None, :, None, :] * t[None, None, :, None]    # [c,R,S,3]
        idx = ((q - lo) / cell).round().long()
        idx = torch.minimum(torch.maximum(idx, torch.zeros_like(idx)), torch.tensor(grid.shape, device=dev) - 1)
        occ = grid[idx[..., 0], idx[..., 1], idx[..., 2]]                  # [c,R,S]
        first = torch.where(occ.any(-1), occ.float().argmax(-1), torch.full_like(occ[..., 0], ns, dtype=torch.long))
        fh[s:s+2048] = torch.where(first >= ns, torch.full_like(first, rmax * 2, dtype=torch.float32), t[first.clamp(max=ns - 1)])
    cosang = nrm @ rays.T                                                  # [P,R]
    f = []
    for rng in (4.0, 8.0, 12.0, 16.0):
        b = (fh <= rng).float(); f.append(b.mean(1))
        for cth in (0.866, 0.5, 0.0):                                      # cones 30/60/90 deg about normal
            m = (cosang >= cth).float(); f.append((b * m).sum(1) / m.sum(1).clamp(min=1))
    f.append(fh.clamp(max=rmax).mean(1) / rmax)
    f.append(fh.clamp(max=rmax).min(1).values / rmax)
    # opposite-hemisphere enclosure (pocket depth): distance to first hit along -normal cone
    m = (cosang <= -0.5).float(); f.append(((fh.clamp(max=rmax) * m).sum(1) / m.sum(1).clamp(min=1)) / rmax)
    # shell counts by element class
    d = torch.cdist(pts, xyz); eo = torch.nn.functional.one_hot(el, 6).float()    # [N,6]
    for lo_, hi_ in ((0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 12.0)):
        w = ((d >= lo_) & (d < hi_)).float(); f.append((w.sum(1, keepdim=True) / 30.0)[:, 0])
        c = w @ eo; f.extend([c[:, k] / 12.0 for k in range(4)])           # C N O S
    return torch.stack(f, 1)                                               # [P, ~32]
N_PF = 4 * 4 + 3 + 4 * 5

def knn(x, y, k, chunk=2048):
    """for each row of x the k nearest rows of y -> (dist [n,k], idx [n,k])"""
    ds, ix = [], []
    for s in range(0, len(x), chunk):
        d = torch.cdist(x[s:s+chunk], y); v, i = d.topk(min(k, y.shape[0]), largest=False); ds.append(v); ix.append(i)
    return torch.cat(ds), torch.cat(ix)

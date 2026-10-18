import sys, os, glob, time, numpy as np, torch
pass
from pocketnet import feat
K0, K1 = int(sys.argv[1]), int(sys.argv[2])
R = 'data/raw'; C = 'data/cache'; dev = 'cuda'
def done_all(): return len(open('logs/done_tars.txt').read().split()) >= 102
while True:
    fin = done_all(); n = 0
    for ii, f in enumerate(sorted(glob.glob(f'{R}/*.npz'))):
        if ii % K1 != K0: continue
        pid = os.path.basename(f)[:-4]; out = f'{C}/{pid}.pt'
        if os.path.exists(out) or os.path.getmtime(f) > time.time() - 5: continue
        try:
            d = np.load(f); ty, el = feat.atom_types(d['pres'], d['pname'])
            if not (100 <= len(ty) <= 14000): torch.save(None, out); continue
            x = torch.tensor(d['pxyz0'], device=dev); e = torch.tensor(el, device=dev); l = torch.tensor(d['lxyz0'], device=dev)
            pts, nrm = feat.sas_points(x, e); pf = feat.point_feats(pts, nrm, x, e)
            dl = torch.cdist(pts, l).min(1).values; da = torch.cdist(x, l).min(1).values
            torch.cuda.empty_cache()
            torch.save(dict(pid=pid, ty=torch.tensor(ty, dtype=torch.int16), el=torch.tensor(el, dtype=torch.int8), xyz=x.cpu(),
                            da=da.half().cpu(), pts=pts.cpu(), pf=pf.half().cpu(), dl=dl.half().cpu(), lig=l.cpu(),
                            rid=torch.tensor(d['prid']), chain=torch.tensor(d['pchain'])), out + '.tmp'); os.replace(out + '.tmp', out); n += 1
        except Exception as ex:
            print('ERR', f, ex, flush=True); torch.save(None, out)
    print(time.strftime('%H:%M:%S'), 'cached', n, 'total', len(os.listdir(C)), flush=True)
    if fin and n == 0: break
    torch.cuda.empty_cache()
    if n == 0: time.sleep(120)

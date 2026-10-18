"""Stream PLAS-20K tarballs -> per-complex npz (protein heavy atoms + ligand heavy atoms, run0 frames 0 and 39)."""
import sys, tarfile, io, numpy as np, glob, os, re
SRC = os.environ.get('PLAS20K_DIR', 'PLAS-20K')  # directory with the PLAS-20K tarballs
OUT = os.environ.get('RAW_DIR', 'data/raw')
os.makedirs(OUT, exist_ok=True); os.makedirs('logs', exist_ok=True)
FRAMES = (0, 39)

def parse(text):
    frames, cur = [], None
    for ln in text.split('\n'):
        if ln.startswith('MODEL'): cur = []; chain = 0
        elif ln.startswith('ENDMDL'):
            frames.append(cur); cur = None
        elif cur is not None:
            if ln.startswith('TER'): chain += 1
            elif ln.startswith(('ATOM', 'HETATM')):
                el = ln[76:78].strip() or ln[12:16].strip()[0]
                if el in ('H', 'D') or ln[17:20] == 'WAT': continue
                cur.append((ln[12:16].strip(), ln[17:20].strip(), int(ln[22:26]), chain, el,
                            float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
    return frames

def save(pid, frames):
    d = {}
    for fi in FRAMES:
        if fi >= len(frames): continue
        a = frames[fi]
        lig = [x for x in a if x[1] == 'LIG']; pro = [x for x in a if x[1] != 'LIG']
        if not lig or len(pro) < 50: return False
        d[f'pxyz{fi}'] = np.array([x[5:8] for x in pro], np.float32)
        d[f'lxyz{fi}'] = np.array([x[5:8] for x in lig], np.float32)
        if fi == FRAMES[0]:
            d['pname'] = np.array([x[0] for x in pro]); d['pres'] = np.array([x[1] for x in pro])
            d['prid'] = np.array([x[2] for x in pro], np.int32); d['pchain'] = np.array([x[3] for x in pro], np.int8)
            d['pel'] = np.array([x[4] for x in pro]); d['lel'] = np.array([x[4] for x in lig])
            d['lname'] = np.array([x[0] for x in lig])
    np.savez_compressed(f'{OUT}/{pid}.npz', **d); return True

def run(tar):
    n = 0
    try:
      with tarfile.open(tar, 'r|gz') as t:
        for m in t:
              if not (m.isfile() and m.name.endswith('expli_2wat_0.pdb')): continue
              pid = m.name.split('/')[-2].lower()
              if os.path.exists(f'{OUT}/{pid}.npz'): continue
              try: ok = save(pid, parse(t.extractfile(m).read().decode()))
              except Exception as e: print('ERR', pid, e, flush=True); ok = False
              n += ok
    except Exception as e: print('TARERR', tar, e, flush=True)
    print('done', os.path.basename(tar), n, flush=True)
    open('logs/done_tars.txt', 'a').write(os.path.basename(tar) + '\n')

if __name__ == '__main__':
    k, K = int(sys.argv[1]), int(sys.argv[2])
    tars = sorted(glob.glob(f'{SRC}/*Chunk-*.tar.gz'))
    done = set(open('logs/done_tars.txt').read().split())
    tars = [t for t in tars if os.path.basename(t) not in done]
    for i, t in enumerate(tars):
        if i % K == k: run(t)

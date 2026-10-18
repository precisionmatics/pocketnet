import numpy as np
from . import feat
def read_pdb(path, lig_codes=None):
    """-> dict(res,name,rid,chain,xyz) of protein heavy atoms (first model, altloc A/blank) and list of ligand coord arrays."""
    prot, ligs = [], {}
    with open(path) as f:
        for ln in f:
            if ln.startswith('ENDMDL'): break
            rec = ln[:6]
            if rec not in ('ATOM  ', 'HETATM'): continue
            if ln[16] not in ' A1': continue
            el = (ln[76:78].strip() or ln[12:16].strip()[:1]).upper()
            if el in ('H', 'D'): continue
            res = ln[17:20].strip(); xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
            if rec == 'ATOM  ' or (res == 'MSE'):
                prot.append((ln[12:16].strip(), res, int(ln[22:26]) if ln[22:26].strip().lstrip('-').isdigit() else 0, ln[21], el, *xyz))
            elif lig_codes and res in lig_codes:
                ligs.setdefault((res, ln[21], ln[22:27]), []).append(xyz)
    P = dict(name=np.array([p[0] for p in prot]), res=np.array([p[1] for p in prot]), rid=np.array([p[2] for p in prot]),
             chain=np.array([p[3] for p in prot]), xyz=np.array([p[5:8] for p in prot], np.float32))
    return P, [np.array(v, np.float32) for v in ligs.values()]

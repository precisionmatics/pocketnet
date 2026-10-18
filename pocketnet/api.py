"""High-level interface: a five-model PocketNet ensemble that turns a PDB file into ranked pockets."""
from __future__ import annotations
import json, os, tempfile
from pathlib import Path
import numpy as np, torch
from . import feat, pockets
from .model import PocketNet
from .pdbio import read_pdb

_HERE = Path(__file__).resolve().parent
WEIGHTS = sorted((_HERE / "weights").glob("v3_fold*.pt"))
MAX_ATOMS = 14000   # structures above this size were not used in training; larger inputs are accepted but untested


def _load(path, device):
    sd = torch.load(path, map_location=device, weights_only=True)
    hidden = sd["emb.weight"].shape[1]
    layers = len([k for k in sd if k.startswith("layers.") and k.endswith("msg.0.weight")])
    m = PocketNet(hidden, layers).to(device)
    m.load_state_dict(sd)
    return m.eval()


class PocketPredictor:
    """Averages the sigmoid scores of the five fold models and extracts at most ten ranked pockets.

    Parameters
    ----------
    device : "cuda", "cpu" or None (cuda if available).
    weights : optional list of checkpoint paths (default: the five bundled fold models).
    """

    def __init__(self, device: str | None = None, weights=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.models = [_load(w, self.device) for w in (weights or WEIGHTS)]
        cal = json.loads((_HERE / "calibration.json").read_text())
        self._cal_x, self._cal_y = np.asarray(cal["x"]), np.asarray(cal["y"])

    @torch.no_grad()
    def score_points(self, structure):
        """Return (surface_points [P,3], ensemble_score [P]) for a PDB file path or a PDB-format string."""
        P = self._read(structure)
        dev = self.device
        ty, el = feat.atom_types(P["res"], P["name"])
        x = torch.tensor(P["xyz"], device=dev); e = torch.tensor(el, device=dev); t = torch.tensor(ty, device=dev)
        pts, nrm = feat.sas_points(x, e)
        pf = feat.point_feats(pts, nrm, x, e)
        s = 0
        for m in self.models:
            if dev == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lo, _ = m(t, x, pts, pf)
            else:
                lo, _ = m(t, x, pts, pf)
            s = s + torch.sigmoid(lo.float())
        return pts.cpu().numpy(), (s / len(self.models)).cpu().numpy(), P

    def predict(self, structure, max_pockets: int = 10, residue_radius: float = 8.0):
        """Ranked pockets for a PDB file (path or text).

        Each pocket is a dict with ``rank``, ``score`` (sum of its 24 highest point scores), ``probability``
        (isotonic calibration of the score on held-out proteins; indicative only), ``center`` (x, y, z in angstrom),
        ``n_points`` and ``residues`` (chain:number of residues with an atom within ``residue_radius`` of the centre).
        """
        pts, s, P = self.score_points(structure)
        found = pockets.extract_with_members(pts, s, topn=max_pockets, **pockets.TUNED)
        out = []
        for rank, c in enumerate(found, 1):
            ctr = np.asarray(c["center"], float)
            near = np.linalg.norm(P["xyz"] - ctr[None], axis=1) <= residue_radius
            res = sorted({f"{P['chain'][j]}:{P['rid'][j]}" for j in np.where(near)[0]})
            out.append(dict(rank=rank, score=float(c["score"]), probability=float(np.interp(c["score"], self._cal_x, self._cal_y)),
                            center=[round(float(v), 3) for v in ctr], n_points=int(len(c["idx"])), residues=res))
        return out

    def _read(self, structure):
        text = str(structure)
        if "\n" in text or text.lstrip().startswith(("ATOM", "HEADER", "REMARK", "CRYST1", "HETATM")):
            with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as f:
                f.write(text); name = f.name
            try: P, _ = read_pdb(name)
            finally: os.unlink(name)
        else:
            P, _ = read_pdb(text)
        if len(P["xyz"]) < 30:
            raise ValueError("fewer than 30 protein atoms were read (only ATOM records of standard residues are used; mmCIF is not supported)")
        return P


def predict(structure, device=None, **kw):
    """One-shot convenience wrapper around :class:`PocketPredictor`."""
    return PocketPredictor(device).predict(structure, **kw)

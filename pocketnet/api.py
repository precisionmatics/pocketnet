"""High-level interface: a five-model PocketNet ensemble that turns a structure into ranked pockets."""
from __future__ import annotations
import json, logging, os
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Union
import numpy as np, torch
from . import feat, pockets as _pk
from .model import PocketNet
from .results import Pocket, PredictionResult
from .structure import Structure, fetch_alphafold, fetch_pdb, looks_like_pdb_id, looks_like_uniprot, read_structure

log = logging.getLogger("pocketnet")
_HERE = Path(__file__).resolve().parent
WEIGHTS = sorted((_HERE / "weights").glob("v3_fold*.pt"))
MAX_ATOMS = 14000   # structures above this size were not used in training; larger inputs are accepted but untested
DEFAULTS = dict(thr_q=0.97, link=6.5, max_pts=300, top_k=24)   # extraction settings used in the paper


def _load(path, device) -> PocketNet:
    sd = torch.load(path, map_location=device, weights_only=True)
    hidden = sd["emb.weight"].shape[1]; layers = len([k for k in sd if k.startswith("layers.") and k.endswith("msg.0.weight")])
    m = PocketNet(hidden, layers).to(device); m.load_state_dict(sd); return m.eval()


def resolve(structure, chains: Optional[Sequence[str]] = None, fetch: bool = True) -> Structure:
    """Turn a path, PDB text, PDB identifier (``3POZ``), UniProt accession (AlphaFold model) or :class:`Structure` into a :class:`Structure`."""
    if isinstance(structure, Structure): return structure.select_chains(chains)
    s = str(structure)
    if fetch and "\n" not in s and not Path(s).expanduser().exists():
        if looks_like_pdb_id(s): return read_structure(fetch_pdb(s), chains)
        if looks_like_uniprot(s): return read_structure(fetch_alphafold(s), chains)
    return read_structure(s, chains)


class PocketPredictor:
    """Averages the sigmoid scores of the five fold models and extracts at most ten ranked pockets.

    Parameters
    ----------
    device : ``"cuda"``, ``"cpu"`` or None (CUDA if available).
    weights : optional list of checkpoint paths (default: the five bundled fold models).
    **extraction : overrides of ``thr_q`` (fraction of points kept is ``1 - thr_q``), ``link`` (clustering cut, angstrom),
        ``max_pts`` and ``top_k``; the defaults are the settings used in the paper.

    Examples
    --------
    >>> from pocketnet import PocketPredictor
    >>> result = PocketPredictor().predict("3POZ")      # downloads from the PDB; a local file path works too
    >>> print(result)
    >>> result.best.center, result.best.residues[:5]
    """

    def __init__(self, device: Optional[str] = None, weights: Optional[Sequence[os.PathLike]] = None, **extraction):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        paths = list(weights or WEIGHTS)
        if not paths: raise FileNotFoundError("no model weights found")
        unknown = set(extraction) - set(DEFAULTS)
        if unknown: raise TypeError(f"unknown extraction setting(s): {sorted(unknown)}")
        self.params = {**DEFAULTS, **extraction}
        self.models = [_load(w, self.device) for w in paths]
        cal = json.loads((_HERE / "calibration.json").read_text()); self._cal_x, self._cal_y = np.asarray(cal["x"]), np.asarray(cal["y"])
        log.info("loaded %d models on %s", len(self.models), self.device)

    def __repr__(self) -> str: return f"PocketPredictor(models={len(self.models)}, device={self.device!r}, params={self.params})"

    @torch.no_grad()
    def score_points(self, structure, chains: Optional[Sequence[str]] = None):
        """Return ``(points [P, 3], ensemble score [P], Structure)``."""
        st = resolve(structure, chains)
        if len(st) < 30: raise ValueError("fewer than 30 protein atoms were read")
        if len(st) > MAX_ATOMS: log.warning("%d atoms: structures above %d atoms were not used in training", len(st), MAX_ATOMS)
        dev = self.device; ty, el = feat.atom_types(st.res, st.name)
        x = torch.tensor(st.xyz, device=dev); e = torch.tensor(el, device=dev); t = torch.tensor(ty, device=dev)
        pts, nrm = feat.sas_points(x, e); pf = feat.point_feats(pts, nrm, x, e); s = 0
        for m in self.models:
            if dev == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16): lo, _ = m(t, x, pts, pf)
            else: lo, _ = m(t, x, pts, pf)
            s = s + torch.sigmoid(lo.float())
        return pts.cpu().numpy(), (s / len(self.models)).cpu().numpy(), st

    def predict(self, structure, chains: Optional[Sequence[str]] = None, max_pockets: int = 10, residue_radius: float = 8.0) -> PredictionResult:
        """Rank the pockets of a structure.

        ``structure`` is a file path (PDB or mmCIF, optionally gzipped), PDB-format text, a PDB identifier, a UniProt accession
        (AlphaFold model) or a :class:`pocketnet.structure.Structure`. ``chains`` restricts the prediction to some chains.
        Each :class:`Pocket` has ``rank``, ``score``, ``probability``, ``center``, ``n_points`` and ``residues`` (``chain:number`` of residues with an
        atom within ``residue_radius`` of the centre).
        """
        pts, s, st = self.score_points(structure, chains)
        found = _pk.extract_with_members(pts, s, topn=max_pockets, **self.params); out = []; keys = st.residue_keys
        for rank, c in enumerate(found, 1):
            ctr = np.asarray(c["center"], float); near = np.linalg.norm(st.xyz - ctr[None], axis=1) <= residue_radius
            res = list(dict.fromkeys(keys[near].tolist()))
            out.append(Pocket(rank=rank, score=float(c["score"]), probability=float(np.interp(c["score"], self._cal_x, self._cal_y)),
                              center=[round(float(v), 3) for v in ctr], n_points=int(len(c["idx"])), residues=sorted(res, key=_res_sort), point_indices=[int(i) for i in c["idx"]]))
        src = st.source if st.source != "<text>" else ""
        return PredictionResult(out, pts, s, st, src)

    def predict_many(self, structures: Iterable, **kw) -> Iterator[tuple[str, Optional[PredictionResult], Optional[Exception]]]:
        """Predict a series of structures; yields ``(name, result, error)`` and continues after a failure."""
        for s in structures:
            name = str(s) if not isinstance(s, Structure) else s.source
            try: yield name, self.predict(s, **kw), None
            except Exception as ex:   # noqa: BLE001 - report and continue with the next structure
                log.warning("%s failed: %s", name, ex); yield name, None, ex


def _res_sort(k: str):
    c, n = k.split(":", 1); digits = "".join(ch for ch in n if ch.lstrip("-").isdigit() or ch == "-")
    try: num = int(digits)
    except ValueError: num = 0
    return (c, num, n)


def predict(structure, device: Optional[str] = None, **kw) -> PredictionResult:
    """One-shot convenience wrapper around :class:`PocketPredictor`."""
    return PocketPredictor(device).predict(structure, **kw)

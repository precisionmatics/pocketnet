"""Result objects returned by :class:`pocketnet.PocketPredictor` and their exporters."""
from __future__ import annotations
import csv, io, json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, Optional
import numpy as np
from .structure import Structure


@dataclass
class Pocket:
    """One predicted pocket. Also readable like a dictionary (``pocket["center"]``) for compatibility with version 0.1."""
    rank: int
    score: float                  # sum of the 24 highest surface-point scores in the pocket
    probability: float            # isotonic calibration of ``score`` on held-out proteins (indicative only)
    center: list[float]           # x, y, z in angstrom (score-weighted mean of the highest-scoring points)
    n_points: int                 # surface points assigned to the pocket
    residues: list[str]           # ``chain:number`` of residues with an atom within ``residue_radius`` of the centre
    point_indices: list[int] = field(default_factory=list, repr=False)

    def __getitem__(self, key: str): return getattr(self, key)
    def keys(self): return [k for k in self.__dataclass_fields__ if k != "point_indices"]
    def to_dict(self, points: bool = False) -> dict:
        d = {k: getattr(self, k) for k in self.keys()}
        if points: d["point_indices"] = self.point_indices
        return d
    def distance_to(self, xyz) -> float: return float(np.linalg.norm(np.asarray(self.center) - np.asarray(xyz, float)))


@dataclass
class PredictionResult:
    """Ranked pockets plus the per-point and per-residue scores they were derived from."""
    pockets: list[Pocket]
    points: np.ndarray            # (P, 3) solvent-accessible surface points
    point_scores: np.ndarray      # (P,) ensemble ligandability score per surface point
    structure: Structure
    source: str = ""
    _residue_scores: Optional[dict] = field(default=None, repr=False)

    def __len__(self) -> int: return len(self.pockets)
    def __iter__(self) -> Iterator[Pocket]: return iter(self.pockets)
    def __getitem__(self, i): return self.pockets[i]
    @property
    def best(self) -> Optional[Pocket]: return self.pockets[0] if self.pockets else None

    def __str__(self) -> str:
        head = f"{self.source or 'structure'}: {len(self.structure)} atoms, {len(self.points)} surface points, {len(self.pockets)} pockets"
        rows = [f"  {p.rank:>2}  score {p.score:6.2f}  p~{p.probability:.2f}  centre ({p.center[0]:8.2f} {p.center[1]:8.2f} {p.center[2]:8.2f})  {len(p.residues):>3} residues" for p in self.pockets]
        return "\n".join([head] + rows)

    # ------------------------------------------------------------------ residue scores
    @property
    def residue_scores(self) -> dict[str, float]:
        """Highest score of any surface point within 4.5 A of each residue (``chain:number`` -> score)."""
        if self._residue_scores is None:
            from scipy.spatial import cKDTree
            keys = self.structure.residue_keys; tree = cKDTree(self.points); out: dict[str, float] = {}
            for k, near in zip(keys, tree.query_ball_point(self.structure.xyz, 4.5)):
                if near: out[k] = max(out.get(k, 0.0), float(self.point_scores[near].max()))
                else: out.setdefault(k, 0.0)
            self._residue_scores = out
        return self._residue_scores

    # ------------------------------------------------------------------ exporters
    def to_dict(self, points: bool = False) -> dict:
        return dict(source=self.source, n_atoms=len(self.structure), n_surface_points=len(self.points), pockets=[p.to_dict(points) for p in self.pockets])
    def to_json(self, path: Optional[str] = None, **kw) -> str:
        s = json.dumps(self.to_dict(), indent=kw.pop("indent", 1), **kw)
        if path: Path(path).write_text(s)
        return s
    def to_csv(self, path: Optional[str] = None) -> str:
        buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n"); w.writerow(["rank", "score", "probability", "center_x", "center_y", "center_z", "n_points", "residues"])
        for p in self.pockets: w.writerow([p.rank, f"{p.score:.3f}", f"{p.probability:.3f}", *p.center, p.n_points, " ".join(p.residues)])
        if path: Path(path).write_text(buf.getvalue())
        return buf.getvalue()
    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame([{**p.to_dict(), "n_residues": len(p.residues)} for p in self.pockets])

    def write_pdb(self, path: str) -> str:
        """PDB file of the protein in which the B-factor column holds 100 x residue score and the occupancy the rank of the best
        pocket containing the residue (0 if none); pocket centres are added as ``PKT`` pseudo-atoms (B-factor = pocket score)."""
        st = self.structure; rs = self.residue_scores; keys = st.residue_keys; rank_of: dict[str, int] = {}
        for p in reversed(self.pockets):
            for r in p.residues: rank_of[r] = p.rank
        lines = []
        for i in range(len(st)):
            nm = st.name[i]; nm = f" {nm:<3}" if len(nm) < 4 else nm
            lines.append(f"ATOM  {i + 1:>5} {nm} {st.res[i]:>3} {st.chain[i][:1]:1}{st.rid[i]:>4}{st.icode[i][:1] or ' '}   {st.xyz[i, 0]:8.3f}{st.xyz[i, 1]:8.3f}{st.xyz[i, 2]:8.3f}{float(rank_of.get(keys[i], 0)):6.2f}{100 * rs.get(keys[i], 0.0):6.2f}          {st.element[i][:2]:>2}")
        n = len(st)
        for p in self.pockets:
            n += 1; lines.append(f"HETATM{n:>5}  PC  PKT Z{p.rank:>4}    {p.center[0]:8.3f}{p.center[1]:8.3f}{p.center[2]:8.3f}{float(p.rank):6.2f}{p.score:6.2f}          {'C':>2}")
        text = "\n".join(lines + ["END", ""]); Path(path).write_text(text); return text

    def write_pymol(self, path: str, pdb_name: str = "structure.pdb") -> str:
        """PyMOL script that loads ``pdb_name`` and colours each pocket's residues."""
        cols = ["orange", "marine", "forest", "purple", "yellow", "salmon", "cyan", "magenta", "limon", "teal"]
        L = [f"load {pdb_name}, prot", "hide everything", "show cartoon, prot", "color grey80, prot", "set cartoon_transparency, 0.0"]
        for p in self.pockets:
            sel = " or ".join(f"(chain {r.split(':')[0]} and resi {r.split(':')[1]})" for r in p.residues) or "none"
            L += [f"select pocket{p.rank}, prot and ({sel})", f"color {cols[(p.rank - 1) % len(cols)]}, pocket{p.rank}", f"show sticks, pocket{p.rank}",
                  f"pseudoatom ctr{p.rank}, pos=[{p.center[0]:.3f},{p.center[1]:.3f},{p.center[2]:.3f}], label={p.rank}", f"show spheres, ctr{p.rank}", f"color {cols[(p.rank - 1) % len(cols)]}, ctr{p.rank}"]
        L += ["set sphere_scale, 0.8, ctr*", "zoom pocket1" if self.pockets else "zoom"]; text = "\n".join(L) + "\n"; Path(path).write_text(text); return text

    def write_chimerax(self, path: str, pdb_name: str = "structure.pdb") -> str:
        """ChimeraX command file that loads ``pdb_name`` and colours the residues and centre of each pocket."""
        cols = ["orange", "dodgerblue", "forestgreen", "purple", "gold", "salmon", "cyan", "magenta", "lime", "teal"]
        L = [f"open {pdb_name}", "hide atoms", "cartoon", "color lightgray target c"]
        for p in self.pockets:
            for r in p.residues[:400]:
                c, n = r.split(":"); L.append(f"color /{c}:{n} {cols[(p.rank - 1) % len(cols)]} target c")
            L.append(f"marker #{10 + p.rank} position {p.center[0]:.3f},{p.center[1]:.3f},{p.center[2]:.3f} color {cols[(p.rank - 1) % len(cols)]} radius 1.2")
        text = "\n".join(L) + "\n"; Path(path).write_text(text); return text

    def save(self, out_dir: str, stem: Optional[str] = None, formats=("csv", "json", "pdb", "pml", "cxc")) -> list[str]:
        """Write the requested formats (``csv``, ``json``, ``pdb``, ``pml``, ``cxc``) into ``out_dir``; returns the file names."""
        d = Path(out_dir); d.mkdir(parents=True, exist_ok=True); stem = stem or Path(self.source).stem or "structure"; out = []
        if "pdb" in formats or "pml" in formats or "cxc" in formats: self.write_pdb(str(d / f"{stem}_pocketnet.pdb")); out.append(f"{stem}_pocketnet.pdb")
        if "csv" in formats: self.to_csv(str(d / f"{stem}_pockets.csv")); out.append(f"{stem}_pockets.csv")
        if "json" in formats: self.to_json(str(d / f"{stem}_pockets.json")); out.append(f"{stem}_pockets.json")
        if "pml" in formats: self.write_pymol(str(d / f"{stem}_pockets.pml"), f"{stem}_pocketnet.pdb"); out.append(f"{stem}_pockets.pml")
        if "cxc" in formats: self.write_chimerax(str(d / f"{stem}_pockets.cxc"), f"{stem}_pocketnet.pdb"); out.append(f"{stem}_pockets.cxc")
        return out

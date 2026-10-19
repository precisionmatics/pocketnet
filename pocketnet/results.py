"""Result objects returned by :class:`pocketnet.PocketPredictor` and their exporters."""
from __future__ import annotations
import csv, io, json, math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, List, Optional, Sequence
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
    def distance_to(self, xyz) -> float:
        """Euclidean distance from this pocket's centre to a point (angstrom)."""
        return float(np.linalg.norm(np.asarray(self.center) - np.asarray(xyz, float)))

    @property
    def is_druggable(self) -> bool:
        """True when the calibrated probability is at least 0.5 (indicative threshold)."""
        return self.probability >= 0.5

    def volume(self, points: np.ndarray) -> float:
        """Estimate of pocket volume (Å³) from the convex hull of its member surface points.

        Parameters
        ----------
        points : (P, 3) array of all surface points in the parent :class:`PredictionResult`.

        Returns
        -------
        float
            Convex-hull volume in Å³, or 0.0 if fewer than 4 member points are available.
        """
        if not self.point_indices:
            return 0.0
        pts = np.asarray(points)[self.point_indices]
        if len(pts) < 4:
            return 0.0
        try:
            from scipy.spatial import ConvexHull
            return float(ConvexHull(pts).volume)
        except Exception:
            return 0.0


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

    def __repr__(self) -> str:
        return f"PredictionResult(source={self.source!r}, pockets={len(self.pockets)}, surface_points={len(self.points)})"

    # ------------------------------------------------------------------ Jupyter display
    def _repr_html_(self) -> str:
        """Rich HTML table for Jupyter notebooks."""
        COLORS = ["#E8743B", "#4F81BD", "#6AA84F", "#8E44AD", "#D4AC0D",
                  "#C0392B", "#17A589", "#2E86C1", "#BA4A00", "#1A5276"]
        rows_html = []
        for p in self.pockets:
            bar_w = int(p.probability * 100)
            col = COLORS[(p.rank - 1) % len(COLORS)]
            swatch = f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{col};margin-right:5px;vertical-align:middle;"></span>'
            bar = (f'<div style="background:#eee;border-radius:3px;width:80px;display:inline-block;vertical-align:middle;">'
                   f'<div style="background:{col};width:{bar_w}px;height:8px;border-radius:3px;"></div></div> '
                   f'<span style="font-size:0.85em">{p.probability:.2f}</span>')
            druggable = '<span style="color:#27AE60;font-weight:600">✓</span>' if p.is_druggable else '<span style="color:#999">–</span>'
            vol = p.volume(self.points)
            vol_str = f"{vol:.0f}" if vol > 0 else "—"
            res_str = ", ".join(p.residues[:6]) + ("…" if len(p.residues) > 6 else "")
            rows_html.append(
                f"<tr>"
                f"<td>{swatch}<b>{p.rank}</b></td>"
                f"<td>{p.score:.2f}</td>"
                f"<td>{bar}</td>"
                f"<td>{druggable}</td>"
                f"<td>{p.n_points}</td>"
                f"<td>{vol_str}</td>"
                f"<td style='font-size:0.82em;color:#555'>{res_str}</td>"
                f"</tr>"
            )
        src = self.source or "structure"
        header = (
            f"<div style='font-family:sans-serif;margin:4px 0 2px'>"
            f"<b>PocketNet</b> &nbsp;·&nbsp; {src} &nbsp;·&nbsp; "
            f"{len(self.structure):,} atoms &nbsp;·&nbsp; {len(self.points):,} surface points &nbsp;·&nbsp; "
            f"<b>{len(self.pockets)}</b> pockets"
            f"</div>"
        )
        table = (
            f"<table style='border-collapse:collapse;font-family:monospace;font-size:0.88em'>"
            f"<thead><tr style='border-bottom:1px solid #ccc'>"
            f"<th style='padding:3px 8px'>Rank</th><th style='padding:3px 8px'>Score</th>"
            f"<th style='padding:3px 8px'>Probability</th><th style='padding:3px 8px'>Druggable</th>"
            f"<th style='padding:3px 8px'>Points</th><th style='padding:3px 8px'>Volume (Å³)</th>"
            f"<th style='padding:3px 8px'>Lining residues</th>"
            f"</tr></thead>"
            f"<tbody>{''.join(rows_html)}</tbody>"
            f"</table>"
        )
        return header + table

    # ------------------------------------------------------------------ filtering
    def filter(
        self,
        min_probability: Optional[float] = None,
        min_score: Optional[float] = None,
        max_rank: Optional[int] = None,
        druggable_only: bool = False,
    ) -> "PredictionResult":
        """Return a new :class:`PredictionResult` containing only the pockets that pass every criterion.

        Pockets are re-ranked 1…N in the filtered result. The surface points and residue
        scores from the original result are kept unchanged.

        Parameters
        ----------
        min_probability : keep pockets with ``probability >= min_probability``.
        min_score : keep pockets with ``score >= min_score``.
        max_rank : keep pockets whose original rank is at most ``max_rank``.
        druggable_only : keep only pockets where :attr:`Pocket.is_druggable` is True.
        """
        kept = []
        for p in self.pockets:
            if min_probability is not None and p.probability < min_probability:
                continue
            if min_score is not None and p.score < min_score:
                continue
            if max_rank is not None and p.rank > max_rank:
                continue
            if druggable_only and not p.is_druggable:
                continue
            kept.append(p)
        # re-rank
        reranked = [
            Pocket(rank=i + 1, score=p.score, probability=p.probability,
                   center=p.center, n_points=p.n_points, residues=p.residues,
                   point_indices=p.point_indices)
            for i, p in enumerate(kept)
        ]
        return PredictionResult(reranked, self.points, self.point_scores, self.structure, self.source, self._residue_scores)

    # ------------------------------------------------------------------ volumes
    @property
    def volumes(self) -> list[float]:
        """Convex-hull volume (Å³) for each pocket in rank order."""
        return [p.volume(self.points) for p in self.pockets]

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
        buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
        w.writerow(["rank", "score", "probability", "is_druggable", "center_x", "center_y", "center_z", "n_points", "volume_A3", "n_residues", "residues"])
        vols = self.volumes
        for p, vol in zip(self.pockets, vols):
            w.writerow([p.rank, f"{p.score:.3f}", f"{p.probability:.3f}", int(p.is_druggable),
                        *[f"{v:.3f}" for v in p.center], p.n_points, f"{vol:.1f}", len(p.residues), " ".join(p.residues)])
        if path: Path(path).write_text(buf.getvalue())
        return buf.getvalue()
    def to_dataframe(self):
        """Return a :class:`pandas.DataFrame` with one row per pocket (includes volume column)."""
        import pandas as pd
        vols = self.volumes
        rows = [{**p.to_dict(), "is_druggable": p.is_druggable, "n_residues": len(p.residues), "volume_A3": vol}
                for p, vol in zip(self.pockets, vols)]
        return pd.DataFrame(rows)

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

    def write_sdf(self, path: str) -> str:
        """Write pocket centres as pseudo-atoms in SDF format.

        Each record is a single carbon atom at the pocket centre.  Pocket
        properties (rank, score, probability, is_druggable, n_points,
        volume_A3, n_residues) are stored as SDF data fields so the file
        can be read by RDKit, OpenBabel or any cheminformatics toolkit.
        """
        vols = self.volumes
        blocks = []
        for p, vol in zip(self.pockets, vols):
            x, y, z = p.center
            block = (
                f"pocket_{p.rank}\n"
                f"  PocketNet 0.2\n"
                f"\n"
                f"  1  0  0  0  0  0  0  0  0  0999 V2000\n"
                f"   {x:10.4f}{y:10.4f}{z:10.4f} C   0  0  0  0  0  0  0  0  0  0  0  0\n"
                f"M  END\n"
                f">  <RANK>\n{p.rank}\n\n"
                f">  <SCORE>\n{p.score:.4f}\n\n"
                f">  <PROBABILITY>\n{p.probability:.4f}\n\n"
                f">  <IS_DRUGGABLE>\n{int(p.is_druggable)}\n\n"
                f">  <N_POINTS>\n{p.n_points}\n\n"
                f">  <VOLUME_A3>\n{vol:.2f}\n\n"
                f">  <N_RESIDUES>\n{len(p.residues)}\n\n"
                f">  <RESIDUES>\n{' '.join(p.residues)}\n\n"
                f"$$$$"
            )
            blocks.append(block)
        text = "\n".join(blocks) + "\n"
        Path(path).write_text(text)
        return text

    # ------------------------------------------------------------------ interactive viewers
    def show(self, width: int = 880, height: int = 540):
        """Render an interactive 3-D widget inside a Jupyter notebook (requires ``pip install py3Dmol``)."""
        from .viewer import show as _show
        return _show(self, width=width, height=height)

    def view(self, open_browser: bool = True) -> str:
        """Open a full interactive NGL.js viewer in the default web browser.

        Writes a self-contained HTML file and opens it.  Returns the file path.
        """
        from .viewer import view as _view
        return _view(self, open_browser=open_browser)

    def save(self, out_dir: str, stem: Optional[str] = None, formats=("csv", "json", "pdb", "pml", "cxc", "sdf")) -> list[str]:
        """Write the requested formats (``csv``, ``json``, ``pdb``, ``pml``, ``cxc``, ``sdf``) into ``out_dir``; returns the file names."""
        d = Path(out_dir); d.mkdir(parents=True, exist_ok=True); stem = stem or Path(self.source).stem or "structure"; out = []
        if "pdb" in formats or "pml" in formats or "cxc" in formats: self.write_pdb(str(d / f"{stem}_pocketnet.pdb")); out.append(f"{stem}_pocketnet.pdb")
        if "csv" in formats: self.to_csv(str(d / f"{stem}_pockets.csv")); out.append(f"{stem}_pockets.csv")
        if "json" in formats: self.to_json(str(d / f"{stem}_pockets.json")); out.append(f"{stem}_pockets.json")
        if "pml" in formats: self.write_pymol(str(d / f"{stem}_pockets.pml"), f"{stem}_pocketnet.pdb"); out.append(f"{stem}_pockets.pml")
        if "cxc" in formats: self.write_chimerax(str(d / f"{stem}_pockets.cxc"), f"{stem}_pocketnet.pdb"); out.append(f"{stem}_pockets.cxc")
        if "sdf" in formats: self.write_sdf(str(d / f"{stem}_pockets.sdf")); out.append(f"{stem}_pockets.sdf")
        return out

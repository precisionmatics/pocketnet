"""Structure input: PDB and mmCIF (optionally gzipped) files, PDB-format text, and fetching from RCSB / AlphaFold DB."""
from __future__ import annotations
import gzip, io, os, re, shlex, urllib.error, urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union
import numpy as np

PathLike = Union[str, os.PathLike]
CACHE_DIR = Path(os.environ.get("POCKETNET_CACHE", Path.home() / ".cache" / "pocketnet"))
_PDB_ID = re.compile(r"^[0-9][A-Za-z0-9]{3}$")
_UNIPROT = re.compile(r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$")


@dataclass
class Structure:
    """Protein heavy atoms of the first model (ATOM records and selenomethionine)."""
    name: np.ndarray      # atom names, e.g. "CA"
    res: np.ndarray       # residue names, e.g. "ALA"
    rid: np.ndarray       # residue numbers (int)
    icode: np.ndarray     # insertion codes ("" if none)
    chain: np.ndarray     # chain identifiers
    element: np.ndarray   # element symbols
    xyz: np.ndarray       # (N, 3) float32 coordinates in angstrom
    bfactor: np.ndarray   # (N,) float32 (pLDDT for AlphaFold models)
    source: str = ""

    def __len__(self) -> int: return len(self.xyz)

    @property
    def chains(self) -> list[str]: return sorted(set(self.chain.tolist()))

    @property
    def residue_keys(self) -> np.ndarray:
        """One identifier per atom, ``chain:number[insertion code]`` (for example ``A:790`` or ``A:52B``)."""
        return np.array([f"{c}:{r}{i}" for c, r, i in zip(self.chain, self.rid, self.icode)])

    def select_chains(self, chains: Optional[Iterable[str]]) -> "Structure":
        if not chains: return self
        keep = np.isin(self.chain, list(chains))
        if not keep.any(): raise ValueError(f"none of the chains {list(chains)} are present (found: {self.chains})")
        return Structure(**{k: (getattr(self, k)[keep] if isinstance(getattr(self, k), np.ndarray) else getattr(self, k)) for k in self.__dataclass_fields__})

    def mean_bfactor(self) -> float: return float(self.bfactor.mean()) if len(self) else float("nan")


def _empty(rows, source) -> Structure:
    if not rows: raise ValueError("no protein atoms were read (ATOM records of the first model are required)")
    n, r, i, ic, c, e, x, b = zip(*rows)
    return Structure(np.array(n), np.array(r), np.array(i, np.int64), np.array(ic), np.array(c), np.array(e), np.array(x, np.float32), np.array(b, np.float32), source)


def _int(s: str) -> int:
    s = s.strip()
    try: return int(s)
    except ValueError: return 0


def parse_pdb(text: str, source: str = "") -> Structure:
    rows = []
    for ln in text.splitlines():
        if ln.startswith("ENDMDL"): break
        rec = ln[:6]
        if rec not in ("ATOM  ", "HETATM") or len(ln) < 54: continue
        if ln[16] not in " A1": continue
        res = ln[17:20].strip()
        if rec == "HETATM" and res != "MSE": continue
        name = ln[12:16].strip(); el = (ln[76:78].strip() if len(ln) >= 78 else "") or name[:1]
        if el.upper() in ("H", "D"): continue
        try: bf = float(ln[60:66])
        except ValueError: bf = 0.0
        rows.append((name, res, _int(ln[22:26]), ln[26].strip(), ln[21].strip() or "A", el.upper(), (float(ln[30:38]), float(ln[38:46]), float(ln[46:54])), bf))
    return _empty(rows, source)


def _cif_tokens(line: str) -> list[str]:
    return shlex.split(line, posix=True) if ("'" in line or '"' in line) else line.split()


def parse_mmcif(text: str, source: str = "") -> Structure:
    """Minimal mmCIF reader for the ``_atom_site`` loop (author chain and residue numbering)."""
    lines = text.splitlines(); cols: list[str] = []; i = 0; start = None
    while i < len(lines):
        if lines[i].strip() == "loop_":
            j = i + 1; names = []
            while j < len(lines) and lines[j].startswith("_"): names.append(lines[j].strip().split()[0]); j += 1
            if names and names[0].startswith("_atom_site."): cols, start = names, j; break
            i = j
        else: i += 1
    if start is None: raise ValueError("no _atom_site loop found in the mmCIF file")
    ix = {c.split(".", 1)[1]: k for k, c in enumerate(cols)}
    need = ["group_PDB", "label_atom_id", "label_comp_id", "Cartn_x", "Cartn_y", "Cartn_z"]
    if any(k not in ix for k in need): raise ValueError("the _atom_site loop lacks required columns")
    g = lambda t, k, d="": t[ix[k]] if k in ix and ix[k] < len(t) else d
    rows = []; model = None
    for ln in lines[start:]:
        if not ln.strip() or ln.startswith("#") or ln.startswith("loop_") or ln.startswith("_") or ln.startswith("data_"): break
        t = _cif_tokens(ln)
        if len(t) < len(cols): continue
        mn = g(t, "pdbx_PDB_model_num", "1")
        if model is None: model = mn
        if mn != model: break
        alt = g(t, "label_alt_id", ".")
        if alt not in (".", "?", "A", "1"): continue
        res = g(t, "label_comp_id")
        if g(t, "group_PDB") != "ATOM" and res != "MSE": continue
        el = g(t, "type_symbol").upper() or g(t, "label_atom_id")[:1]
        if el in ("H", "D"): continue
        ic = g(t, "pdbx_PDB_ins_code", "?"); ic = "" if ic in ("?", ".") else ic
        seq = g(t, "auth_seq_id") or g(t, "label_seq_id")
        chain = g(t, "auth_asym_id") or g(t, "label_asym_id") or "A"
        try: bf = float(g(t, "B_iso_or_equiv", "0"))
        except ValueError: bf = 0.0
        rows.append((g(t, "label_atom_id").strip('"'), res, _int(seq), ic, chain, el, (float(g(t, "Cartn_x")), float(g(t, "Cartn_y")), float(g(t, "Cartn_z"))), bf))
    return _empty(rows, source)


def _read_text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f: return f.read()
    return path.read_text()


def read_structure(src: PathLike, chains: Optional[Sequence[str]] = None) -> Structure:
    """Read a structure from a file path (``.pdb``, ``.ent``, ``.cif``, ``.mmcif``, optionally ``.gz``) or from PDB / mmCIF text."""
    s = str(src)
    if "\n" in s or s.lstrip().startswith(("ATOM", "HEADER", "REMARK", "CRYST1", "data_", "loop_", "HETATM")):
        text, name = s, "<text>"
    else:
        p = Path(s).expanduser()
        if not p.exists(): raise FileNotFoundError(f"structure file not found: {p}")
        text, name = _read_text(p), p.name
    st = parse_mmcif(text, name) if re.search(r"^data_|^_atom_site\.", text[:20000], re.M) else parse_pdb(text, name)
    return st.select_chains(chains)


def looks_like_pdb_id(s: str) -> bool: return bool(_PDB_ID.match(s)) and not Path(s).exists()
def looks_like_uniprot(s: str) -> bool: return bool(_UNIPROT.match(s.upper())) and not Path(s).exists()


def _download(url: str, dest: Path, timeout: float = 60.0) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "pocketnet (+https://github.com/precisionmatics/pocketnet)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: data = r.read()
    except urllib.error.HTTPError as e: raise RuntimeError(f"download failed ({e.code}) for {url}") from e
    except urllib.error.URLError as e: raise RuntimeError(f"could not reach {url}: {e.reason}") from e
    tmp = dest.with_suffix(dest.suffix + ".part"); tmp.write_bytes(data); tmp.replace(dest); return dest


def fetch_pdb(pdb_id: str, dest_dir: Optional[PathLike] = None, fmt: str = "pdb") -> Path:
    """Download a structure from the RCSB Protein Data Bank (cached in ``~/.cache/pocketnet``). ``fmt`` is ``pdb`` or ``cif``."""
    if not _PDB_ID.match(pdb_id): raise ValueError(f"not a PDB identifier: {pdb_id!r}")
    ext = {"pdb": "pdb", "cif": "cif"}[fmt]; d = Path(dest_dir) if dest_dir else CACHE_DIR; f = d / f"{pdb_id.upper()}.{ext}"
    return f if f.exists() else _download(f"https://files.rcsb.org/download/{pdb_id.upper()}.{ext}", f)


def fetch_alphafold(uniprot: str, dest_dir: Optional[PathLike] = None) -> Path:
    """Download the AlphaFold DB model of a UniProt accession (cached in ``~/.cache/pocketnet``)."""
    acc = uniprot.upper(); d = Path(dest_dir) if dest_dir else CACHE_DIR; f = d / f"AF-{acc}-F1.pdb"
    return f if f.exists() else _download(f"https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v4.pdb", f)

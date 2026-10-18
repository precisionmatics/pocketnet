import json
import numpy as np
from pocketnet.structure import read_structure


def test_exports(result, tmp_path):
    files = result.save(tmp_path, "t")
    assert set(files) == {"t_pocketnet.pdb", "t_pockets.csv", "t_pockets.json", "t_pockets.pml", "t_pockets.cxc"}
    assert all((tmp_path / f).stat().st_size > 0 for f in files)
    d = json.loads((tmp_path / "t_pockets.json").read_text()); assert len(d["pockets"]) == len(result)
    assert (tmp_path / "t_pockets.csv").read_text().splitlines()[0].startswith("rank,score,probability")


def test_bfactor_pdb_roundtrip(result, tmp_path):
    f = tmp_path / "o.pdb"; result.write_pdb(str(f))
    st = read_structure(f)
    assert len(st) == len(result.structure) and np.allclose(st.xyz, result.structure.xyz, atol=1e-3)
    assert st.bfactor.max() <= 100 and st.bfactor.max() > 50      # 100 x score of the best residue


def test_residue_scores_cover_pocket_residues(result):
    rs = result.residue_scores
    assert all(0.0 <= v <= 1.0 for v in rs.values()) and set(result.best.residues) <= set(rs)
    assert np.mean([rs[r] for r in result.best.residues]) > np.mean(list(rs.values()))


def test_str_and_scripts(result, tmp_path):
    assert "pockets" in str(result) and "select pocket1" in result.write_pymol(str(tmp_path / "a.pml")) and "marker" in result.write_chimerax(str(tmp_path / "a.cxc"))

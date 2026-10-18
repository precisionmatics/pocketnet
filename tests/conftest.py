from pathlib import Path
import pytest
from pocketnet import PocketPredictor

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "3POZ.pdb"


@pytest.fixture(scope="session")
def predictor(): return PocketPredictor(device="cpu")


@pytest.fixture(scope="session")
def example(): return EXAMPLE


@pytest.fixture(scope="session")
def result(predictor, example): return predictor.predict(example)


def pdb_to_mmcif(pdb_path: Path) -> str:
    """Tiny converter used to test the mmCIF reader on the example structure."""
    head = ["data_TEST", "loop_", "_atom_site.group_PDB", "_atom_site.id", "_atom_site.type_symbol", "_atom_site.label_atom_id", "_atom_site.label_alt_id",
            "_atom_site.label_comp_id", "_atom_site.label_asym_id", "_atom_site.label_seq_id", "_atom_site.pdbx_PDB_ins_code", "_atom_site.Cartn_x", "_atom_site.Cartn_y",
            "_atom_site.Cartn_z", "_atom_site.B_iso_or_equiv", "_atom_site.auth_seq_id", "_atom_site.auth_asym_id", "_atom_site.pdbx_PDB_model_num"]
    rows = []
    for i, ln in enumerate(l for l in pdb_path.read_text().splitlines() if l[:6] in ("ATOM  ", "HETATM")):
        rows.append(f"{ln[:6].strip()} {i + 1} {ln[76:78].strip() or ln[12:16].strip()[0]} {ln[12:16].strip()} . {ln[17:20].strip()} {ln[21]} {ln[22:26].strip()} ? {ln[30:38].strip()} {ln[38:46].strip()} {ln[46:54].strip()} {ln[60:66].strip()} {ln[22:26].strip()} {ln[21]} 1")
    return "\n".join(head + rows + ["#", ""])

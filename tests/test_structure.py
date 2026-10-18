import gzip
import numpy as np
import pytest
from pocketnet.structure import read_structure, parse_pdb, looks_like_pdb_id, looks_like_uniprot
from conftest import pdb_to_mmcif


def test_pdb_basic(example):
    st = read_structure(example)
    assert len(st) > 2000 and st.chains and set(st.element) <= {"C", "N", "O", "S", "SE"}


def test_mmcif_matches_pdb(example, tmp_path):
    cif = tmp_path / "x.cif"; cif.write_text(pdb_to_mmcif(example))
    a, b = read_structure(example), read_structure(cif)
    assert len(a) == len(b) and np.allclose(a.xyz, b.xyz) and (a.res == b.res).all()


def test_gzip(example, tmp_path):
    f = tmp_path / "x.pdb.gz"
    with gzip.open(f, "wt") as h: h.write(example.read_text())
    assert len(read_structure(f)) == len(read_structure(example))


def test_text_input(example):
    assert len(read_structure(example.read_text())) == len(read_structure(example))


def test_errors(tmp_path):
    with pytest.raises(FileNotFoundError): read_structure(tmp_path / "nope.pdb")
    with pytest.raises(ValueError): parse_pdb("REMARK nothing\nEND\n")
    with pytest.raises(ValueError): read_structure("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n", chains=["Z"])


def test_identifier_detection():
    assert looks_like_pdb_id("3POZ") and not looks_like_pdb_id("protein.pdb")
    assert looks_like_uniprot("P00533") and not looks_like_uniprot("3POZ")

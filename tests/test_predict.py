import json
import numpy as np
from pocketnet import PocketPredictor, Pocket


def test_ranking_and_fields(result):
    assert 1 <= len(result) <= 10
    assert [p.rank for p in result] == list(range(1, len(result) + 1))
    assert all(result[i].score >= result[i + 1].score for i in range(len(result) - 1))
    assert all(0.0 <= p.probability <= 1.0 for p in result)
    assert all(len(p.center) == 3 and p.n_points > 0 and p.residues for p in result)


def test_first_pocket_is_on_the_atp_site(result, example):
    """EGFR kinase domain with TAK-285 (03P): the first pocket centre lies within 4 A of a ligand atom."""
    lig = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in example.read_text().splitlines() if l.startswith("HETATM") and l[17:20].strip() == "03P"])
    assert len(lig) and np.linalg.norm(lig - np.array(result.best.center), axis=1).min() <= 4.0


def test_dict_style_access_is_kept(result):
    p = result.best
    assert isinstance(p, Pocket) and p["center"] == p.center and p["score"] == p.score
    json.dumps([q.to_dict() for q in result])


def test_chain_selection(predictor, example):
    r = predictor.predict(example, chains=["A"])
    assert set(r.structure.chain.tolist()) == {"A"}


def test_max_pockets_and_extraction_override(example):
    p = PocketPredictor(device="cpu", link=8.0)
    assert len(p.predict(example, max_pockets=2)) <= 2 and p.params["link"] == 8.0


def test_unknown_setting_is_rejected():
    import pytest
    with pytest.raises(TypeError): PocketPredictor(device="cpu", nonsense=1)


def test_predict_many_continues_after_failure(predictor, example):
    out = list(predictor.predict_many([str(example), "does_not_exist.pdb"]))
    assert out[0][1] is not None and out[1][1] is None and isinstance(out[1][2], FileNotFoundError)

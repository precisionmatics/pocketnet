import json
from pathlib import Path
import numpy as np
from pocketnet import PocketPredictor

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "3POZ.pdb"


def test_predict_example_on_cpu():
    res = PocketPredictor(device="cpu").predict(EXAMPLE)
    assert 1 <= len(res) <= 10
    assert [r["rank"] for r in res] == list(range(1, len(res) + 1))
    assert all(res[i]["score"] >= res[i + 1]["score"] for i in range(len(res) - 1))
    assert all(0.0 <= r["probability"] <= 1.0 for r in res)
    # the ATP-site pocket of the EGFR kinase domain (ligand TAK-285) is within 4 A of the first pocket centre
    lig = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in EXAMPLE.read_text().splitlines() if l.startswith("HETATM") and l[17:20].strip() == "03P"])
    if len(lig):
        d = np.linalg.norm(lig - np.array(res[0]["center"]), axis=1).min()
        assert d <= 4.0
    json.dumps(res)

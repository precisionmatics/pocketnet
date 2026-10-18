"""PocketNet: ligand-binding pocket prediction from protein structures.

>>> from pocketnet import PocketPredictor
>>> result = PocketPredictor().predict("3POZ.pdb")    # also: mmCIF, PDB identifiers, UniProt accessions (AlphaFold)
>>> print(result)
>>> result.save("out/")                                # CSV, JSON, B-factor PDB, PyMOL and ChimeraX scripts
"""
__version__ = "0.2.0"
from .api import PocketPredictor, predict, resolve                       # noqa: E402,F401
from .results import Pocket, PredictionResult                            # noqa: E402,F401
from .structure import Structure, read_structure, fetch_pdb, fetch_alphafold   # noqa: E402,F401

__all__ = ["PocketPredictor", "predict", "resolve", "Pocket", "PredictionResult", "Structure", "read_structure", "fetch_pdb", "fetch_alphafold", "__version__"]

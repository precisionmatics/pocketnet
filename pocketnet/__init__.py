"""PocketNet: ligand-binding pocket prediction from protein structures.

>>> from pocketnet import PocketPredictor
>>> pockets = PocketPredictor().predict("examples/3POZ.pdb")
>>> pockets[0]["center"], pockets[0]["score"]
"""
__version__ = "0.1.0"
from .api import PocketPredictor, predict  # noqa: E402,F401

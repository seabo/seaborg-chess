"""ChessFormer: a faithful PyTorch re-implementation of the architecture from
Monroe & Chalmers, "Mastering Chess with a Transformer Model" (arXiv:2409.12272),
adapted for supervised training on the Lichess Stockfish-evaluation dataset.
"""

from .config import ChessFormerConfig, CF_6M, CF_TINY
from .model import ChessFormer
from . import encoding

__all__ = ["ChessFormerConfig", "CF_6M", "CF_TINY", "ChessFormer", "encoding"]

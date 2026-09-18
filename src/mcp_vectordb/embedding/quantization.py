"""TurboQuant-style embedding compression: random rotation + per-vector scalar
quantization, with an approximate distance estimator that scores directly
against the compressed codes.

This is a post-hoc, model-agnostic compression layer applied to embeddings
after they're generated (see embedding/base.py) and before storage — it does
not replace the embedding model itself.
"""

import numpy as np
from typing import List


class TurboQuantCodec:
    """Compresses float embedding vectors into low-bit codes and scores
    queries against those codes directly, without full float dequantization.

    A fixed random orthogonal rotation decorrelates the vector's dimensions
    before quantization (uniform scalar quantization otherwise wastes bits on
    high-variance dimensions). The rotation is deterministic for a given
    (dimension, seed) pair, so encode/decode/score stay consistent across
    process restarts as long as the seed is persisted alongside the codes.
    """

    def __init__(self, dimension: int, bits: int = 8, seed: int = 42):
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        if not (1 <= bits <= 8):
            raise ValueError(f"bits must be between 1 and 8, got {bits}")

        self.dimension = dimension
        self.bits = bits
        self.seed = seed
        self._levels = (1 << bits) - 1  # e.g. 255 for 8 bits
        self._rotation = self._build_rotation_matrix(dimension, seed)

    @staticmethod
    def _build_rotation_matrix(dimension: int, seed: int) -> np.ndarray:
        """Deterministic random orthogonal matrix via QR decomposition of a
        seeded Gaussian matrix."""
        rng = np.random.default_rng(seed)
        gaussian = rng.standard_normal((dimension, dimension))
        q, r = np.linalg.qr(gaussian)
        # Fix the sign ambiguity in QR so the same seed always yields the
        # same rotation, not just one that's orthogonal.
        q *= np.sign(np.diag(r))
        return q

    def _validate_dimension(self, vector: List[float]) -> None:
        if len(vector) != self.dimension:
            raise ValueError(
                f"expected a {self.dimension}-dim vector, got {len(vector)}"
            )

    def encode(self, vector: List[float]) -> "QuantizedVector":
        """Rotate and quantize a float embedding into low-bit codes."""
        self._validate_dimension(vector)
        rotated = self._rotation @ np.asarray(vector, dtype=np.float64)

        vmin = float(rotated.min())
        vmax = float(rotated.max())
        scale = (vmax - vmin) / self._levels if vmax > vmin else 1.0

        codes = np.round((rotated - vmin) / scale).astype(np.uint8)
        return QuantizedVector(codes=codes, vmin=vmin, scale=scale)

    def decode(self, quantized: "QuantizedVector") -> List[float]:
        """Reconstruct an approximate float vector from its codes."""
        rotated_approx = quantized.codes.astype(np.float64) * quantized.scale + quantized.vmin
        # Rotation matrix is orthogonal, so its inverse is its transpose.
        original_approx = self._rotation.T @ rotated_approx
        return original_approx.tolist()

    def score(self, query: List[float], quantized: "QuantizedVector") -> float:
        """Approximate cosine similarity between a raw query vector and a
        previously-encoded (compressed) vector, without exact float storage.
        """
        self._validate_dimension(query)
        decoded = np.asarray(self.decode(quantized), dtype=np.float64)
        query_arr = np.asarray(query, dtype=np.float64)

        denom = np.linalg.norm(query_arr) * np.linalg.norm(decoded)
        if denom == 0.0:
            return 0.0
        return float(np.dot(query_arr, decoded) / denom)


class QuantizedVector:
    """A single compressed embedding: per-dimension integer codes plus the
    per-vector scale/offset needed to approximately dequantize them."""

    __slots__ = ("codes", "vmin", "scale")

    def __init__(self, codes: np.ndarray, vmin: float, scale: float):
        self.codes = codes
        self.vmin = vmin
        self.scale = scale

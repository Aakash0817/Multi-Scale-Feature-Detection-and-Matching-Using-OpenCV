"""
Scale-Adaptive Feature Extractor
=================================
Implements SURF / SIFT feature detection and description for image stitching.

Reference:
    Bay, H., Tuytelaars, T., & Van Gool, L. (2006).
    SURF: Speeded Up Robust Features. ECCV, 3951, 404–417.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class ExtractionResult:
    """Container for feature extraction output."""
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray
    image_shape: Tuple[int, int]
    algorithm: str


class ScaleAdaptiveFeatureExtractor:
    """
    Scale-adaptive feature extractor that supports SURF and SIFT.

    SURF is preferred for speed and illumination robustness.
    SIFT is used as fallback (requires only opencv-contrib-python).

    Parameters
    ----------
    algorithm : str
        Feature algorithm to use: 'SURF', 'SIFT', or 'AUTO'.
        'AUTO' will try SURF first, then fall back to SIFT.
    hessian_threshold : int
        SURF-specific: threshold for the Hessian keypoint detector.
    n_octaves : int
        Number of pyramid octaves for multi-scale detection.
    n_octave_layers : int
        Number of layers per octave (controls scale sampling density).
    """

    SUPPORTED_ALGORITHMS = ("SURF", "SIFT", "AUTO")

    def __init__(
        self,
        algorithm: str = "AUTO",
        hessian_threshold: int = 400,
        n_octaves: int = 4,
        n_octave_layers: int = 3,
    ):
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"algorithm must be one of {self.SUPPORTED_ALGORITHMS}, got '{algorithm}'"
            )
        self.algorithm = algorithm
        self.hessian_threshold = hessian_threshold
        self.n_octaves = n_octaves
        self.n_octave_layers = n_octave_layers
        self._detector = self._build_detector()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, image: np.ndarray) -> ExtractionResult:
        """
        Detect keypoints and compute descriptors for a single image.

        Parameters
        ----------
        image : np.ndarray
            BGR or grayscale image (uint8).

        Returns
        -------
        ExtractionResult
        """
        gray = self._to_gray(image)
        keypoints, descriptors = self._detector.detectAndCompute(gray, None)
        return ExtractionResult(
            keypoints=list(keypoints),
            descriptors=descriptors if descriptors is not None else np.array([]),
            image_shape=gray.shape,
            algorithm=self._active_algorithm,
        )

    def draw_keypoints(self, image: np.ndarray, result: ExtractionResult) -> np.ndarray:
        """Return a copy of the image with detected keypoints drawn."""
        return cv2.drawKeypoints(
            image,
            result.keypoints,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_detector(self):
        """Instantiate the cv2 feature detector."""
        if self.algorithm == "SURF":
            return self._try_surf()
        if self.algorithm == "SIFT":
            return self._build_sift()
        # AUTO
        surf = self._try_surf(silent=True)
        if surf is not None:
            return surf
        return self._build_sift()

    def _try_surf(self, silent: bool = False):
        try:
            detector = cv2.xfeatures2d.SURF_create(
                hessianThreshold=self.hessian_threshold,
                nOctaves=self.n_octaves,
                nOctaveLayers=self.n_octave_layers,
                extended=False,
                upright=False,
            )
            self._active_algorithm = "SURF"
            return detector
        except AttributeError:
            if not silent:
                print(
                    "[FeatureExtractor] SURF not available "
                    "(requires opencv-contrib-python with non-free modules). "
                    "Falling back to SIFT."
                )
            return None

    def _build_sift(self):
        detector = cv2.SIFT_create(
            nOctaveLayers=self.n_octave_layers,
            contrastThreshold=0.04,
            edgeThreshold=10,
            sigma=1.6,
        )
        self._active_algorithm = "SIFT"
        return detector

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

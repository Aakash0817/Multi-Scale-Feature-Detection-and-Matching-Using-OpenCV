"""
Feature Matcher
===============
Matches descriptors between image pairs using Lowe's ratio test.

Reference:
    Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints.
    IJCV, 60(2), 91–110.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from .feature_extractor import ExtractionResult


@dataclass
class MatchResult:
    """Container for matching output."""
    good_matches: List[cv2.DMatch]
    src_pts: np.ndarray   # shape (N, 2)  - float32
    dst_pts: np.ndarray   # shape (N, 2)  - float32
    match_ratio: float    # good / total candidates


class FeatureMatcher:
    """
    Match features between two images using either BFMatcher or FLANN.

    Parameters
    ----------
    method : str
        'BF'    — Brute-Force matcher (exact, good for small descriptor sets)
        'FLANN' — Approximate nearest-neighbour (faster for large sets)
    ratio_threshold : float
        Lowe's ratio test threshold. Lower ⟹ stricter.  Default 0.75.
    """

    def __init__(self, method: str = "BF", ratio_threshold: float = 0.75):
        if method not in ("BF", "FLANN"):
            raise ValueError(f"method must be 'BF' or 'FLANN', got '{method}'")
        self.method = method
        self.ratio_threshold = ratio_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(
        self,
        result_a: ExtractionResult,
        result_b: ExtractionResult,
    ) -> MatchResult:
        """
        Match descriptors from two ExtractionResults.

        Parameters
        ----------
        result_a : ExtractionResult  – query image features
        result_b : ExtractionResult  – train image features

        Returns
        -------
        MatchResult  (empty if too few matches found)
        """
        if result_a.descriptors.size == 0 or result_b.descriptors.size == 0:
            return self._empty_result()

        desc_a, desc_b = self._ensure_float32(result_a, result_b)
        matcher = self._build_matcher(result_a.algorithm)
        raw_matches = matcher.knnMatch(desc_a, desc_b, k=2)
        good = self._ratio_test(raw_matches)

        if len(good) < 4:
            return self._empty_result()

        src_pts = np.float32(
            [result_a.keypoints[m.queryIdx].pt for m in good]
        ).reshape(-1, 2)
        dst_pts = np.float32(
            [result_b.keypoints[m.trainIdx].pt for m in good]
        ).reshape(-1, 2)

        return MatchResult(
            good_matches=good,
            src_pts=src_pts,
            dst_pts=dst_pts,
            match_ratio=len(good) / max(len(raw_matches), 1),
        )

    def draw_matches(
        self,
        img_a: "np.ndarray",
        result_a: ExtractionResult,
        img_b: "np.ndarray",
        result_b: ExtractionResult,
        match_result: MatchResult,
        max_draw: int = 50,
    ) -> "np.ndarray":
        """Return a side-by-side visualisation of matches."""
        return cv2.drawMatches(
            img_a,
            result_a.keypoints,
            img_b,
            result_b.keypoints,
            match_result.good_matches[:max_draw],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_matcher(self, algorithm: str):
        norm = cv2.NORM_L2  # both SURF and SIFT use L2
        if self.method == "BF":
            return cv2.BFMatcher(norm, crossCheck=False)
        # FLANN for float descriptors
        index_params = dict(algorithm=1, trees=5)   # KD-tree
        search_params = dict(checks=50)
        return cv2.FlannBasedMatcher(index_params, search_params)

    def _ratio_test(self, matches) -> List[cv2.DMatch]:
        good = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.ratio_threshold * n.distance:
                    good.append(m)
        return good

    @staticmethod
    def _ensure_float32(
        result_a: ExtractionResult,
        result_b: ExtractionResult,
    ) -> Tuple[np.ndarray, np.ndarray]:
        a = result_a.descriptors.astype(np.float32)
        b = result_b.descriptors.astype(np.float32)
        return a, b

    @staticmethod
    def _empty_result() -> MatchResult:
        empty = np.empty((0, 2), dtype=np.float32)
        return MatchResult(good_matches=[], src_pts=empty, dst_pts=empty, match_ratio=0.0)

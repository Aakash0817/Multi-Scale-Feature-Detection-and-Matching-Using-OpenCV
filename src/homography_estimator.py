"""
RANSAC Homography Estimator
============================
Reference:
    Fischler, M. A., & Bolles, R. C. (1981).
    Random Sample Consensus. CACM, 24(6), 381–395.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.feature_matcher import MatchResult


@dataclass
class HomographyResult:
    H: Optional[np.ndarray]
    mask: Optional[np.ndarray]
    n_inliers: int
    n_matches: int
    inlier_ratio: float
    success: bool

    @property
    def n_outliers(self):
        return self.n_matches - self.n_inliers


class RANSACHomographyEstimator:
    """
    Estimates H such that:  dst_pt ≈ H @ src_pt  (homogeneous coords)

    src_pts come from img_a (query), dst_pts from img_b (train).
    So H maps img_a → img_b coordinate frame.
    """

    def __init__(self, reproj_threshold: float = 4.0, confidence: float = 0.995,
                 max_iters: int = 3000, min_inlier_ratio: float = 0.20):
        self.reproj_threshold = reproj_threshold
        self.confidence = confidence
        self.max_iters = max_iters
        self.min_inlier_ratio = min_inlier_ratio

    def estimate(self, match_result: MatchResult) -> HomographyResult:
        n = len(match_result.good_matches)
        if n < 4:
            return self._failed(n)

        H, mask = cv2.findHomography(
            match_result.src_pts,   # img_a keypoints
            match_result.dst_pts,   # img_b keypoints
            method=cv2.RANSAC,
            ransacReprojThreshold=self.reproj_threshold,
            confidence=self.confidence,
            maxIters=self.max_iters,
        )

        if H is None or mask is None:
            return self._failed(n)

        n_inliers = int(mask.sum())
        ratio = n_inliers / n
        return HomographyResult(
            H=H, mask=mask,
            n_inliers=n_inliers, n_matches=n,
            inlier_ratio=ratio,
            success=(ratio >= self.min_inlier_ratio),
        )

    @staticmethod
    def _failed(n):
        return HomographyResult(H=None, mask=None,
                                n_inliers=0, n_matches=n,
                                inlier_ratio=0.0, success=False)

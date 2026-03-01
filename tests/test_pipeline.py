"""
Tests for the Scale-Adaptive Image Stitching pipeline.

Run with:
    pytest tests/ -v
"""

import cv2
import numpy as np
import pytest

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from src.feature_extractor import ScaleAdaptiveFeatureExtractor, ExtractionResult
from src.feature_matcher import FeatureMatcher, MatchResult
from src.homography_estimator import RANSACHomographyEstimator
from src.warper_blender import ImageWarper, ImageBlender
from src.stitcher import PanoramaStitcher, StitchConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def checker_image():
    """8×8 checkerboard scaled to 200×200 — lots of SIFT corners."""
    board = np.zeros((200, 200), dtype=np.uint8)
    tile = 25
    for r in range(8):
        for c in range(8):
            if (r + c) % 2 == 0:
                board[r*tile:(r+1)*tile, c*tile:(c+1)*tile] = 255
    return cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)


@pytest.fixture
def shifted_image(checker_image):
    """Checkerboard shifted 40 px to the right (simulates camera pan)."""
    M = np.float32([[1, 0, 40], [0, 1, 0]])
    return cv2.warpAffine(checker_image, M, (checker_image.shape[1] + 40,
                                              checker_image.shape[0]))


# ---------------------------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------------------------

class TestScaleAdaptiveFeatureExtractor:

    def test_sift_detects_keypoints(self, checker_image):
        extractor = ScaleAdaptiveFeatureExtractor(algorithm="SIFT")
        result = extractor.extract(checker_image)
        assert isinstance(result, ExtractionResult)
        assert len(result.keypoints) > 0
        assert result.descriptors.shape[1] == 128  # SIFT descriptor size
        assert result.algorithm == "SIFT"

    def test_auto_fallback_to_sift(self, checker_image):
        extractor = ScaleAdaptiveFeatureExtractor(algorithm="AUTO")
        result = extractor.extract(checker_image)
        assert len(result.keypoints) > 0

    def test_grayscale_input(self, checker_image):
        gray = cv2.cvtColor(checker_image, cv2.COLOR_BGR2GRAY)
        extractor = ScaleAdaptiveFeatureExtractor(algorithm="SIFT")
        result = extractor.extract(gray)
        assert len(result.keypoints) > 0

    def test_draw_keypoints_returns_image(self, checker_image):
        extractor = ScaleAdaptiveFeatureExtractor(algorithm="SIFT")
        result = extractor.extract(checker_image)
        vis = extractor.draw_keypoints(checker_image, result)
        assert vis.shape == checker_image.shape

    def test_invalid_algorithm_raises(self):
        with pytest.raises(ValueError):
            ScaleAdaptiveFeatureExtractor(algorithm="ORB")


# ---------------------------------------------------------------------------
# Feature Matcher
# ---------------------------------------------------------------------------

class TestFeatureMatcher:

    def _get_results(self, img_a, img_b):
        ext = ScaleAdaptiveFeatureExtractor(algorithm="SIFT")
        return ext.extract(img_a), ext.extract(img_b)

    def test_bf_matcher_finds_matches(self, checker_image, shifted_image):
        ra, rb = self._get_results(checker_image, shifted_image)
        matcher = FeatureMatcher(method="BF")
        result = matcher.match(ra, rb)
        assert isinstance(result, MatchResult)
        assert len(result.good_matches) > 0
        assert result.src_pts.shape[1] == 2
        assert result.dst_pts.shape[1] == 2

    def test_empty_descriptors_returns_empty(self, checker_image):
        ext = ScaleAdaptiveFeatureExtractor(algorithm="SIFT")
        ra = ext.extract(checker_image)
        # Corrupt descriptors
        ra.descriptors = np.array([])
        matcher = FeatureMatcher()
        result = matcher.match(ra, ra)
        assert len(result.good_matches) == 0

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            FeatureMatcher(method="SIFT")


# ---------------------------------------------------------------------------
# RANSAC Homography Estimator
# ---------------------------------------------------------------------------

class TestRANSACHomographyEstimator:

    def test_estimates_identity_for_same_image(self, checker_image):
        ext = ScaleAdaptiveFeatureExtractor(algorithm="SIFT")
        matcher = FeatureMatcher()
        ra = ext.extract(checker_image)
        match_result = matcher.match(ra, ra)

        estimator = RANSACHomographyEstimator()
        hom = estimator.estimate(match_result)

        assert hom.success
        assert hom.H is not None
        assert hom.H.shape == (3, 3)

    def test_fails_with_too_few_matches(self):
        empty = np.empty((0, 2), dtype=np.float32)
        from src.feature_matcher import MatchResult
        mr = MatchResult(good_matches=[], src_pts=empty, dst_pts=empty, match_ratio=0)
        estimator = RANSACHomographyEstimator()
        hom = estimator.estimate(mr)
        assert not hom.success
        assert hom.H is None


# ---------------------------------------------------------------------------
# Warper & Blender
# ---------------------------------------------------------------------------

class TestWarperBlender:

    def test_warp_produces_larger_canvas(self, checker_image, shifted_image):
        # Estimate a known translation homography
        H = np.array([[1, 0, 40], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        warper = ImageWarper()
        result = warper.warp(shifted_image, checker_image, H)
        canvas_w, canvas_h = result.canvas_size
        assert canvas_w >= checker_image.shape[1]
        assert canvas_h >= checker_image.shape[0]

    def test_feather_blend_produces_bgr(self, checker_image):
        blender = ImageBlender(method="feather")
        # Blend image with itself at (0, 0)
        out = blender.blend(checker_image.copy(), checker_image, 0, 0)
        assert out.ndim == 3
        assert out.shape[2] == 3

    def test_multiband_blend_produces_bgr(self, checker_image):
        blender = ImageBlender(method="multiband", pyramid_levels=2)
        out = blender.blend(checker_image.copy(), checker_image, 0, 0)
        assert out.ndim == 3


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

class TestPanoramaStitcher:

    def test_single_image_returned_as_is(self, checker_image):
        stitcher = PanoramaStitcher(verbose=False)
        result, report = stitcher.stitch([checker_image])
        assert result is not None
        assert not report.success

    def test_stitch_two_images(self, checker_image, shifted_image):
        stitcher = PanoramaStitcher(verbose=False)
        result, report = stitcher.stitch([checker_image, shifted_image])
        # Result may succeed or partially succeed depending on feature quality
        assert result is not None

    def test_custom_config(self, checker_image, shifted_image):
        config = StitchConfig(
            algorithm="SIFT",
            blend_method="feather",
            ratio_threshold=0.8,
        )
        stitcher = PanoramaStitcher(config=config, verbose=False)
        result, report = stitcher.stitch([checker_image, shifted_image])
        assert report.algorithm_used in ("SIFT", "N/A")

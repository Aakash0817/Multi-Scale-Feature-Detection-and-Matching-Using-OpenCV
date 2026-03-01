"""
Panorama Stitcher Pipeline
===========================
Two modes:
  'custom'  — educational SIFT+RANSAC pipeline (this module)
  'opencv'  — uses cv2.Stitcher (production quality, handles parallax)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass, field

from src.feature_extractor import ScaleAdaptiveFeatureExtractor
from src.feature_matcher import FeatureMatcher
from src.homography_estimator import RANSACHomographyEstimator
from src.warper_blender import ImageWarper, ImageBlender


@dataclass
class StitchConfig:
    algorithm: str        = "AUTO"
    hessian_threshold: int = 300
    n_octaves: int         = 4
    n_octave_layers: int   = 3
    matcher_method: str    = "BF"
    ratio_threshold: float = 0.70
    reproj_threshold: float = 4.0
    ransac_confidence: float = 0.995
    ransac_max_iters: int  = 3000
    min_inlier_ratio: float = 0.20
    blend_method: str      = "feather"
    feather_sigma: float   = 30.0
    pyramid_levels: int    = 4
    # Resize input images before processing (helps with large phone photos)
    max_input_dim: int     = 1600   # px — set 0 to disable


@dataclass
class StitchReport:
    n_images_in: int
    n_images_stitched: int
    algorithm_used: str
    pair_reports: List[dict] = field(default_factory=list)
    success: bool  = False
    message: str   = ""


def _resize_if_needed(img: np.ndarray, max_dim: int) -> np.ndarray:
    """Downscale image so its largest dimension ≤ max_dim."""
    if max_dim <= 0:
        return img
    h, w = img.shape[:2]
    largest = max(h, w)
    if largest <= max_dim:
        return img
    scale = max_dim / largest
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


class PanoramaStitcher:
    """
    End-to-end panorama stitcher.

    Usage
    -----
    >>> stitcher = PanoramaStitcher()
    >>> panorama, report = stitcher.stitch(images)
    >>> cv2.imwrite("panorama.jpg", panorama)
    """

    def __init__(self, config: Optional[StitchConfig] = None, verbose: bool = True):
        self.config  = config or StitchConfig()
        self.verbose = verbose
        self._build_components()

    # ── Public API ────────────────────────────────────────────────────────

    def stitch(self, images: List[np.ndarray]) -> Tuple[Optional[np.ndarray], StitchReport]:
        if len(images) < 2:
            return (images[0].copy() if images else None,
                    StitchReport(len(images), 0, "N/A",
                                 success=False, message="Need at least 2 images"))

        # Resize large images (e.g. phone photos) before processing
        resized = [_resize_if_needed(img, self.config.max_input_dim) for img in images]
        if resized[0].shape != images[0].shape:
            h0, w0 = images[0].shape[:2]
            h1, w1 = resized[0].shape[:2]
            self._log(f"Resized images from {w0}×{h0} → {w1}×{h1}")

        report = StitchReport(n_images_in=len(resized),
                              n_images_stitched=1, algorithm_used="")

        panorama = resized[0].copy()
        for i, next_img in enumerate(resized[1:], start=1):
            self._log(f"[{i}/{len(resized)-1}] Stitching pair ...")
            result, info = self._stitch_pair(panorama, next_img)
            report.pair_reports.append(info)

            if result is None:
                self._log(f"  FAILED: {info.get('error', 'unknown')}")
                continue

            panorama = result
            report.n_images_stitched += 1
            self._log(f"  OK | matches={info.get('n_matches',0)} "
                      f"| inliers={info['n_inliers']} "
                      f"({info['inlier_ratio']:.1%}) "
                      f"| alg={info['algorithm']}")

        report.algorithm_used = (report.pair_reports[0].get("algorithm", "N/A")
                                 if report.pair_reports else "N/A")
        report.success = report.n_images_stitched >= 2
        report.message = (f"Stitched {report.n_images_stitched}/"
                          f"{report.n_images_in} images")
        return panorama, report

    def stitch_from_paths(self, paths):
        images = []
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                raise FileNotFoundError(f"Could not load: {p}")
            images.append(img)
        return self.stitch(images)

    # ── Core pair logic ───────────────────────────────────────────────────

    def _stitch_pair(self, img_a: np.ndarray, img_b: np.ndarray):
        """
        Stitch img_b onto img_a.

        findHomography(pts_a, pts_b) → H maps img_a → img_b.
        The warper internally computes H_inv = inv(H) which maps img_b → img_a,
        and uses that to warpPerspective img_b into img_a's coordinate frame.
        """
        feat_a = self.extractor.extract(img_a)
        feat_b = self.extractor.extract(img_b)
        info   = {"algorithm": feat_a.algorithm,
                  "kp_a": len(feat_a.keypoints),
                  "kp_b": len(feat_b.keypoints)}

        if feat_a.descriptors.size == 0 or feat_b.descriptors.size == 0:
            info["error"] = "No features detected"
            return None, info

        match_result = self.matcher.match(feat_a, feat_b)
        info["n_matches"] = len(match_result.good_matches)

        if len(match_result.good_matches) < 10:
            info["error"] = f"Too few matches ({len(match_result.good_matches)})"
            return None, info

        hom = self.ransac.estimate(match_result)
        info.update({"n_inliers": hom.n_inliers, "inlier_ratio": hom.inlier_ratio})

        if not hom.success:
            info["error"] = (f"RANSAC failed — inlier ratio "
                             f"{hom.inlier_ratio:.2f} < {self.config.min_inlier_ratio}")
            return None, info

        # Pass H (img_a→img_b) to warper; warper inverts it internally.
        warp_result = self.warper.warp(img_b, img_a, hom.H)
        panorama    = self.blender.blend(
            warp_result.warped, img_a,
            warp_result.x_offset, warp_result.y_offset,
        )
        return panorama, info

    # ── Component construction ────────────────────────────────────────────

    def _build_components(self):
        c = self.config
        self.extractor = ScaleAdaptiveFeatureExtractor(
            algorithm=c.algorithm,
            hessian_threshold=c.hessian_threshold,
            n_octaves=c.n_octaves,
            n_octave_layers=c.n_octave_layers,
        )
        self.matcher = FeatureMatcher(
            method=c.matcher_method,
            ratio_threshold=c.ratio_threshold,
        )
        self.ransac = RANSACHomographyEstimator(
            reproj_threshold=c.reproj_threshold,
            confidence=c.ransac_confidence,
            max_iters=c.ransac_max_iters,
            min_inlier_ratio=c.min_inlier_ratio,
        )
        self.warper  = ImageWarper()
        self.blender = ImageBlender(
            method=c.blend_method,
            feather_sigma=c.feather_sigma,
            pyramid_levels=c.pyramid_levels,
        )

    def _log(self, msg):
        if self.verbose:
            print(msg)


# ── OpenCV built-in stitcher (recommended for production) ─────────────────

class OpenCVPanoramaStitcher:
    """
    Wraps cv2.Stitcher — handles bundle adjustment, cylindrical projection,
    exposure compensation and proper seam finding automatically.

    Use this when the custom SIFT+RANSAC pipeline produces distorted results
    (e.g. images taken with camera translation / large parallax).

    Usage
    -----
    >>> stitcher = OpenCVPanoramaStitcher()
    >>> panorama, status = stitcher.stitch(images)
    """

    STATUS = {0: "OK", 1: "ERR_NEED_MORE_IMGS",
              2: "ERR_HOMOGRAPHY_EST_FAIL", 3: "ERR_CAMERA_PARAMS_ADJUST_FAIL"}

    def __init__(self, mode: str = "panorama", max_input_dim: int = 1600):
        """
        Parameters
        ----------
        mode : 'panorama' (default) or 'scans'
        max_input_dim : resize input images to this max dimension first
        """
        self.mode          = mode
        self.max_input_dim = max_input_dim

    def stitch(self, images: List[np.ndarray]) -> Tuple[Optional[np.ndarray], str]:
        resized = [_resize_if_needed(img, self.max_input_dim) for img in images]

        m = cv2.STITCHER_PANORAMA if self.mode == "panorama" else cv2.STITCHER_SCANS

        if int(cv2.__version__.split('.')[0]) >= 4:
            stitcher = cv2.Stitcher_create(m)
        else:
            stitcher = cv2.createStitcher(m)

        stitcher.setPanoConfidenceThresh(0.5)

        status, panorama = stitcher.stitch(resized)
        status_str = self.STATUS.get(status, f"UNKNOWN({status})")

        if status == cv2.Stitcher_OK:
            return panorama, status_str
        return None, status_str


    def stitch_from_paths(self, paths) -> Tuple[Optional[np.ndarray], str]:
        images = []
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                raise FileNotFoundError(f"Could not load: {p}")
            images.append(img)
        return self.stitch(images)

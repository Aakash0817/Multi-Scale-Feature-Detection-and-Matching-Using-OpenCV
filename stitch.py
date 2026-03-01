#!/usr/bin/env python3
"""
Scale-Adaptive Image Stitcher - CLI
=====================================
Two modes:
  --mode custom   Custom SIFT+RANSAC pipeline (educational, rotational shots)
  --mode opencv   OpenCV built-in Stitcher   (recommended for phone photos)

Examples
--------
  python stitch.py --input ./samples --output panorama.jpg --mode opencv
  python stitch.py --input ./samples --output panorama.jpg --mode opencv --debug
  python stitch.py --input img1.jpg img2.jpg --output out.jpg --mode custom --debug
"""

import argparse
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(os.path.abspath(__file__)).parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np

try:
    from src.stitcher          import PanoramaStitcher, StitchConfig, OpenCVPanoramaStitcher
    from src.feature_extractor import ScaleAdaptiveFeatureExtractor
    from src.feature_matcher   import FeatureMatcher
except ImportError as e:
    print(f"Import error: {e}")
    src = _PROJECT_ROOT / "src"
    for f in ["__init__.py", "stitcher.py", "feature_extractor.py",
              "feature_matcher.py", "homography_estimator.py", "warper_blender.py"]:
        print(f"  {f:35s} {'OK' if (src / f).exists() else 'MISSING!'}")
    sys.exit(1)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def collect_images(inputs):
    paths = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            found = sorted(f for f in p.iterdir() if f.suffix.lower() in SUPPORTED_EXTS)
            paths.extend(found)
        elif p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            paths.append(p)
        else:
            print(f"  [warn] Skipping '{p}'")
    return paths


def resize_for_debug(img, max_dim=1600):
    """Resize image so its largest side <= max_dim before debug processing."""
    h, w = img.shape[:2]
    largest = max(h, w)
    if largest <= max_dim:
        return img
    scale = max_dim / largest
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def save_image(path: Path, img: np.ndarray, label: str):
    """Save image with error checking and a clear console message."""
    ok = cv2.imwrite(str(path), img)
    if ok:
        h, w = img.shape[:2]
        print(f"  [debug] Saved {label}")
        print(f"          {path}  ({w}x{h} px)")
    else:
        print(f"  [debug] ERROR: could not save {label}")
        print(f"          Path : {path}")
        print(f"          Check the folder exists and you have write permission.")


def run_debug(img_a: np.ndarray, img_b: np.ndarray, out_dir: Path):
    """
    Standalone debug routine — works for BOTH --mode opencv and --mode custom.
    Extracts keypoints and matches independently using SIFT, saves 3 images:
      debug_keypoints_1.jpg  keypoints on first image
      debug_keypoints_2.jpg  keypoints on second image
      debug_matches.jpg      matched keypoint pairs side by side
    All files are saved next to the output panorama.
    """
    print("\n[debug] Running debug visualisation ...")
    print(f"        Output folder: {out_dir.resolve()}")

    # Resize to a sensible size so imwrite never fails on huge phone photos
    a = resize_for_debug(img_a)
    b = resize_for_debug(img_b)

    # Always use SIFT for debug (works without opencv-contrib)
    extractor = ScaleAdaptiveFeatureExtractor(algorithm="SIFT")
    matcher   = FeatureMatcher(method="BF", ratio_threshold=0.75)

    feat_a = extractor.extract(a)
    feat_b = extractor.extract(b)
    print(f"  [debug] Keypoints: image 1 = {len(feat_a.keypoints)}, "
          f"image 2 = {len(feat_b.keypoints)}")

    # Keypoint images
    kp_img_a = extractor.draw_keypoints(a, feat_a)
    kp_img_b = extractor.draw_keypoints(b, feat_b)
    save_image(out_dir / "debug_keypoints_1.jpg", kp_img_a, "keypoints image 1")
    save_image(out_dir / "debug_keypoints_2.jpg", kp_img_b, "keypoints image 2")

    # Match visualisation
    match_result = matcher.match(feat_a, feat_b)
    n_matches = len(match_result.good_matches)
    print(f"  [debug] Good matches after ratio test: {n_matches}")

    if n_matches == 0:
        print("  [debug] No matches found — skipping debug_matches.jpg")
        return

    vis = matcher.draw_matches(a, feat_a, b, feat_b, match_result, max_draw=60)
    save_image(out_dir / "debug_matches.jpg", vis, f"match visualisation ({n_matches} matches)")


def main():
    parser = argparse.ArgumentParser(
        description="Scale-Adaptive Image Stitcher (SIFT + RANSAC)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",  nargs="+", required=True, metavar="PATH",
                        help="Image file(s) or folder(s) to stitch.")
    parser.add_argument("--output", default="panorama.jpg",
                        help="Output panorama file path.")
    parser.add_argument("--mode",   choices=["custom", "opencv"], default="opencv",
                        help="opencv = cv2.Stitcher (recommended). "
                             "custom = SIFT+RANSAC pipeline.")
    # Custom pipeline options
    parser.add_argument("--algorithm",         choices=["AUTO", "SURF", "SIFT"], default="AUTO")
    parser.add_argument("--hessian-threshold", type=int,   default=300)
    parser.add_argument("--matcher",           choices=["BF", "FLANN"], default="BF")
    parser.add_argument("--ratio-threshold",   type=float, default=0.70)
    parser.add_argument("--reproj-threshold",  type=float, default=4.0)
    parser.add_argument("--blend",             choices=["feather", "multiband"], default="feather")
    parser.add_argument("--feather-sigma",     type=float, default=30.0)
    parser.add_argument("--pyramid-levels",    type=int,   default=4)
    parser.add_argument("--max-input-dim",     type=int,   default=1600,
                        help="Resize images so largest dimension <= this. 0 = no resize.")
    parser.add_argument("--debug",  action="store_true",
                        help="Save keypoint + match debug images next to the output file.")
    parser.add_argument("--quiet",  action="store_true",
                        help="Suppress progress output.")
    args = parser.parse_args()

    # ── Collect and load images ───────────────────────────────────────────
    paths = collect_images(args.input)
    if len(paths) < 2:
        print("Error: need at least 2 images.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(paths)} image(s): " + ", ".join(p.name for p in paths))
    images = [cv2.imread(str(p)) for p in paths]
    images = [img for img in images if img is not None]

    if len(images) < 2:
        print("Error: could not load images — check file paths.", file=sys.stderr)
        sys.exit(1)

    # ── Debug visualisation (runs for BOTH modes) ─────────────────────────
    # Output directory is the folder containing the output panorama file.
    out_path = Path(args.output).resolve()
    out_dir  = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.debug:
        run_debug(images[0], images[1], out_dir)

    # ── Stitch ────────────────────────────────────────────────────────────
    t0 = time.perf_counter()

    if args.mode == "opencv":
        print(f"\nUsing OpenCV built-in Stitcher ...")
        stitcher        = OpenCVPanoramaStitcher(max_input_dim=args.max_input_dim)
        panorama, status = stitcher.stitch(images)
        elapsed          = time.perf_counter() - t0

        print(f"\n{'='*50}")
        print(f"Result : {status}")
        print(f"Time   : {elapsed:.2f}s")
        print(f"{'='*50}")

        if panorama is None:
            print(f"Stitching failed: {status}", file=sys.stderr)
            print("Try: --mode custom   or swap the order of --input images")
            sys.exit(1)

    else:  # custom
        config = StitchConfig(
            algorithm         = args.algorithm,
            hessian_threshold = args.hessian_threshold,
            matcher_method    = args.matcher,
            ratio_threshold   = args.ratio_threshold,
            reproj_threshold  = args.reproj_threshold,
            blend_method      = args.blend,
            feather_sigma     = args.feather_sigma,
            pyramid_levels    = args.pyramid_levels,
            max_input_dim     = args.max_input_dim,
        )
        stitcher        = PanoramaStitcher(config=config, verbose=not args.quiet)
        panorama, report = stitcher.stitch(images)
        elapsed          = time.perf_counter() - t0

        print(f"\n{'='*50}")
        print(f"Result    : {'SUCCESS' if report.success else 'FAILED'}")
        print(f"Message   : {report.message}")
        print(f"Algorithm : {report.algorithm_used}")
        print(f"Time      : {elapsed:.2f}s")
        print(f"{'='*50}")

        if not report.success or panorama is None:
            print("Custom pipeline failed. Try: --mode opencv", file=sys.stderr)
            sys.exit(1)

    # ── Save output ───────────────────────────────────────────────────────
    cv2.imwrite(str(out_path), panorama)
    h, w = panorama.shape[:2]
    print(f"Panorama saved -> {out_path}  ({w}x{h} px)")

    if args.debug:
        print(f"\nDebug files saved in: {out_dir}")


if __name__ == "__main__":
    main()

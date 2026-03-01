#!/usr/bin/env python3
"""
Scale-Adaptive Image Stitcher — CLI
=====================================
Two modes:
  --mode custom   Custom SIFT+RANSAC pipeline (educational, good for wide-FOV rotational shots)
  --mode opencv   OpenCV built-in Stitcher   (recommended for phone photos / parallax scenes)

Examples
--------
  python stitch.py --input ./samples --output panorama.jpg --mode opencv
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
    from src.stitcher import PanoramaStitcher, StitchConfig, OpenCVPanoramaStitcher
except ImportError as e:
    print(f"Import error: {e}")
    src = _PROJECT_ROOT / "src"
    for f in ["__init__.py","stitcher.py","feature_extractor.py",
              "feature_matcher.py","homography_estimator.py","warper_blender.py"]:
        print(f"  {f:35s} {'OK' if (src/f).exists() else 'MISSING!'}")
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


def debug_step(images, stitcher):
    if len(images) < 2 or not hasattr(stitcher, 'extractor'):
        return
    print("\n[debug] Extracting keypoints for first pair ...")
    f0 = stitcher.extractor.extract(images[0])
    f1 = stitcher.extractor.extract(images[1])
    cv2.imwrite("debug_kp0.jpg", stitcher.extractor.draw_keypoints(images[0], f0))
    cv2.imwrite("debug_kp1.jpg", stitcher.extractor.draw_keypoints(images[1], f1))
    print(f"  Saved debug_kp0.jpg ({len(f0.keypoints)} kp)")
    print(f"  Saved debug_kp1.jpg ({len(f1.keypoints)} kp)")
    match_res = stitcher.matcher.match(f0, f1)
    vis = stitcher.matcher.draw_matches(images[0], f0, images[1], f1, match_res)
    cv2.imwrite("debug_matches.jpg", vis)
    print(f"  Saved debug_matches.jpg ({len(match_res.good_matches)} matches)")


def main():
    parser = argparse.ArgumentParser(
        description="Scale-Adaptive Image Stitcher (SIFT + RANSAC)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",  nargs="+", required=True, metavar="PATH")
    parser.add_argument("--output", default="panorama.jpg")
    parser.add_argument("--mode",   choices=["custom", "opencv"], default="opencv",
                        help="'opencv' = cv2.Stitcher (recommended for phone photos). "
                             "'custom' = SIFT+RANSAC pipeline (educational).")
    # Custom pipeline options
    parser.add_argument("--algorithm",         choices=["AUTO","SURF","SIFT"], default="AUTO")
    parser.add_argument("--hessian-threshold", type=int,   default=300)
    parser.add_argument("--matcher",           choices=["BF","FLANN"], default="BF")
    parser.add_argument("--ratio-threshold",   type=float, default=0.70)
    parser.add_argument("--reproj-threshold",  type=float, default=4.0)
    parser.add_argument("--blend",             choices=["feather","multiband"], default="feather")
    parser.add_argument("--feather-sigma",     type=float, default=30.0)
    parser.add_argument("--pyramid-levels",    type=int,   default=4)
    parser.add_argument("--max-input-dim",     type=int,   default=1600,
                        help="Resize images so largest dimension ≤ this. 0 = no resize.")
    parser.add_argument("--debug",  action="store_true")
    parser.add_argument("--quiet",  action="store_true")
    args = parser.parse_args()

    paths = collect_images(args.input)
    if len(paths) < 2:
        print("Error: need at least 2 images.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(paths)} image(s): " + ", ".join(p.name for p in paths))
    images = [cv2.imread(str(p)) for p in paths]
    images = [img for img in images if img is not None]

    t0 = time.perf_counter()

    if args.mode == "opencv":
        print(f"\nUsing OpenCV built-in Stitcher (mode=panorama) ...")
        stitcher  = OpenCVPanoramaStitcher(max_input_dim=args.max_input_dim)
        panorama, status = stitcher.stitch(images)
        elapsed   = time.perf_counter() - t0
        print(f"\n{'='*50}")
        print(f"Result   : {status}")
        print(f"Time     : {elapsed:.2f}s")
        print(f"{'='*50}")
        if panorama is None:
            print(f"Stitching failed: {status}", file=sys.stderr)
            print("Try: --mode custom   or swap the order of --input images")
            sys.exit(1)

    else:  # custom
        config = StitchConfig(
            algorithm=args.algorithm,
            hessian_threshold=args.hessian_threshold,
            matcher_method=args.matcher,
            ratio_threshold=args.ratio_threshold,
            reproj_threshold=args.reproj_threshold,
            blend_method=args.blend,
            feather_sigma=args.feather_sigma,
            pyramid_levels=args.pyramid_levels,
            max_input_dim=args.max_input_dim,
        )
        stitcher = PanoramaStitcher(config=config, verbose=not args.quiet)
        if args.debug:
            debug_step(images, stitcher)
        print("\nStarting custom SIFT+RANSAC stitching pipeline ...")
        panorama, report = stitcher.stitch(images)
        elapsed = time.perf_counter() - t0
        print(f"\n{'='*50}")
        print(f"Result   : {'SUCCESS' if report.success else 'FAILED'}")
        print(f"Message  : {report.message}")
        print(f"Algorithm: {report.algorithm_used}")
        print(f"Time     : {elapsed:.2f}s")
        print(f"{'='*50}")
        if not report.success or panorama is None:
            print("Custom pipeline failed. Try: --mode opencv", file=sys.stderr)
            sys.exit(1)

    out_path = Path(args.output)
    cv2.imwrite(str(out_path), panorama)
    h, w = panorama.shape[:2]
    print(f"Panorama saved -> {out_path}  ({w}×{h} px)")


if __name__ == "__main__":
    main()

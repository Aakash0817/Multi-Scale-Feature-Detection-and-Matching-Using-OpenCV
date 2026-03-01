# Multi-Scale Feature Detection and Matching Using OpenCV

A modular, research-backed implementation of panoramic image stitching using
SIFT/SURF multi-scale feature detection, Lowe's ratio test matching, and
RANSAC-based homography estimation with cylindrical projection blending.

---

## Table of Contents

- [Overview](#overview)
- [Key Takeaways](#key-takeaways)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Architecture](#pipeline-architecture)
- [Algorithm Details](#algorithm-details)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Shooting Tips](#shooting-tips)
- [License](#license)

---

## Overview

This project implements a complete panoramic image stitching pipeline from
scratch using OpenCV, structured as a reusable Python library with a
command-line interface. The pipeline covers every stage of multi-scale
feature detection — from keypoint extraction through to cylindrical
projection, homography estimation, and seam blending.

Two stitching modes are provided:

| Mode | Description | Best For |
|---|---|---|
| `custom` | SIFT + RANSAC pipeline (this codebase) | Educational use, rotational shots |
| `opencv` | `cv2.Stitcher` wrapper | Production use, phone photos |

---

## Key Takeaways

**What was built and what was learned across this project:**

**Multi-Scale Feature Detection**
SIFT and SURF detect keypoints across multiple scales using a Gaussian
scale-space pyramid. This makes feature detection invariant to zoom, rotation,
and moderate illumination changes — critical for real-world image stitching
where no two shots are taken under identical conditions.

**Why Ratio Test Filtering Matters**
Applying Lowe's ratio test (d1/d2 < threshold) before RANSAC significantly
reduces the number of false matches passed downstream. Skipping this step
causes RANSAC to fail or produce poor homographies even with many raw matches.

**Homography Direction is Critical**
`findHomography(pts_a, pts_b)` returns H that maps image A to image B.
`warpPerspective(img_b, M)` applies M as the inverse map. Getting this
direction wrong — passing H instead of H_inv, or vice versa — produces
completely incorrect warps, which was a key debugging lesson in this project.

**Planar Homography Cannot Handle Parallax**
A single 3x3 homography assumes the scene is planar or the camera rotates
about its optical centre. When the camera translates laterally (as with phone
panoramas), parallax causes misalignment that no homography can fully
correct. Cylindrical projection solves this for rotational shots.

**Cylindrical Projection Removes Trapezoid Distortion**
Projecting both images onto a virtual cylinder before matching converts
rotational camera movement into pure horizontal translation. This eliminates
the trapezoidal warping that appears when a planar homography is applied
directly to wide-angle images taken from different positions.

**Seam Blending is Not Optional**
A hard cut at the overlap boundary is always visible regardless of how
accurate the alignment is. A cosine-gradient or Laplacian pyramid blend
across the full overlap width is necessary to produce a seamless output,
particularly where exposure or colour differs between frames.

**Image Resolution Affects Everything**
High-resolution phone photos (12MP+) cause memory issues and slow RANSAC
significantly. Downscaling to a maximum dimension of 1600px before processing
dramatically improves speed with no meaningful loss in stitching accuracy,
because feature matching does not require full resolution.

**Modular Design Enables Debugging**
Separating the pipeline into discrete modules — extractor, matcher, estimator,
warper, blender — made it possible to isolate and fix bugs at each stage
independently. This is far more practical than a monolithic stitching function.

---

## Project Structure

```
Multi-Scale-Feature-Detection-OpenCV/
|
|-- samples/                        Input images for stitching
|   |-- IMG_1813.jpg
|   `-- IMG_1814.jpg
|
|-- src/                            Core library modules
|   |-- __init__.py
|   |-- feature_extractor.py        SIFT / SURF multi-scale keypoint detection
|   |-- feature_matcher.py          BF / FLANN descriptor matching + ratio test
|   |-- homography_estimator.py     RANSAC homography estimation
|   |-- warper_blender.py           Cylindrical warp + feather / multiband blend
|   `-- stitcher.py                 Pipeline orchestrator + OpenCV wrapper
|
|-- stitch.py                       Command-line entry point
|-- panorama.jpg                    Sample output panorama
|-- requirements.txt                Python dependencies
|-- .gitignore
`-- README.md
```

---

## Requirements

- Python 3.8 or higher
- pip

| Package | Version | Purpose |
|---|---|---|
| `opencv-python` | >= 4.8 | Core computer vision |
| `numpy` | >= 1.24 | Array operations |
| `pytest` | >= 7.4 | Unit testing (optional) |

> **SURF support:** SURF is a patented algorithm and requires the non-free
> OpenCV modules. Install `opencv-contrib-python` instead of `opencv-python`
> if you need SURF. SIFT is used automatically as the default (open-source
> since OpenCV 4.4).

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/Multi-Scale-Feature-Detection-OpenCV.git
cd Multi-Scale-Feature-Detection-OpenCV
```

**2. Create and activate a virtual environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Usage

### Command-line interface

```bash
# Stitch all images in the samples folder (recommended default)
python stitch.py --input ./samples --output panorama.jpg --mode opencv

# Stitch specific files using the custom SIFT+RANSAC pipeline
python stitch.py --input samples/IMG_1813.jpg samples/IMG_1814.jpg \
                 --output panorama.jpg --mode custom

# Enable debug output (saves keypoint and match visualisations)
python stitch.py --input ./samples --output panorama.jpg --debug

# High-quality multi-band seam blending
python stitch.py --input ./samples --output panorama.jpg \
                 --mode custom --blend multiband
```

### Python library

```python
import cv2
from src.stitcher import PanoramaStitcher, StitchConfig

# Default usage — auto-selects SIFT, feather blending
stitcher = PanoramaStitcher()
images   = [cv2.imread("samples/IMG_1813.jpg"),
            cv2.imread("samples/IMG_1814.jpg")]

panorama, report = stitcher.stitch(images)
cv2.imwrite("panorama.jpg", panorama)
print(report.message)   # "Stitched 2/2 images"
```

### Custom configuration

```python
from src.stitcher import PanoramaStitcher, StitchConfig

config = StitchConfig(
    algorithm        = "SIFT",
    ratio_threshold  = 0.72,       # stricter Lowe ratio test
    reproj_threshold = 4.0,        # tighter RANSAC inlier threshold
    blend_method     = "multiband",
    pyramid_levels   = 5,
    max_input_dim    = 1600,       # downscale large phone photos first
)
stitcher = PanoramaStitcher(config=config)
panorama, report = stitcher.stitch(images)
```

### Component-level access

```python
from src.feature_extractor    import ScaleAdaptiveFeatureExtractor
from src.feature_matcher      import FeatureMatcher
from src.homography_estimator import RANSACHomographyEstimator

extractor = ScaleAdaptiveFeatureExtractor(algorithm="SIFT")
matcher   = FeatureMatcher(method="BF", ratio_threshold=0.75)
ransac    = RANSACHomographyEstimator(reproj_threshold=4.0)

feat_a       = extractor.extract(img_a)
feat_b       = extractor.extract(img_b)
match_result = matcher.match(feat_a, feat_b)
hom_result   = ransac.estimate(match_result)

print(f"Keypoints  : {len(feat_a.keypoints)} / {len(feat_b.keypoints)}")
print(f"Matches    : {len(match_result.good_matches)}")
print(f"Inliers    : {hom_result.n_inliers} ({hom_result.inlier_ratio:.1%})")
print(f"Homography :\n{hom_result.H}")
```

---

## Pipeline Architecture

```
Input Images
     |
     v
[ Cylindrical Projection ]        Removes perspective distortion
     |                            before feature matching
     v
[ Multi-Scale Feature Extraction ]
  SIFT  : 128-dim descriptor, scale + rotation invariant
  SURF  : 64-dim descriptor, faster, illumination robust
  Scale space: multi-octave Gaussian pyramid
     |
     v
[ Descriptor Matching ]
  Method  : Brute-Force (exact) or FLANN (approximate)
  Filter  : Lowe's ratio test  d1 / d2 < threshold
     |
     v
[ RANSAC Homography Estimation ]
  Randomly sample 4 correspondences
  Fit candidate homography H
  Score inliers: reprojection error < T pixels
  Refine H on full inlier consensus set
     |
     v
[ Perspective Warp ]
  Apply H_inv via warpPerspective
  Auto-sized canvas with translation offset
     |
     v
[ Seam Blending ]
  Feather  : Cosine-gradient alpha blend across overlap
  Multiband: Laplacian pyramid blend for high quality
     |
     v
[ Rectangular Crop ]
  Remove black borders from cylindrical distortion
     |
     v
Panoramic Output Image
```

---

## Algorithm Details

### Multi-Scale Feature Detection

Both SIFT and SURF detect keypoints across multiple scales using a Gaussian
scale-space pyramid, making them invariant to zoom, rotation, and moderate
illumination changes.

| Property | SIFT | SURF |
|---|---|---|
| Descriptor size | 128 floats | 64 floats |
| Scale invariant | Yes | Yes |
| Rotation invariant | Yes | Yes |
| Illumination robust | Good | Strong |
| Speed | Moderate | Fast |
| Licence | Open (OpenCV 4.4+) | Non-free (patent) |

### Lowe's Ratio Test

For each keypoint, the two nearest descriptor matches are found. A match is
accepted only if the closest distance is significantly smaller than the
second-closest:

```
match accepted  if  d1 / d2 < ratio_threshold
```

A threshold of 0.70 to 0.75 retains good matches while discarding ambiguous ones.

### RANSAC Homography Estimation

```
Initialise: N points, K iterations, T threshold, d minimum inliers

Repeat K times:
    Sample 4 random correspondences
    Compute candidate homography H
    Count inliers: reprojection_error(H, pt) < T
    Update best H if inlier count improves

Refine: recompute H using all inliers of best model
Output: H, inlier mask
```

A planar homography has 8 degrees of freedom, requiring a minimum of 4 point
correspondences to solve.

### Cylindrical Projection

Before stitching, each image is mapped onto a virtual cylinder of focal
length `f`. This converts rotational camera motion into pure horizontal
translation, eliminating the trapezoid distortion produced by a flat
homography when images are taken from different positions.

```
x_cylinder = f * arctan(x_plane / f)
y_cylinder = f * y_plane / sqrt(x_plane^2 + f^2)
```

---

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--input` | required | Image file(s) or folder path |
| `--output` | `panorama.jpg` | Output panorama file path |
| `--mode` | `opencv` | `opencv` or `custom` |
| `--algorithm` | `AUTO` | `SIFT`, `SURF`, or `AUTO` |
| `--hessian-threshold` | `300` | SURF Hessian detector threshold |
| `--matcher` | `BF` | `BF` (brute-force) or `FLANN` |
| `--ratio-threshold` | `0.70` | Lowe's ratio test cutoff |
| `--reproj-threshold` | `4.0` | RANSAC reprojection error (px) |
| `--blend` | `feather` | `feather` or `multiband` |
| `--feather-sigma` | `30.0` | Gaussian sigma for feather blend |
| `--pyramid-levels` | `4` | Laplacian pyramid levels |
| `--max-input-dim` | `1600` | Max image dimension before processing |
| `--debug` | off | Save keypoint and match debug images |
| `--quiet` | off | Suppress progress output |

---

## Configuration

The `StitchConfig` dataclass exposes all tunable parameters:

```python
@dataclass
class StitchConfig:
    algorithm         : str   = "AUTO"    # SIFT | SURF | AUTO
    hessian_threshold : int   = 300       # SURF only
    n_octaves         : int   = 4         # pyramid octaves
    n_octave_layers   : int   = 3         # layers per octave
    matcher_method    : str   = "BF"      # BF | FLANN
    ratio_threshold   : float = 0.70      # Lowe ratio test
    reproj_threshold  : float = 4.0       # RANSAC pixel threshold
    ransac_confidence : float = 0.995
    ransac_max_iters  : int   = 3000
    min_inlier_ratio  : float = 0.20      # minimum accepted inlier fraction
    blend_method      : str   = "feather" # feather | multiband
    feather_sigma     : float = 30.0
    pyramid_levels    : int   = 4
    max_input_dim     : int   = 1600      # 0 = no resize
```

---

## Shooting Tips

To get the best results from this pipeline:

- Maintain at least 30 to 50 percent overlap between consecutive frames.
- Keep camera exposure and white balance consistent across all shots.
- Rotate the camera around a fixed point rather than translating sideways.
  Parallax from lateral movement cannot be fully corrected by a single homography.
- Pass images to `--input` in left-to-right capture order.
- If stitching fails, lower `--hessian-threshold` to detect more keypoints,
  or raise `--ratio-threshold` to accept more matches.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

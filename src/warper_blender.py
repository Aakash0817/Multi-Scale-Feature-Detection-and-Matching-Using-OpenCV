"""
Image Warper & Blender
=======================
Perspective warping and clean mask-based blending for panorama stitching.

IMPORTANT — Homography convention used here:
    H = findHomography(pts_a, pts_b)   →  H maps img_a coords → img_b coords
    H_inv = inv(H)                      →  H_inv maps img_b coords → img_a coords

    warpPerspective(img_b, M, size) computes:
        output[p] = img_b[ inv(M)(p) ]
    So to get output[p_a] = img_b[corresponding_b_pixel] we need inv(M) = H,
    i.e. M = H_inv.

    perspectiveTransform(img_b_corners, H_inv) gives those corners in img_a frame.

Both operations therefore use H_inv consistently.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class WarpResult:
    warped: np.ndarray
    canvas_size: Tuple[int, int]
    x_offset: int
    y_offset: int


class ImageWarper:
    """
    Warp src image into dst's coordinate frame.

    Parameters
    ----------
    H : 3×3 homography  — forward map img_a → img_b
                          (output of findHomography(pts_a, pts_b))
    """

    def warp(self, src: np.ndarray, dst: np.ndarray, H: np.ndarray) -> WarpResult:
        h_s, w_s = src.shape[:2]
        h_d, w_d = dst.shape[:2]

        # H maps img_a → img_b.  We need H_inv (img_b → img_a) for everything.
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            H_inv = H.copy()

        # Project img_b corners into img_a coordinate frame using H_inv
        corners_src = np.float32([[0,0],[w_s,0],[w_s,h_s],[0,h_s]]).reshape(-1,1,2)
        corners_dst = np.float32([[0,0],[w_d,0],[w_d,h_d],[0,h_d]]).reshape(-1,1,2)

        projected = cv2.perspectiveTransform(corners_src, H_inv)
        all_corners = np.concatenate([corners_dst, projected], axis=0)

        x_min = int(np.floor(all_corners[:, 0, 0].min()))
        y_min = int(np.floor(all_corners[:, 0, 1].min()))
        x_max = int(np.ceil (all_corners[:, 0, 0].max()))
        y_max = int(np.ceil (all_corners[:, 0, 1].max()))

        x_offset = max(0, -x_min)
        y_offset = max(0, -y_min)

        canvas_w = x_max - x_min
        canvas_h = y_max - y_min

        # Clamp canvas to a sane size to avoid OOM on large images
        MAX_DIM = 8000
        if canvas_w > MAX_DIM or canvas_h > MAX_DIM:
            scale = min(MAX_DIM / canvas_w, MAX_DIM / canvas_h)
            canvas_w = int(canvas_w * scale)
            canvas_h = int(canvas_h * scale)
            x_offset = int(x_offset * scale)
            y_offset = int(y_offset * scale)
            S = np.array([[scale,0,0],[0,scale,0],[0,0,1]], dtype=np.float64)
            H_inv = S @ H_inv

        # Translation to shift negative coords onto the positive canvas
        T = np.array([[1,0,x_offset],[0,1,y_offset],[0,0,1]], dtype=np.float64)
        M = T @ H_inv   # final transform: img_b → canvas

        warped = cv2.warpPerspective(
            src, M, (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return WarpResult(warped=warped,
                          canvas_size=(canvas_w, canvas_h),
                          x_offset=x_offset,
                          y_offset=y_offset)


class ImageBlender:
    """
    Blend warped src with dst using binary masks + soft seam.

    Strategy:
      - Only warped has pixels  → copy warped
      - Only dst has pixels     → copy dst
      - Overlap region          → weighted blend (feather or multiband)
    """

    def __init__(self, method: str = "feather",
                 feather_sigma: float = 30.0,
                 pyramid_levels: int = 4):
        if method not in ("feather", "multiband"):
            raise ValueError("method must be 'feather' or 'multiband'")
        self.method = method
        self.feather_sigma = feather_sigma
        self.pyramid_levels = pyramid_levels

    def blend(self, warped: np.ndarray, dst: np.ndarray,
              x_offset: int, y_offset: int) -> np.ndarray:
        h_d, w_d = dst.shape[:2]
        canvas = warped.copy().astype(np.float32)

        y1, y2 = y_offset, y_offset + h_d
        x1, x2 = x_offset, x_offset + w_d

        # Clamp to canvas bounds
        cy2 = min(y2, canvas.shape[0])
        cx2 = min(x2, canvas.shape[1])
        dy  = cy2 - y1
        dx  = cx2 - x1

        if dy <= 0 or dx <= 0:
            return self._crop_black(canvas.astype(np.uint8))

        roi   = canvas[y1:cy2, x1:cx2]
        dst_f = dst[:dy, :dx].astype(np.float32)

        # Resize dst_f to match roi if they differ due to clamping
        if roi.shape[:2] != dst_f.shape[:2]:
            dst_f = cv2.resize(dst_f, (roi.shape[1], roi.shape[0]))

        mask_warp = self._content_mask(roi)
        mask_dst  = self._content_mask(dst_f)

        if self.method == "multiband":
            blended = self._multiband_blend(roi, dst_f, mask_warp, mask_dst)
        else:
            blended = self._feather_blend(roi, dst_f, mask_warp, mask_dst)

        canvas[y1:cy2, x1:cx2] = blended
        return self._crop_black(canvas.astype(np.uint8))

    # ── Blend implementations ──────────────────────────────────────────────

    def _feather_blend(self, warp_roi, dst, mask_w, mask_d):
        k     = max(3, int(self.feather_sigma * 4) | 1)
        sigma = self.feather_sigma

        w_w = cv2.GaussianBlur(mask_w.astype(np.float32), (k, k), sigma)
        w_d = cv2.GaussianBlur(mask_d.astype(np.float32), (k, k), sigma)

        total = w_w + w_d + 1e-8
        alpha = (w_d / total)[..., np.newaxis]

        blended = (1.0 - alpha) * warp_roi + alpha * dst

        # Hard-copy non-overlap regions (no blending needed there)
        only_warp = (mask_w > 0) & (mask_d == 0)
        only_dst  = (mask_d > 0) & (mask_w == 0)
        blended[only_warp] = warp_roi[only_warp]
        blended[only_dst]  = dst[only_dst]

        return blended

    def _multiband_blend(self, warp_roi, dst, mask_w, mask_d):
        seam   = self._seam_mask(mask_w, mask_d)
        seam3  = np.stack([seam]*3, axis=-1)

        lp_w = self._laplacian_pyramid(warp_roi, self.pyramid_levels)
        lp_d = self._laplacian_pyramid(dst,      self.pyramid_levels)
        gp_m = self._gaussian_pyramid (seam3,    self.pyramid_levels)

        pyr = [gm * ld + (1 - gm) * lw
               for lw, ld, gm in zip(lp_w, lp_d, gp_m)]
        blended = self._reconstruct(pyr)

        only_warp = (mask_w > 0) & (mask_d == 0)
        only_dst  = (mask_d > 0) & (mask_w == 0)
        blended[only_warp] = warp_roi[only_warp]
        blended[only_dst]  = dst[only_dst]

        return np.clip(blended, 0, 255)

    # ── Utilities ──────────────────────────────────────────────────────────

    @staticmethod
    def _content_mask(img):
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        return (gray > 0).astype(np.uint8)

    @staticmethod
    def _seam_mask(mask_w, mask_d):
        overlap = (mask_w > 0) & (mask_d > 0)
        if not overlap.any():
            return mask_d.astype(np.float32)
        cols = np.where(overlap.any(axis=0))[0]
        seam_col = int(cols.mean())
        seam = np.zeros_like(mask_w, dtype=np.float32)
        seam[:, seam_col:] = 1.0
        return cv2.GaussianBlur(seam, (51, 51), 25)

    def _gaussian_pyramid(self, img, levels):
        gp = [img.astype(np.float32)]
        cur = img.astype(np.float32)
        for _ in range(levels):
            cur = cv2.pyrDown(cur)
            gp.append(cur)
        return gp

    def _laplacian_pyramid(self, img, levels):
        gp = self._gaussian_pyramid(img, levels)
        lp = []
        for i in range(levels):
            up = cv2.pyrUp(gp[i+1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
            lp.append(gp[i].astype(np.float32) - up)
        lp.append(gp[levels].astype(np.float32))
        return lp

    def _reconstruct(self, pyramid):
        img = pyramid[-1]
        for lap in reversed(pyramid[:-1]):
            img = cv2.pyrUp(img, dstsize=(lap.shape[1], lap.shape[0]))
            img = img + lap
        return img

    @staticmethod
    def _crop_black(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        if coords is None:
            return img
        x, y, w, h = cv2.boundingRect(coords)
        return img[y:y+h, x:x+w]

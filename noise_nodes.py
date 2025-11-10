"""
Noise Pattern Extraction and Application Nodes for ComfyUI

Features:
- Mask-first region sampling (if a non-empty mask is connected, it overrides auto-detect).
- Always outputs a square crop of exactly sample_size × sample_size.
- Real noise extraction via high-pass / frequency separation, centered at 0.5.
- Seamless tiling of the noise pattern (pairwise blend or mirror+cosine).
- ApplyNoisePattern is alpha-safe:
  - Respects RGBA alpha and/or external mask (union).
  - Optional mask feathering to avoid ragged edges.
  - Optional handling of premultiplied alpha.
  - Optional compositing back over a base image (ideal for your 2-image workflow).
- ApplyNoisePattern now emits a debug info string so you can verify masks/coverage/output shape.

This file does NOT introduce new required inputs compared to your last working version.
"""

import math
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, uniform_filter

# Optional OpenCV for advanced detection and fast resizing
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[NoiseMatch] Warning: opencv-python not found. Automatic detection will use fallback method.")


# -----------------------------
# Noise Region Detection
# -----------------------------

class NoiseRegionDetector:
    """
    Automatically detects smooth regions suitable for noise sampling,
    or uses user-provided mask to extract noise sample area.

    ALWAYS returns a square crop of size (sample_size × sample_size).
    Mask-first behavior: if a mask is supplied and non-empty, it is used regardless of detection_method.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detection_method": (["automatic", "mask_based", "center_crop"], {
                    "default": "automatic",
                    "tooltip": "automatic: Find smooth regions | mask_based: Use provided mask | center_crop: Simple center"
                }),
                "sample_size": ("INT", {
                    "default": 256,
                    "min": 64,
                    "max": 1024,
                    "step": 32,
                    "tooltip": "Size of extracted square region (pixels)"
                }),
                "smoothness_threshold": ("FLOAT", {
                    "default": 0.005,
                    "min": 0.0001,
                    "max": 0.1,
                    "step": 0.0001,
                    "display": "slider",
                    "tooltip": "Lower = stricter (smoother areas only). Higher = more permissive"
                }),
            },
            "optional": {
                "mask": ("MASK",),  # Optional mask from Load Image node
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("noise_sample", "info")
    FUNCTION = "detect_region"
    CATEGORY = "image/noise"

    # ---- helpers ----
    def _resize_image_like(self, arr, target_h, target_w, is_mask=False):
        """
        Resize H×W or H×W×C numpy array to target_h × target_w (nearest for mask).
        """
        if arr.shape[0] == target_h and arr.shape[1] == target_w:
            return arr

        if HAS_CV2:
            interp = cv2.INTER_NEAREST if is_mask else (cv2.INTER_AREA if arr.shape[0] > target_h else cv2.INTER_CUBIC)
            return cv2.resize(arr, (target_w, target_h), interpolation=interp)
        else:
            from scipy.ndimage import zoom
            zoom_h = target_h / arr.shape[0]
            zoom_w = target_w / arr.shape[1]
            if arr.ndim == 2:
                order = 0 if is_mask else (1 if arr.shape[0] > target_h else 3)
                return zoom(arr, (zoom_h, zoom_w), order=order)
            else:
                order = 1 if arr.shape[0] > target_h else 3
                return zoom(arr, (zoom_h, zoom_w, 1), order=order)

    def _ensure_size(self, crop, size):
        """
        Ensure crop is exactly size × size (resizes if necessary).
        """
        if crop.shape[0] == size and crop.shape[1] == size:
            return crop
        return self._resize_image_like(crop, size, size, is_mask=False)

    # ---- main ----
    def detect_region(self, image, detection_method, sample_size, smoothness_threshold, mask=None):
        """
        Detect and crop smooth region for noise sampling.
        """
        device = image.device
        batch_size = image.shape[0]

        results = []
        info_list = []

        for i in range(batch_size):
            img = image[i].cpu().numpy()  # [H, W, C]
            h, w, _ = img.shape

            # Clamp effective size to fit inside the image
            size_eff = max(64, min(sample_size, h, w))

            # Determine if we have a usable mask for this item
            use_mask = False
            mask_data = None
            if mask is not None:
                mask_data = mask[min(i, mask.shape[0] - 1)].cpu().numpy()  # [H, W]
                if mask_data.shape[0] != h or mask_data.shape[1] != w:
                    mask_data = self._resize_image_like(mask_data, h, w, is_mask=True)
                use_mask = (mask_data > 0.5).any()

            # Mask-first behavior
            if use_mask:
                crop, info = self._extract_from_mask(img, mask_data, size_eff, smoothness_threshold)
            elif detection_method == "center_crop":
                crop, info = self._center_crop(img, size_eff)
            else:  # automatic
                crop, info = self._detect_smooth_region(img, size_eff, smoothness_threshold)

            crop = self._ensure_size(crop, size_eff)
            results.append(crop)
            info_list.append(info)

        output = np.stack(results, axis=0)
        combined_info = "\n\n".join(info_list)
        return (torch.from_numpy(output).float().to(device), combined_info)

    def _center_crop(self, img, size):
        """
        Center crop with enforced exact size.
        """
        h, w = img.shape[:2]
        cy, cx = h // 2, w // 2
        half = size // 2

        y1 = int(np.clip(cy - half, 0, max(0, h - size)))
        x1 = int(np.clip(cx - half, 0, max(0, w - size)))
        y2 = y1 + size
        x2 = x1 + size

        crop = img[y1:y2, x1:x2].copy()
        crop = self._ensure_size(crop, size)

        gray = np.dot(crop[:, :, :3], [0.299, 0.587, 0.114])
        var = np.var(gray)

        info = f"""Center Crop Method:
Region: y=[{y1}:{y2}], x=[{x1}:{x2}]
Output size: {size} x {size} pixels
Variance: {var:.6f}
Quality: {'Good - smooth area' if var < 0.01 else 'Fair' if var < 0.05 else 'Warning - textured area'}
"""
        return crop, info

    def _extract_from_mask(self, img, mask, size, threshold):
        """
        Extract region based on user-provided mask.

        Strategy:
        - If the masked area can fully contain a size×size window, scan inside the mask and
          pick the smoothest such window (lowest variance).
        - Otherwise, center on the mask's bbox and crop as best as possible.
        """
        h, w = img.shape[:2]
        gray = np.dot(img[:, :, :3], [0.299, 0.587, 0.114])

        binary_mask = (mask > 0.5).astype(np.uint8)
        coords = np.argwhere(binary_mask > 0)

        if len(coords) == 0:
            return self._center_crop(img, size)

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Try to find the smoothest window fully inside the mask
        step = max(size // 4, 16)
        best_var = float('inf')
        best_pos = None

        for y in range(y_min, max(y_min, y_max - size + 1) + 1, step):
            for x in range(x_min, max(x_min, x_max - size + 1) + 1, step):
                y2 = min(h, y + size)
                x2 = min(w, x + size)
                if (y2 - y) < size or (x2 - x) < size:
                    continue
                mask_win = binary_mask[y:y2, x:x2]
                if mask_win.mean() < 0.98:
                    continue
                window = gray[y:y2, x:x2]
                var = np.var(window)
                if var < best_var:
                    best_var = var
                    best_pos = (y, x)

        if best_pos is None:
            cy = int((y_min + y_max) / 2)
            cx = int((x_min + x_max) / 2)
            half = size // 2
            y1 = int(np.clip(cy - half, 0, max(0, h - size)))
            x1 = int(np.clip(cx - half, 0, max(0, w - size)))
            y2 = y1 + size
            x2 = x1 + size
        else:
            y1, x1 = best_pos
            y2 = y1 + size
            x2 = x1 + size

        crop = img[y1:y2, x1:x2].copy()
        crop = self._ensure_size(crop, size)

        gray_crop = np.dot(crop[:, :, :3], [0.299, 0.587, 0.114])
        var = np.var(gray_crop)
        coverage = binary_mask[y1:y2, x1:x2].mean() * 100.0

        info = f"""Mask-based Extraction:
Region: y=[{y1}:{y2}], x=[{x1}:{x2}]
Output size: {size} x {size} pixels
Mask coverage in crop: {coverage:.1f}%
Variance: {var:.6f}
Quality: {'Excellent' if var < 0.001 else 'Good' if var < 0.01 else 'Fair' if var < 0.05 else 'Poor - very textured'}
"""
        return crop, info

    def _detect_smooth_region(self, img, size, threshold):
        """
        Automatically detect smoothest region in image.
        """
        h, w, _ = img.shape
        gray = np.dot(img[:, :, :3], [0.299, 0.587, 0.114])

        if HAS_CV2 and h > size * 2 and w > size * 2:
            crop, info = self._detect_smooth_advanced(img, gray, size, threshold)
        else:
            crop, info = self._detect_smooth_simple(img, gray, size, threshold)

        return crop, info

    def _detect_smooth_advanced(self, img, gray, size, threshold):
        """
        Advanced smooth region detection using local variance and connected components.
        """
        h, w = gray.shape

        window_size = size
        local_mean = uniform_filter(gray, size=window_size, mode='reflect')
        local_mean_sq = uniform_filter(gray**2, size=window_size, mode='reflect')
        local_var = local_mean_sq - local_mean**2

        smooth_mask = (local_var < threshold).astype(np.uint8) * 255

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            smooth_mask, connectivity=8) if HAS_CV2 else (0, None, None, None)

        best_region, best_area, best_label = None, 0, -1
        if HAS_CV2 and num_labels > 1:
            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                width = stats[label, cv2.CC_STAT_WIDTH]
                height = stats[label, cv2.CC_STAT_HEIGHT]
                if width >= size and height >= size and area > best_area:
                    best_area = area
                    best_label = label
                    best_region = stats[label]

        if best_region is None:
            return self._detect_smooth_simple(img, gray, size, threshold)

        x = best_region[cv2.CC_STAT_LEFT]
        y = best_region[cv2.CC_STAT_TOP]
        region_w = best_region[cv2.CC_STAT_WIDTH]
        region_h = best_region[cv2.CC_STAT_HEIGHT]

        cx = x + region_w // 2
        cy = y + region_h // 2
        crop_size = min(size, region_w, region_h, h, w)

        x1 = int(np.clip(cx - crop_size // 2, 0, max(0, w - crop_size)))
        y1 = int(np.clip(cy - crop_size // 2, 0, max(0, h - crop_size)))
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        crop = img[y1:y2, x1:x2].copy()
        crop = self._ensure_size(crop, size)

        crop_gray = np.dot(crop[:, :, :3], [0.299, 0.587, 0.114])
        actual_var = np.var(crop_gray)
        quality = 'Excellent' if actual_var < 0.001 else 'Good' if actual_var < 0.01 else 'Fair' if actual_var < 0.05 else 'Poor'

        info = f"""Automatic Detection (Advanced):
Found smooth region with variance: {actual_var:.6f}
Region: y=[{y1}:{y2}], x=[{x1}:{x2}]
Output size: {size} x {size} pixels
Detected area (label {best_label}): {best_area} pixels
Quality: {quality}
"""
        return crop, info

    def _detect_smooth_simple(self, img, gray, size, threshold):
        """
        Simple grid search for smoothest region. Always returns size×size crop.
        """
        h, w = gray.shape

        best_var = float('inf')
        best_pos = (0, 0)

        step = max(size // 4, 32)
        for y in range(0, max(1, h - size + 1), step):
            for x in range(0, max(1, w - size + 1), step):
                window = gray[y:y+size, x:x+size]
                var = np.var(window)
                if var < best_var:
                    best_var = var
                    best_pos = (y, x)
                    if var < threshold * 0.1:
                        break

        y1, x1 = best_pos
        y2 = min(h, y1 + size)
        x2 = min(w, x1 + size)

        if (y2 - y1) < size:
            y1 = max(0, y2 - size)
        if (x2 - x1) < size:
            x1 = max(0, x2 - size)

        crop = img[y1:y2, x1:x2].copy()
        crop = self._ensure_size(crop, size)

        quality = 'Excellent' if best_var < 0.001 else 'Good' if best_var < 0.01 else 'Fair' if best_var < 0.05 else 'Poor'

        info = f"""Automatic Detection (Grid Search):
Found smoothest region with variance: {best_var:.6f}
Region: y=[{y1}:{y2}], x=[{x1}:{x2}]
Output size: {size} x {size} pixels
Quality: {quality}
{'WARNING: High variance - consider manual mask or different area' if best_var > 0.05 else ''}
"""
        return crop, info


# -----------------------------
# Real Noise Extraction
# -----------------------------

class RealNoiseExtractor:
    """
    Extracts actual noise/grain pattern from a real photograph sample.

    Algorithm:
    1. Separate high frequencies (noise) from low frequencies (image content)
    2. Center the result at 0.5
    3. Seamless tiling (optional)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_sample": ("IMAGE",),
                "high_pass_radius": ("FLOAT", {
                    "default": 2.0, "min": 0.1, "max": 20.0, "step": 0.1,
                    "display": "slider", "tooltip": "Gaussian blur sigma. Higher = coarser separation"
                }),
                "method": (["high_pass_exact", "high_pass_centered", "frequency_separation"], {
                    "default": "high_pass_centered",
                    "tooltip": "high_pass_exact: O-B+0.5 | high_pass_centered: (O-B-mean)+0.5 | frequency_separation: (O-B)/2+0.5"
                }),
                "contrast_boost": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                    "display": "slider", "tooltip": "Amplify noise contrast"
                }),
                "preserve_color": ("BOOLEAN", {"default": False, "tooltip": "Extract RGB noise per-channel vs grayscale"}),
                "make_tileable": ("BOOLEAN", {"default": True, "tooltip": "Make the noise pattern seamless"}),
                "seam_fix_width": ("INT", {"default": 16, "min": 4, "max": 128, "step": 2,
                                           "tooltip": "Edge width used for seamless pairwise blending"}),
                "tile_method": (["pairwise_blend", "mirror_cosine"], {
                    "default": "pairwise_blend", "tooltip": "Seamless method"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("noise_pattern",)
    FUNCTION = "extract_noise"
    CATEGORY = "image/noise"

    def extract_noise(self, noise_sample, high_pass_radius, method, contrast_boost, preserve_color,
                      make_tileable, seam_fix_width, tile_method):
        """
        Returns: torch.Tensor [B, H, W, C] centered at 0.5 (50% gray)
        """
        device = noise_sample.device
        batch_size = noise_sample.shape[0]

        results = []
        for i in range(batch_size):
            img = noise_sample[i].cpu().numpy()  # [H, W, C]

            if preserve_color:
                chs = min(img.shape[2], 3)
                noise_channels = []
                for c in range(chs):
                    channel_noise = self._extract_channel(img[:, :, c], high_pass_radius, contrast_boost, method)
                    noise_channels.append(channel_noise)
                while len(noise_channels) < 3:
                    noise_channels.append(noise_channels[-1])
                result = np.stack(noise_channels[:3], axis=-1)
            else:
                gray = np.dot(img[:, :, :3], [0.299, 0.587, 0.114])
                noise = self._extract_channel(gray, high_pass_radius, contrast_boost, method)
                result = np.stack([noise, noise, noise], axis=-1)

            if make_tileable:
                if tile_method == "pairwise_blend":
                    result = self._make_tileable_pairwise(result, seam_fix_width)
                else:
                    result = self._make_tileable_mirror_cosine(result)

            # Re-center to 0.5 to guard against seam fix nudging mean
            mean_shift = np.mean(result) - 0.5
            result = np.clip(result - mean_shift, 0.0, 1.0)

            results.append(result.astype(np.float32))

        output = np.stack(results, axis=0)
        return (torch.from_numpy(output).float().to(device),)

    def _extract_channel(self, channel, radius, contrast_boost, method):
        low_freq = gaussian_filter(channel, sigma=radius, mode='reflect')
        high_freq = channel.astype(np.float64) - low_freq.astype(np.float64)

        if method == "high_pass_exact":
            result = high_freq + 0.5
        elif method == "high_pass_centered":
            result = (high_freq - np.mean(high_freq)) + 0.5
        else:  # frequency_separation
            result = high_freq / 2.0 + 0.5

        if contrast_boost != 1.0:
            result = (result - 0.5) * contrast_boost + 0.5

        return np.clip(result, 0.0, 1.0)

    def _make_tileable_pairwise(self, pattern, bleed):
        """
        Pairwise blend opposing edges with a cosine ramp to enforce periodic boundaries.
        """
        out = pattern.copy()
        H, W, C = out.shape
        bleed = int(max(1, min(bleed, H // 2, W // 2)))

        if bleed > 1:
            t = (1 - np.cos(np.linspace(0, np.pi, bleed))) * 0.5
        else:
            t = np.array([0.5], dtype=np.float32)

        # Left-Right
        for k in range(bleed):
            li, ri = k, W - 1 - k
            blended = (1 - t[k]) * out[:, li, :] + t[k] * out[:, ri, :]
            out[:, li, :] = blended
            out[:, ri, :] = blended

        # Top-Bottom
        for k in range(bleed):
            ti, bi = k, H - 1 - k
            blended = (1 - t[k]) * out[ti, :, :] + t[k] * out[bi, :, :]
            out[ti, :, :] = blended
            out[bi, :, :] = blended

        return out

    def _make_tileable_mirror_cosine(self, pattern):
        """
        Seamless via 2x2 mirror mosaic + 2D Hann window, then center crop.
        Robust to any channel count and avoids H/W swap bugs.
        """
        H, W, C = pattern.shape

        # 2x2 mirror mosaic using concatenate (safer than np.block)
        top = np.concatenate([pattern, np.flip(pattern, axis=1)], axis=1)      # (H, 2W, C)
        bottom = np.concatenate([np.flip(pattern, axis=0), np.flip(np.flip(pattern, 0), 1)], axis=1)  # (H, 2W, C)
        big = np.concatenate([top, bottom], axis=0)  # (2H, 2W, C)

        H2, W2, _ = big.shape
        wy = np.hanning(H2)[:, None]         # (2H, 1)
        wx = np.hanning(W2)[None, :]         # (1, 2W)
        win = (wy * wx)[..., None]           # (2H, 2W, 1)

        bigw = big * win                      # (2H, 2W, C)

        # Extract centered H×W tile
        y1 = (H2 - H) // 2
        x1 = (W2 - W) // 2
        out = bigw[y1:y1 + H, x1:x1 + W, :]

        return out


# -----------------------------
# Apply Noise Pattern (alpha-safe)
# -----------------------------

class ApplyNoisePattern:
    """
    Applies extracted noise pattern to an image using various blend modes.

    Modes:
    - linear_light: Most accurate for frequency separation workflows
    - overlay: Gentler, preserves more contrast
    - soft_light: Subtle application
    - add: Direct addition (for testing)

    Alpha handling:
    - Respects RGBA alpha and/or external mask (union).
    - Optional feathering to avoid ragged edges.
    - Optional premultiplied-alpha handling.
    - Optional compositing over a base_image (recommended for your 2-image workflow).
    """

    BLEND_MODES = ["linear_light", "overlay", "soft_light", "add"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),           # Base or RGBA region image to apply noise to
                "noise_pattern": ("IMAGE",),   # Extracted noise pattern (centered at 0.5)
                "strength": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01,
                    "display": "slider", "tooltip": "Noise intensity. ~0.5 typical for Linear Light"
                }),
                "blend_mode": (cls.BLEND_MODES, {
                    "default": "linear_light", "tooltip": "Blending algorithm"
                }),
            },
            "optional": {
                "mask": ("MASK",),             # Optional mask (0-1)
                "base_image": ("IMAGE",),      # Optional background to composite onto
                "respect_alpha": ("BOOLEAN", {"default": True, "tooltip": "Use image alpha as mask if present"}),
                "mask_feather_px": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1,
                                              "tooltip": "Feather the effective mask (in pixels) to soften edges"}),
                "treat_input_as_premultiplied": ("BOOLEAN", {"default": False,
                                                             "tooltip": "Unpremultiply RGB by alpha before blending; re-premultiply after"}),
                "offset_x": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "offset_y": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "randomize_phase": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("result", "info")
    FUNCTION = "apply_noise"
    CATEGORY = "image/noise"

    def apply_noise(self, image, noise_pattern, strength, blend_mode,
                    mask=None, base_image=None, respect_alpha=True, mask_feather_px=1.0,
                    treat_input_as_premultiplied=False,
                    offset_x=0, offset_y=0, randomize_phase=False, seed=0):
        """
        Alpha-safe noise application with optional compositing over base_image.
        """
        device = image.device
        batch_size = image.shape[0]

        rng = np.random.default_rng(seed if randomize_phase else None)

        results = []
        infos = []

        for i in range(batch_size):
            img_np = image[i].cpu().numpy()  # [H, W, C]
            H, W, C = img_np.shape

            # Split RGB / Alpha (if present)
            if C >= 4:
                rgb = img_np[:, :, :3].astype(np.float32)
                alpha_from_img = img_np[:, :, 3].astype(np.float32)
            else:
                rgb = img_np.astype(np.float32)
                alpha_from_img = None

            # Choose noise pattern for this batch item (use first if fewer)
            pattern_idx = min(i, noise_pattern.shape[0] - 1)
            pattern = noise_pattern[pattern_idx].cpu().numpy()  # [pH, pW, Cpat]
            # Use 3 channels for blending
            if pattern.shape[2] > 3:
                pattern = pattern[:, :, :3].astype(np.float32)
            else:
                pattern = pattern.astype(np.float32)

            # Phase offset
            if randomize_phase:
                oy = int(rng.integers(0, max(1, pattern.shape[0])))
                ox = int(rng.integers(0, max(1, pattern.shape[1])))
            else:
                oy = int(offset_y) % max(1, pattern.shape[0])
                ox = int(offset_x) % max(1, pattern.shape[1])
            if oy != 0 or ox != 0:
                pattern = np.roll(pattern, shift=(oy, ox), axis=(0, 1))

            # Tile to image size
            tiled_pattern = self._tile_pattern(pattern, H, W)

            # Effective mask: alpha union external mask
            eff_mask = None
            mask_used = []
            if respect_alpha and (alpha_from_img is not None):
                eff_mask = alpha_from_img.copy()
                mask_used.append("alpha")
            if mask is not None:
                ext = mask[min(i, mask.shape[0] - 1)].cpu().numpy().astype(np.float32)  # [H, W]
                if ext.shape[0] != H or ext.shape[1] != W:
                    ext = self._resize_mask(ext, H, W)
                eff_mask = ext if eff_mask is None else np.maximum(eff_mask, ext)
                mask_used.append("external-mask")
            if eff_mask is None:
                eff_mask = np.ones((H, W), dtype=np.float32)
                mask_used.append("none->full")

            eff_mask = np.clip(eff_mask, 0.0, 1.0)

            # Feather if requested
            if mask_feather_px > 0.0:
                eff_mask = gaussian_filter(eff_mask, sigma=float(mask_feather_px))

            eff_mask = np.clip(eff_mask, 0.0, 1.0)
            eff_mask_3 = eff_mask[:, :, None]

            # Handle premultiplied if requested
            if treat_input_as_premultiplied and (alpha_from_img is not None):
                denom = np.maximum(alpha_from_img, 1e-6)
                work_rgb = (rgb / denom[:, :, None]).clip(0.0, 1.0)
            else:
                work_rgb = rgb

            # Blend in straight color space
            blended_rgb = self._apply_blend(work_rgb, tiled_pattern, strength, blend_mode)

            # If we unpremultiplied RGB earlier, re-premultiply now
            if treat_input_as_premultiplied and (alpha_from_img is not None):
                blended_rgb = (blended_rgb * alpha_from_img[:, :, None]).clip(0.0, 1.0)

            # Restrict blending to mask area
            masked_result_rgb = rgb * (1.0 - eff_mask_3) + blended_rgb * eff_mask_3

            # Optional composite over base_image
            used_base = False
            if base_image is not None:
                base_np = base_image[min(i, base_image.shape[0] - 1)].cpu().numpy().astype(np.float32)
                if base_np.shape[0] != H or base_np.shape[1] != W:
                    base_np = self._resize_image_like(base_np, H, W, is_mask=False)
                base_rgb = base_np[:, :, :3]
                out_rgb = base_rgb * (1.0 - eff_mask_3) + masked_result_rgb * eff_mask_3
                result_np = out_rgb
                used_base = True
            else:
                # Preserve alpha if present; otherwise return RGB
                if alpha_from_img is not None:
                    result_np = np.concatenate([masked_result_rgb, alpha_from_img[:, :, None]], axis=-1)
                else:
                    result_np = masked_result_rgb

            # Collect info for debugging
            cov = float(eff_mask.mean()) * 100.0
            ch_out = result_np.shape[2]
            info = (
                f"Apply Noise (alpha-safe)\n"
                f"- Image: {H}x{W}x{C} | Pattern: {tiled_pattern.shape[0]}x{tiled_pattern.shape[1]}x{tiled_pattern.shape[2]}\n"
                f"- Blend: {blend_mode} @ {strength:.3f}\n"
                f"- Mask sources: {', '.join(mask_used)} | Coverage: {cov:.1f}% | Feather: {mask_feather_px}px\n"
                f"- Premultiplied handling: {'ON' if (treat_input_as_premultiplied and alpha_from_img is not None) else 'OFF'}\n"
                f"- Base image composited: {'YES' if used_base else 'NO'}\n"
                f"- Output channels: {ch_out} ({'RGBA' if ch_out==4 else 'RGB'})"
            )

            results.append(result_np.astype(np.float32))
            infos.append(info)

        output = np.stack(results, axis=0)
        info_join = "\n\n".join(infos)
        return (torch.from_numpy(output).float().to(device), info_join)

    # --- helpers for ApplyNoisePattern ---

    def _resize_image_like(self, arr, target_h, target_w, is_mask=False):
        if arr.shape[0] == target_h and arr.shape[1] == target_w:
            return arr
        if HAS_CV2:
            interp = cv2.INTER_NEAREST if is_mask else (cv2.INTER_AREA if arr.shape[0] > target_h else cv2.INTER_CUBIC)
            return cv2.resize(arr, (target_w, target_h), interpolation=interp)
        else:
            from scipy.ndimage import zoom
            zoom_h = target_h / arr.shape[0]
            zoom_w = target_w / arr.shape[1]
            if arr.ndim == 2:
                order = 0 if is_mask else (1 if arr.shape[0] > target_h else 3)
                return zoom(arr, (zoom_h, zoom_w), order=order)
            else:
                order = 1 if arr.shape[0] > target_h else 3
                return zoom(arr, (zoom_h, zoom_w, 1), order=order)

    def _resize_mask(self, mask, target_h, target_w):
        return self._resize_image_like(mask, target_h, target_w, is_mask=True)

    def _tile_pattern(self, pattern, target_height, target_width):
        ph, pw, _ = pattern.shape
        repeat_h = max(1, math.ceil(target_height / max(1, ph)))
        repeat_w = max(1, math.ceil(target_width / max(1, pw)))
        tiled = np.tile(pattern, (repeat_h, repeat_w, 1))
        return tiled[:target_height, :target_width, :]

    def _apply_blend(self, base_rgb, pattern_rgb, strength, mode):
        if mode == "linear_light":
            return self._blend_linear_light(base_rgb, pattern_rgb, strength)
        elif mode == "overlay":
            return self._blend_overlay(base_rgb, pattern_rgb, strength)
        elif mode == "soft_light":
            return self._blend_soft_light(base_rgb, pattern_rgb, strength)
        elif mode == "add":
            return self._blend_add(base_rgb, pattern_rgb, strength)
        else:
            return self._blend_linear_light(base_rgb, pattern_rgb, strength)

    def _blend_linear_light(self, base, blend, strength):
        offset = (blend - 0.5) * 2.0 * strength
        result = base + offset
        return np.clip(result, 0.0, 1.0)

    def _blend_overlay(self, base, blend, strength):
        mask = base < 0.5
        result = np.where(
            mask,
            2.0 * base * blend,
            1.0 - 2.0 * (1.0 - base) * (1.0 - blend)
        )
        result = base * (1.0 - strength) + result * strength
        return np.clip(result, 0.0, 1.0)

    def _blend_soft_light(self, base, blend, strength):
        result = (1.0 - 2.0 * blend) * base * base + 2.0 * blend * base
        result = base * (1.0 - strength) + result * strength
        return np.clip(result, 0.0, 1.0)

    def _blend_add(self, base, blend, strength):
        result = base + (blend - 0.5) * strength
        return np.clip(result, 0.0, 1.0)


# -----------------------------
# Noise Pattern Visualizer
# -----------------------------

class NoisePatternVisualizer:
    """
    Helper node to visualize noise pattern statistics and quality.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_pattern": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("pattern", "stats")
    FUNCTION = "analyze"
    CATEGORY = "image/noise"
    OUTPUT_NODE = True

    def analyze(self, noise_pattern):
        pattern = noise_pattern[0].cpu().numpy()
        mean = np.mean(pattern)
        std = np.std(pattern)
        min_val = np.min(pattern)
        max_val = np.max(pattern)
        center_offset = abs(mean - 0.5)

        chinfo = ""
        if pattern.shape[-1] >= 3:
            r_mean = np.mean(pattern[:, :, 0])
            g_mean = np.mean(pattern[:, :, 1])
            b_mean = np.mean(pattern[:, :, 2])
            chinfo = f"""
Channel means:
  R: {r_mean:.4f}
  G: {g_mean:.4f}
  B: {b_mean:.4f}
"""

        stats = f"""Noise Pattern Analysis:
Size: {pattern.shape[1]} x {pattern.shape[0]} pixels
Channels: {pattern.shape[2]}
Mean: {mean:.4f} (should be ~0.5)
Std Dev: {std:.4f}
Min: {min_val:.4f}
Max: {max_val:.4f}
Center Offset: {center_offset:.4f} ({'✓ OK' if center_offset < 0.01 else '⚠ WARNING: Not centered!'}){chinfo}
Status: {'Ready to apply' if center_offset < 0.01 and std > 0.001 else 'Check extraction settings'}
"""

        return (noise_pattern, stats)


# -----------------------------
# Node registration
# -----------------------------

NODE_CLASS_MAPPINGS = {
    "NoiseRegionDetector": NoiseRegionDetector,
    "RealNoiseExtractor": RealNoiseExtractor,
    "ApplyNoisePattern": ApplyNoisePattern,
    "NoisePatternVisualizer": NoisePatternVisualizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NoiseRegionDetector": "Detect Noise Sample Region",
    "RealNoiseExtractor": "Extract Real Noise Pattern",
    "ApplyNoisePattern": "Apply Noise Pattern (Alpha-safe)",
    "NoisePatternVisualizer": "Analyze Noise Pattern (Debug)",
}
import argparse
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .dither import dither, compute_local_density
from .svg_writer import write_svg_circles


def _parse_palette(palette_arg: Optional[str]) -> List[str]:
	if not palette_arg:
		return ["#000000"]
	parts = [p.strip() for p in palette_arg.split(",") if p.strip()]
	parsed: List[str] = []
	for p in parts:
		if not p.startswith("#"):
			p = "#" + p
		if len(p) not in (4, 7):
			raise ValueError(f"Invalid color: {p}")
		parsed.append(p.lower())
	return parsed


def _load_and_resize_to_grid(image_path: str, width_mm: float, height_mm: Optional[float], step_mm: float) -> Tuple[np.ndarray, float, float, int, int]:
	img = Image.open(image_path).convert("RGB")
	w_px, h_px = img.size
	if width_mm <= 0 or step_mm <= 0:
		raise ValueError("width-mm and step-mm must be > 0")
	cols = max(1, int(round(width_mm / step_mm)))
	if height_mm is None:
		aspect = h_px / w_px
		height_mm_calc = width_mm * aspect
	else:
		height_mm_calc = height_mm
	rows = max(1, int(round(height_mm_calc / step_mm)))
	if cols < 1 or rows < 1:
		raise ValueError("Computed grid is empty; adjust width-mm/height-mm/step-mm")
	resized = img.resize((cols, rows), Image.LANCZOS)
	arr = np.asarray(resized, dtype=np.uint8)
	return arr, width_mm, height_mm_calc, cols, rows


def _to_grayscale_luma(rgb: np.ndarray) -> np.ndarray:
	# Expect shape (H, W, 3) uint8
	r = rgb[..., 0].astype(np.float32)
	g = rgb[..., 1].astype(np.float32)
	b = rgb[..., 2].astype(np.float32)
	# Rec. 601 luma approximation
	gray = 0.299 * r + 0.587 * g + 0.114 * b
	return np.clip(gray, 0, 255).astype(np.uint8)


def _hex_to_rgb(hx: str) -> Tuple[int, int, int]:
	h = hx.lstrip("#")
	if len(h) == 3:
		r = int(h[0] * 2, 16)
		g = int(h[1] * 2, 16)
		b = int(h[2] * 2, 16)
	else:
		r = int(h[0:2], 16)
		g = int(h[2:4], 16)
		b = int(h[4:6], 16)
	return (r, g, b)


def _compute_color_contribution(rgb: np.ndarray, target_color: Tuple[int, int, int], threshold: float = 0.5, white_threshold: float = 0.85) -> np.ndarray:
	"""
	Compute how much of target_color is present at each pixel.
	Returns grayscale (0-255) representing contribution/density of that color.
	Preserves white/bright areas while allowing colors to overlap in darker regions.
	
	threshold: 0-1, higher = stricter (less area covered, more white space)
	"""
	rgbf = rgb.astype(np.float32)
	target = np.array(target_color, dtype=np.float32)
	
	# Calculate image brightness/luminance
	luma = 0.299 * rgbf[..., 0] + 0.587 * rgbf[..., 1] + 0.114 * rgbf[..., 2]
	target_luma = 0.299 * target[0] + 0.587 * target[1] + 0.114 * target[2]
	
	# Strong white preservation: if pixel is very bright, zero contribution
	brightness_ratio = luma / 255.0
	white_mask = brightness_ratio > white_threshold  # Preserve near-white areas
	
	# Compute color similarity (lower distance = more similar)
	diff = rgbf - target[None, None, :]
	dist = np.sqrt(np.sum(diff * diff, axis=2))  # (H,W)
	max_dist = np.sqrt(3 * 255 * 255)
	normalized_dist = dist / max_dist
	
	# Color contribution based on similarity with exponential falloff
	# Lower threshold means color appears more readily
	color_match = np.exp(-normalized_dist / (1.0 - threshold + 0.01))
	
	# Modulate by darkness: colors only appear on darker pixels
	# Bright pixels should remain white
	darkness_factor = 1.0 - brightness_ratio
	darkness_factor = np.power(darkness_factor, 0.5)  # Soften the falloff
	
	# Combine: color appears where it matches AND pixel is dark enough
	contribution = color_match * darkness_factor * 255.0
	
	# Zero out pure white areas completely
	contribution[white_mask] = 0
	
	return np.clip(contribution, 0, 255).astype(np.uint8)


def _extract_points_from_mask(mask_bw: np.ndarray, step_mm: float, density: Optional[np.ndarray] = None) -> List[Tuple[float, float, float]]:
	# mask_bw: bool or {0,1} shape (H,W)
	# density: optional float32 (H,W) with 0-1 values for variable sizing
	# Returns list of (x_mm, y_mm, radius_scale) where radius_scale is 0.5-1.5
	points: List[Tuple[float, float, float]] = []
	rows, cols = mask_bw.shape
	# Place points on grid centers
	for y in range(rows):
		for x in range(cols):
			if mask_bw[y, x]:
				px = (x + 0.5) * step_mm
				py = (y + 0.5) * step_mm
				# Variable dot size based on local density
				if density is not None:
					# Scale radius from 0.5x to 1.5x based on darkness
					# Darker areas = larger dots
					scale = 0.5 + density[y, x] * 1.0
				else:
					scale = 1.0
				points.append((px, py, scale))
	return points


def run_cli(argv: Optional[Sequence[str]] = None) -> None:
	parser = argparse.ArgumentParser(description="Raster to pen-plotter-optimized SVG (dots)")
	parser.add_argument("input", help="Input raster image (png, jpg, etc.)")
	parser.add_argument("-o", "--output", default="out.svg", help="Output SVG path")
	parser.add_argument("--width-mm", type=float, required=True, help="Target width in mm")
	parser.add_argument("--height-mm", type=float, default=None, help="Target height in mm (optional)")
	parser.add_argument("--step-mm", type=float, default=0.5, help="Grid spacing in mm between sample points")
	parser.add_argument("--dot-mm", type=float, default=None, help="Dot diameter in mm (defaults to 0.8*step-mm)")
	parser.add_argument("--palette", type=str, default=None, help="Comma-separated hex colors, e.g. #000,#f00,#0af")
	parser.add_argument("--mode", choices=["grayscale", "palette"], default="grayscale", help="Dithering mode")
	parser.add_argument("--threshold", type=float, default=0.5, help="Color threshold 0-1 (higher = more selective colors, palette mode only)")
	parser.add_argument("--white-threshold", type=float, default=0.85, help="White preservation 0-1 (higher = preserve only brightest whites, palette mode only)")
	parser.add_argument("--dither-method", type=str, default="floyd-steinberg", 
	                    choices=["floyd-steinberg", "blue-noise", "ordered", "white-noise"],
	                    help="Dithering algorithm: floyd-steinberg (default), blue-noise (stochastic), ordered (Bayer), white-noise (random)")
	parser.add_argument("--variable-dots", action="store_true", help="Enable variable dot sizes based on local tone (darker = larger)")
	parser.add_argument("--order", choices=["none", "nearest"], default="nearest", help="Point visiting order for travel minimization")
	args = parser.parse_args(argv)

	palette_hex = _parse_palette(args.palette)
	if args.mode == "grayscale" and len(palette_hex) != 1:
		raise SystemExit("In grayscale mode, provide zero or one color. For multi-color, use --mode palette")

	rgb, width_mm, height_mm, cols, rows = _load_and_resize_to_grid(
		args.input, args.width_mm, args.height_mm, args.step_mm
	)
	dot_mm = args.dot_mm if args.dot_mm and args.dot_mm > 0 else max(0.1, 0.8 * args.step_mm)

	layers: List[Tuple[str, str, List[Tuple[float, float, float]], float]] = []  # (layer_name, hex, points_with_scale, base_radius)
	if args.mode == "grayscale":
		gray = _to_grayscale_luma(rgb)
		# Compute density map if variable dots enabled
		density = compute_local_density(gray, kernel_size=5) if args.variable_dots else None
		# Apply selected dithering method
		mask = dither(gray, method=args.dither_method) > 0
		points = _extract_points_from_mask(mask, args.step_mm, density=density)
		layers.append(("black", palette_hex[0], points, 0.5 * dot_mm))
	else:
		# Overlapping color layers: each color gets its own dithered contribution
		for pi, hx in enumerate(palette_hex):
			target_rgb = _hex_to_rgb(hx)
			# Compute how much this color contributes at each pixel
			contribution = _compute_color_contribution(rgb, target_rgb, threshold=args.threshold, white_threshold=args.white_threshold)
			# Compute density map if variable dots enabled
			density = compute_local_density(contribution, kernel_size=5) if args.variable_dots else None
			# Apply selected dithering method
			mask = dither(contribution, method=args.dither_method) > 0
			points = _extract_points_from_mask(mask, args.step_mm, density=density)
			layer_name = f"color_{pi+1}_{hx.lstrip('#')}"
			layers.append((layer_name, hx, points, 0.5 * dot_mm))

	# Ensure output directory exists
	os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
	write_svg_circles(
		layers=layers,
		width_mm=width_mm,
		height_mm=height_mm,
		order=args.order,
		output_path=args.output,
	)
	total_dots = sum(len(p) for _, _, p, _ in layers)
	print(f"Wrote {args.output}  ({cols}x{rows} grid, {total_dots} dots, method={args.dither_method}, variable_dots={args.variable_dots})")

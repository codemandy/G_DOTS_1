import argparse
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .dither import dither, compute_local_density, blue_noise_dither
from .svg_writer import write_svg_circles
from .poisson import poisson_disk_variable


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
	# Place points on grid centers; jitter can be applied by outer caller via kwargs
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
	parser.add_argument("--opacity-min", type=float, default=0.15, help="Minimum opacity for dots (0-1, default 0.15)")
	parser.add_argument("--opacity-max", type=float, default=0.85, help="Maximum opacity for dots (0-1, default 0.85)")
	parser.add_argument("--order", choices=["none", "nearest"], default="nearest", help="Point visiting order for travel minimization")
	parser.add_argument("--jitter-mm", type=float, default=0.0, help="Uniform jitter amplitude in mm to break grid alignment (0 to disable)")
	parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible jitter/noise")
	parser.add_argument("--fill-mode", choices=["opacity", "rgba"], default="opacity", help="Use SVG fill-opacity attribute (opacity) or rgba(...) fill (rgba)")
	parser.add_argument("--tone-stacks", type=int, default=1, help="Number of stochastic tone stacks per color (>=1). Uses blue-noise to distribute tone across stacks")
	parser.add_argument("--stack-alpha", type=float, default=0.25, help="Fixed alpha (0-1) per stack when tone stacking is enabled")
	parser.add_argument("--placement", choices=["grid", "poisson"], default="grid", help="Point placement strategy: grid (default) or poisson (variable density)")
	parser.add_argument("--poisson-k", type=int, default=30, help="Bridson k (candidates per active point) for Poisson sampling")
	parser.add_argument("--poisson-min-scale", type=float, default=0.6, help="Minimum scale factor for local radius in Poisson (lower -> tighter min spacing)")
	parser.add_argument("--watercolor", action="store_true", help="Use per-point sampled color and alpha for watercolor-like blending (poisson only)")
	parser.add_argument("--poisson-density-power", type=float, default=1.0, help="Exponent to shape darkness->density mapping ( >1 emphasizes darks, <1 lightens effect)")
	parser.add_argument("--alpha-min", type=float, default=0.15, help="Minimum per-point alpha for watercolor mode")
	parser.add_argument("--alpha-max", type=float, default=0.85, help="Maximum per-point alpha for watercolor mode")
	parser.add_argument("--color-sample-radius", type=int, default=3, help="Pixel radius for color averaging (NxN neighborhood, default 3)")
	parser.add_argument("--quantize-colors", type=int, default=None, help="Optional: reduce to N dominant colors via k-means")
	parser.add_argument("--radius-min-scale", type=float, default=0.6, help="Minimum radius multiplier relative to r (default 0.6)")
	parser.add_argument("--radius-max-scale", type=float, default=1.0, help="Maximum radius multiplier in dark areas (default 1.0)")
	args = parser.parse_args(argv)

	palette_hex = _parse_palette(args.palette)
	if args.mode == "grayscale" and len(palette_hex) != 1:
		raise SystemExit("In grayscale mode, provide zero or one color. For multi-color, use --mode palette")

	rgb, width_mm, height_mm, cols, rows = _load_and_resize_to_grid(
		args.input, args.width_mm, args.height_mm, args.step_mm
	)
	dot_mm = args.dot_mm if args.dot_mm and args.dot_mm > 0 else max(0.1, 0.8 * args.step_mm)

	layers: List[Tuple[str, str, List[Tuple[float, float, float]], float]] = []  # (layer_name, hex, points_with_scale, base_radius)
	# RNG for jitter
	rng = np.random.default_rng(args.seed) if args.seed is not None else np.random.default_rng()

	def _apply_jitter(points: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
		if args.jitter_mm <= 0:
			return points
		jittered: List[Tuple[float, float, float]] = []
		for (px, py, scale) in points:
			jx = rng.uniform(-args.jitter_mm, args.jitter_mm)
			jy = rng.uniform(-args.jitter_mm, args.jitter_mm)
			nx = min(max(px + jx, 0.0), width_mm)
			ny = min(max(py + jy, 0.0), height_mm)
			jittered.append((nx, ny, scale))
		return jittered

	if args.placement == "poisson":
		# Poisson placement (variable density): generate points directly in mm space
		# Build a function mapping mm coords -> local radius scale in [min_scale, 1.0+]
		H, W = rgb.shape[0], rgb.shape[1]
		
		# Helper: sample and average color over NxN neighborhood
		def sample_color_averaged(x_mm: float, y_mm: float, radius_px: int) -> Tuple[int, int, int]:
			"""Sample and average color over NxN neighborhood centered at (x_mm, y_mm)"""
			xi = int(round((x_mm / width_mm) * (W - 1)))
			yi = int(round((y_mm / height_mm) * (H - 1)))
			
			r_sum, g_sum, b_sum, count = 0, 0, 0, 0
			for dy in range(-radius_px, radius_px + 1):
				for dx in range(-radius_px, radius_px + 1):
					x_sample = max(0, min(W - 1, xi + dx))
					y_sample = max(0, min(H - 1, yi + dy))
					r, g, b = rgb[y_sample, x_sample]
					r_sum += int(r)
					g_sum += int(g)
					b_sum += int(b)
					count += 1
			return (r_sum // count, g_sum // count, b_sum // count)
		
		# Helper: quantize colors using k-means
		def quantize_palette(colors: List[Tuple[int, int, int, float]], n_colors: int) -> List[Tuple[int, int, int, float]]:
			"""Reduce colors to n_colors using k-means clustering"""
			try:
				from sklearn.cluster import KMeans
			except ImportError:
				print("[warning] scikit-learn not available; skipping quantization")
				return colors
			
			if len(colors) <= n_colors:
				return colors
			
			# Extract RGB only for clustering
			rgb_only = np.array([(r, g, b) for r, g, b, a in colors], dtype=np.float32)
			alphas = [a for r, g, b, a in colors]
			
			kmeans = KMeans(n_clusters=n_colors, random_state=args.seed or 42, n_init=10)
			kmeans.fit(rgb_only)
			centers = kmeans.cluster_centers_.astype(int)
			labels = kmeans.labels_
			
			# Map each color to nearest center, preserve alpha
			return [(int(centers[label][0]), int(centers[label][1]), int(centers[label][2]), alphas[i]) 
			        for i, label in enumerate(labels)]
		
		# Sample grayscale intensity from image at mm position using nearest neighbor
		def sample_darkness(x_mm: float, y_mm: float) -> float:
			xi = min(W - 1, max(0, int(round((x_mm / width_mm) * (W - 1)))))
			yi = min(H - 1, max(0, int(round((y_mm / height_mm) * (H - 1)))))
			r, g, b = rgb[yi, xi]
			luma = 0.299 * float(r) + 0.587 * float(g) + 0.114 * float(b)
			# darkness in [0,1]; raise to power to shape density response
			d = 1.0 - (luma / 255.0)
			return float(np.clip(d, 0.0, 1.0)) ** float(max(0.01, args.poisson_density_power))
		# Map darkness -> local radius scale: smaller radius in darker regions
		# radius_scale = (1 - darkness) * (1 - min_scale) + min_scale
		def r_scale_fn(x: float, y: float) -> float:
			d = sample_darkness(x, y)
			return max(args.poisson_min_scale, (1.0 - d) * (1.0 - args.poisson_min_scale) + args.poisson_min_scale)
		
		base_r = max(0.1, args.step_mm)
		pts = poisson_disk_variable(
			width_mm,
			height_mm,
			base_r,
			r_scale_fn,
			min_scale=args.poisson_min_scale,
			k=args.poisson_k,
			seed=args.seed,
		)
		# Prepare layers
		if args.watercolor:
			# Collect all point data first
			point_data: List[Tuple[float, float, float, int, int, int, float]] = []  # x, y, radius_scale, r, g, b, alpha
			for (x, y, r_local) in pts:
				# Variable radius: 0.6-1.0× r_local based on darkness
				d = sample_darkness(x, y)
				radius_scale = args.radius_min_scale + (args.radius_max_scale - args.radius_min_scale) * d
				radius = radius_scale * r_local
				
				# Average color over neighborhood
				rc, gc, bc = sample_color_averaged(x, y, args.color_sample_radius)
				
				# Opacity from darkness using configured range
				alpha = args.alpha_min + (args.alpha_max - args.alpha_min) * float(d)
				alpha = float(max(0.0, min(1.0, alpha)))
				
				point_data.append((x, y, radius / (0.5 * dot_mm if dot_mm > 0 else 1.0), rc, gc, bc, alpha))
			
			# Group by color into layers
			from collections import defaultdict
			color_groups: dict = defaultdict(list)
			
			# Apply optional quantization first
			if args.quantize_colors and args.quantize_colors > 0:
				# Extract colors for quantization
				colors_with_alpha = [(r, g, b, a) for x, y, rs, r, g, b, a in point_data]
				quantized = quantize_palette(colors_with_alpha, args.quantize_colors)
				# Update point_data with quantized colors
				point_data = [(x, y, rs, qr, qg, qb, qa) 
				             for (x, y, rs, _, _, _, _), (qr, qg, qb, qa) in zip(point_data, quantized)]
			
			# Group points by RGB color
			for x, y, radius_scale, rc, gc, bc, alpha in point_data:
				color_key = (rc, gc, bc)
				color_groups[color_key].append((x, y, radius_scale, alpha))
			
			# Create a layer for each color
			per_point_layers = []
			for i, (color_rgb, point_list) in enumerate(sorted(color_groups.items())):
				rc, gc, bc = color_rgb
				color_hex = f"#{rc:02x}{gc:02x}{bc:02x}"
				points = [(x, y, rs) for x, y, rs, a in point_list]
				# Per-point alpha for this layer
				alphas = [(rc, gc, bc, a) for x, y, rs, a in point_list]
				layers.append((f"watercolor_{i+1}_{color_hex.lstrip('#')}", color_hex, points, 0.5 * dot_mm))
				per_point_layers.append(alphas)
		else:
			# Palette or grayscale by nearest palette color, fixed alpha range
			per_color_points: List[List[Tuple[float, float, float]]] = [[] for _ in palette_hex]
			for (x, y, r_local) in pts:
				radius = 0.6 * r_local
				xi = min(W - 1, max(0, int(round((x / width_mm) * (W - 1)))))
				yi = min(H - 1, max(0, int(round((y / height_mm) * (H - 1)))))
				rc, gc, bc = [int(v) for v in rgb[yi, xi]]
				# choose nearest palette color index
				best_i = 0
				best_d2 = 1e18
				for i, hx in enumerate(palette_hex):
					pr, pg, pb = _hex_to_rgb(hx)
					dr = pr - rc
					dg = pg - gc
					db = pb - bc
					d2 = dr * dr + dg * dg + db * db
					if d2 < best_d2:
						best_d2 = d2
						best_i = i
				per_color_points[best_i].append((x, y, radius / (0.5 * dot_mm if dot_mm > 0 else 1.0)))
			for i, hx in enumerate(palette_hex):
				layers.append((f"poisson_{i+1}_{hx.lstrip('#')}", hx, per_color_points[i], 0.5 * dot_mm))
			per_point_layers = None
	else:
		if args.mode == "grayscale":
			gray = _to_grayscale_luma(rgb)
			if args.tone_stacks and args.tone_stacks > 1:
				# Stochastic tone stacking: distribute grayscale darkness over multiple stacks
				# Use darkness as density: 1 - gray/255
				darkness = 1.0 - (gray.astype(np.float32) / 255.0)
				v = darkness * float(args.tone_stacks)
				m = np.floor(v).astype(np.int16)
				r = v - m.astype(np.float32)
				residual = np.clip(np.round(r * 255.0), 0, 255).astype(np.uint8)
				residual_mask = blue_noise_dither(residual) > 0
				for k in range(int(args.tone_stacks)):
					mask_k = (m > k) | ((m == k) & residual_mask)
					points = _extract_points_from_mask(mask_k, args.step_mm, density=None)
					layer_name = f"black_s{k+1}"
					layers.append((layer_name, palette_hex[0], points, 0.5 * dot_mm))
				print("[info] tone-stacks enabled: ignoring --variable-dots and --jitter-mm for grayscale mode")
			else:
				# Compute density map if variable dots enabled
				density = compute_local_density(gray, kernel_size=5) if args.variable_dots else None
				# Apply selected dithering method
				mask = dither(gray, method=args.dither_method) > 0
				points = _extract_points_from_mask(mask, args.step_mm, density=density)
				points = _apply_jitter(points)
				layers.append(("black", palette_hex[0], points, 0.5 * dot_mm))
		else:
			# Overlapping color layers: each color gets its own dithered contribution
			for pi, hx in enumerate(palette_hex):
				target_rgb = _hex_to_rgb(hx)
				# Compute how much this color contributes at each pixel
				contribution = _compute_color_contribution(rgb, target_rgb, threshold=args.threshold, white_threshold=args.white_threshold)
				if args.tone_stacks and args.tone_stacks > 1:
					# Stochastic tone stacking per color
					D = contribution.astype(np.float32) / 255.0
					v = D * float(args.tone_stacks)
					m = np.floor(v).astype(np.int16)
					r = v - m.astype(np.float32)
					residual = np.clip(np.round(r * 255.0), 0, 255).astype(np.uint8)
					residual_mask = blue_noise_dither(residual) > 0
					for k in range(int(args.tone_stacks)):
						mask_k = (m > k) | ((m == k) & residual_mask)
						points = _extract_points_from_mask(mask_k, args.step_mm, density=None)
						layer_name = f"color_{pi+1}_s{k+1}_{hx.lstrip('#')}"
						layers.append((layer_name, hx, points, 0.5 * dot_mm))
				else:
					# Compute density map if variable dots enabled
					density = compute_local_density(contribution, kernel_size=5) if args.variable_dots else None
					# Apply selected dithering method
					mask = dither(contribution, method=args.dither_method) > 0
					points = _extract_points_from_mask(mask, args.step_mm, density=density)
					points = _apply_jitter(points)
					layer_name = f"color_{pi+1}_{hx.lstrip('#')}"
					layers.append((layer_name, hx, points, 0.5 * dot_mm))

	# Ensure output directory exists
	os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
	# Choose fill mode and opacity settings
	effective_fill_mode = ("rgba" if (args.tone_stacks and args.tone_stacks > 1) else ("rgba" if args.fill_mode == "rgba" else "opacity"))
	effective_opacity_min = (args.stack_alpha if (args.tone_stacks and args.tone_stacks > 1) else args.opacity_min)
	effective_opacity_max = (args.stack_alpha if (args.tone_stacks and args.tone_stacks > 1) else args.opacity_max)

	write_svg_circles(
		layers=layers,
		width_mm=width_mm,
		height_mm=height_mm,
		order=args.order,
		output_path=args.output,
		opacity_min=effective_opacity_min,
		opacity_max=effective_opacity_max,
		fill_mode=effective_fill_mode,
		per_point_rgba=per_point_layers,
	)
	total_dots = sum(len(p) for _, _, p, _ in layers)
	print(f"Wrote {args.output}  ({cols}x{rows} grid, {total_dots} dots, method={args.dither_method}, variable_dots={args.variable_dots})")

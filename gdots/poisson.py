import math
from typing import Callable, List, Optional, Tuple

import numpy as np


def _grid_index(x: float, y: float, cell_size: float) -> Tuple[int, int]:
	return int(x // cell_size), int(y // cell_size)


def poisson_disk_variable(
	width: float,
	height: float,
	r_base: float,
	r_scale_fn: Callable[[float, float], float],
	*,
	min_scale: float = 0.6,
	k: int = 30,
	seed: Optional[int] = None,
) -> List[Tuple[float, float, float]]:
	"""
	Bridson-style Poisson disk sampling with spatially varying radius r(x,y) = r_base * r_scale_fn(x,y).
	Returns a list of (x_mm, y_mm, r_local_mm) tuples.

	Approximations:
	- Background grid cell size uses the minimum radius (r_base * min_scale) for neighbor lookup.
	- Acceptance ensures distance to neighbors >= max(r_local, r_neighbor).
	"""
	rng = np.random.default_rng(seed)
	# Use the smallest possible radius to size the background grid
	r_min = max(1e-6, r_base * min_scale)
	cell_size = r_min / math.sqrt(2.0)
	grid_w = int(math.ceil(width / cell_size))
	grid_h = int(math.ceil(height / cell_size))
	grid: List[List[Optional[int]]] = [[None for _ in range(grid_h)] for _ in range(grid_w)]
	points: List[Tuple[float, float, float]] = []
	active: List[int] = []

	def r_at(x: float, y: float) -> float:
		s = max(0.05, float(r_scale_fn(x, y)))
		return r_base * s

	def fits(x: float, y: float, r_local: float) -> bool:
		if x < 0 or y < 0 or x >= width or y >= height:
			return False
		gx, gy = _grid_index(x, y, cell_size)
		# neighborhood search around (gx, gy)
		ng = 2  # search +/- 2 cells (conservative)
		for ix in range(max(0, gx - ng), min(grid_w, gx + ng + 1)):
			for iy in range(max(0, gy - ng), min(grid_h, gy + ng + 1)):
				pi = grid[ix][iy]
				if pi is None:
					continue
				px, py, pr = points[pi]
				dx = px - x
				dy = py - y
				thr = max(r_local, pr)
				if dx * dx + dy * dy < thr * thr:
					return False
		return True

	# Initial point
	for _ in range(100):
		x0 = float(rng.uniform(0.0, width))
		y0 = float(rng.uniform(0.0, height))
		r0 = r_at(x0, y0)
		if fits(x0, y0, r0):
			points.append((x0, y0, r0))
			gx, gy = _grid_index(x0, y0, cell_size)
			grid[gx][gy] = 0
			active.append(0)
			break
	if not active:
		return points

	while active:
		idx = int(rng.integers(0, len(active)))
		pi = active[idx]
		px, py, pr = points[pi]
		accepted = False
		# Generate up to k candidates in the annulus [pr, 2*pr]
		for _ in range(k):
			rr = float(rng.uniform(pr, 2.0 * pr))
			ang = float(rng.uniform(0.0, 2.0 * math.pi))
			x = px + rr * math.cos(ang)
			y = py + rr * math.sin(ang)
			r_loc = r_at(x, y)
			if fits(x, y, r_loc):
				points.append((x, y, r_loc))
				gx, gy = _grid_index(x, y, cell_size)
				grid[gx][gy] = len(points) - 1
				active.append(len(points) - 1)
				accepted = True
				break
		if not accepted:
			# remove from active list
			active.pop(idx)

	return points

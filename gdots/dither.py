import numpy as np
from typing import Tuple


def floyd_steinberg_dither(grayscale: np.ndarray) -> np.ndarray:
	"""
	Binary Floyd–Steinberg dithering on a grayscale image (uint8 0..255).
	Returns uint8 array of 0 or 255 with same HxW.
	"""
	if grayscale.dtype != np.uint8:
		raise TypeError("grayscale must be uint8")
	arr = grayscale.astype(np.float32).copy()
	h, w = arr.shape
	for y in range(h):
		for x in range(w):
			old = arr[y, x]
			new = 255.0 if old >= 128.0 else 0.0
			err = old - new
			arr[y, x] = new
			# Distribute error to neighbors
			if x + 1 < w:
				arr[y, x + 1] += err * (7.0 / 16.0)
			if y + 1 < h and x > 0:
				arr[y + 1, x - 1] += err * (3.0 / 16.0)
			if y + 1 < h:
				arr[y + 1, x] += err * (5.0 / 16.0)
			if y + 1 < h and x + 1 < w:
				arr[y + 1, x + 1] += err * (1.0 / 16.0)
	return np.clip(arr, 0, 255).astype(np.uint8)


def _generate_blue_noise_mask(size: int = 64, seed: int = 42) -> np.ndarray:
	"""
	Generate a blue noise threshold mask using void-and-cluster method.
	Returns a mask with values 0-255.
	"""
	np.random.seed(seed)
	mask = np.zeros((size, size), dtype=np.float32)
	
	# Initial random binary pattern
	initial = np.random.rand(size, size) > 0.5
	
	# Create blue noise through iterative filtering
	# Simplified approach: use distance transform on random points
	indices = np.argwhere(initial)
	if len(indices) == 0:
		indices = np.array([[size//2, size//2]])
	
	# Create distance-based noise
	y, x = np.mgrid[0:size, 0:size]
	for i, (py, px) in enumerate(indices[:min(len(indices), 200)]):
		dist = np.sqrt((x - px)**2 + (y - py)**2)
		mask += (1.0 / (dist + 1.0)) * (i + 1)
	
	# Normalize to 0-255
	mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8) * 255.0
	return mask.astype(np.uint8)


# Cache blue noise mask
_BLUE_NOISE_CACHE = None


def blue_noise_dither(grayscale: np.ndarray) -> np.ndarray:
	"""
	Blue noise (stochastic) dithering using a precomputed threshold mask.
	More natural-looking than ordered dithering, reduces visible patterns.
	Returns uint8 array of 0 or 255 with same HxW.
	"""
	global _BLUE_NOISE_CACHE
	if _BLUE_NOISE_CACHE is None:
		_BLUE_NOISE_CACHE = _generate_blue_noise_mask(64)
	
	if grayscale.dtype != np.uint8:
		raise TypeError("grayscale must be uint8")
	
	h, w = grayscale.shape
	mask = _BLUE_NOISE_CACHE
	mh, mw = mask.shape
	
	# Tile the mask to cover the image
	threshold = np.tile(mask, (h // mh + 1, w // mw + 1))[:h, :w]
	
	# Apply threshold
	result = np.where(grayscale > threshold, 255, 0).astype(np.uint8)
	return result


def ordered_dither(grayscale: np.ndarray, matrix_size: int = 8) -> np.ndarray:
	"""
	Ordered (Bayer) dithering using a threshold matrix.
	Creates a regular pattern but with good distribution.
	Returns uint8 array of 0 or 255 with same HxW.
	"""
	if grayscale.dtype != np.uint8:
		raise TypeError("grayscale must be uint8")
	
	# Generate Bayer matrix
	def bayer_matrix(n: int) -> np.ndarray:
		if n == 1:
			return np.array([[0]], dtype=np.float32)
		else:
			smaller = bayer_matrix(n // 2)
			return np.block([
				[4 * smaller + 0, 4 * smaller + 2],
				[4 * smaller + 3, 4 * smaller + 1]
			])
	
	# Use closest power of 2
	size = 2 ** int(np.log2(matrix_size))
	if size < 2:
		size = 2
	
	bayer = bayer_matrix(size)
	# Normalize to 0-255
	bayer = (bayer / (size * size) * 255.0).astype(np.float32)
	
	h, w = grayscale.shape
	# Tile the matrix
	threshold = np.tile(bayer, (h // size + 1, w // size + 1))[:h, :w]
	
	# Apply threshold
	result = np.where(grayscale > threshold, 255, 0).astype(np.uint8)
	return result


def white_noise_dither(grayscale: np.ndarray, seed: int = None) -> np.ndarray:
	"""
	Simple white noise (random threshold) dithering.
	Very stochastic but can look grainy. Good for artistic effects.
	Returns uint8 array of 0 or 255 with same HxW.
	"""
	if grayscale.dtype != np.uint8:
		raise TypeError("grayscale must be uint8")
	
	if seed is not None:
		np.random.seed(seed)
	
	h, w = grayscale.shape
	threshold = np.random.randint(0, 256, size=(h, w), dtype=np.uint8)
	result = np.where(grayscale > threshold, 255, 0).astype(np.uint8)
	return result


def dither(grayscale: np.ndarray, method: str = "floyd-steinberg") -> np.ndarray:
	"""
	Apply dithering to a grayscale image using the specified method.
	
	Args:
		grayscale: uint8 array (H, W) with values 0-255
		method: one of ["floyd-steinberg", "blue-noise", "ordered", "white-noise"]
	
	Returns:
		Binary uint8 array (0 or 255) with same shape
	"""
	method = method.lower()
	if method == "floyd-steinberg":
		return floyd_steinberg_dither(grayscale)
	elif method == "blue-noise":
		return blue_noise_dither(grayscale)
	elif method == "ordered":
		return ordered_dither(grayscale)
	elif method == "white-noise":
		return white_noise_dither(grayscale)
	else:
		raise ValueError(f"Unknown dither method: {method}. Choose from: floyd-steinberg, blue-noise, ordered, white-noise")


def compute_local_density(grayscale: np.ndarray, kernel_size: int = 3) -> np.ndarray:
	"""
	Compute local average density for variable dot sizing.
	Returns float32 array (H, W) with values 0-1 representing local darkness.
	"""
	from scipy.ndimage import uniform_filter
	# Normalize to 0-1 (darker = higher value)
	normalized = 1.0 - (grayscale.astype(np.float32) / 255.0)
	# Apply local averaging
	local_avg = uniform_filter(normalized, size=kernel_size, mode='constant')
	return local_avg.astype(np.float32)

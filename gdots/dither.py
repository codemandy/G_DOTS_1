import numpy as np


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

G-DOTS: Raster to Pen-Plotter SVG (Dots)

Convert PNG/JPG into pen-plotter-optimized SVG using dithering and dot placement, with real-world millimeter scaling suitable for AxiDraw and similar plotters.

Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Usage

```bash
# Grayscale mode (single color)
python -m gdots input.jpg --width-mm 210 --step-mm 0.6 \
  --mode grayscale --palette #000 --order nearest -o out.svg

# Multi-color watercolor mode (overlapping layers with white space)
python -m gdots input.jpg --width-mm 150 --step-mm 0.7 \
  --mode palette --palette "#d4a373,#87ceeb,#9370db" \
  --threshold 0.45 --white-threshold 0.8 --order nearest -o out.svg
```

**Parameters:**
- `--width-mm`: output width in mm (height keeps aspect unless `--height-mm` specified)
- `--step-mm`: grid spacing in mm; smaller yields more points
- `--dot-mm`: dot diameter in mm (default 0.8× step)
- `--mode`: `grayscale` for one-ink dithering, `palette` for multi-ink overlapping color layers
- `--palette`: comma-separated hex colors like `#000,#f00,#0af`
- `--threshold`: 0-1, controls color selectivity (lower = more overlap, ~0.4-0.5 recommended)
- `--white-threshold`: 0-1, preserves bright areas (0.8-0.85 keeps whites clean like watercolor paper)
- `--order`: `nearest` to reduce travel; `none` keeps raster order

Notes

- Uses Floyd–Steinberg for binary dithering on each layer independently
- **Palette mode creates overlapping layers**: each color is dithered based on its contribution/presence in the image, allowing colors to blend naturally
- Each color is in its own SVG `<g>` layer group with ID like `color_1_7b3f00` for easy selection in Inkscape/AxiDraw
- Circles are exported as stroked outlines with stroke set to the ink color. Adjust stroke-width logic if your plotter/ink needs heavier or lighter coverage
- For watercolor/ink washes: plot light/transparent colors first; proceed to darker inks; layers will overlay naturally

Roadmap

- Blue-noise / void-and-cluster dithering
- Variable dot sizes from tone
- Path optimization across layers
- GUI preview

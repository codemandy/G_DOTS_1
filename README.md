G-DOTS: Raster to Pen-Plotter SVG (Dots)

Convert PNG/JPG into pen-plotter-optimized SVG using dithering and dot placement, with real-world millimeter scaling suitable for AxiDraw and similar plotters.

Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Usage

```bash
# Stage 1: Grayscale mode (single color, Floyd-Steinberg)
python -m gdots input.jpg --width-mm 210 --step-mm 0.6 \
  --mode grayscale --palette #000 --order nearest -o out.svg

# Stage 2: Blue noise with variable dots (stochastic screening)
python -m gdots input.jpg --width-mm 150 --step-mm 0.7 \
  --mode grayscale --dither-method blue-noise --variable-dots -o out.svg

# Stage 2: Multi-color watercolor with blue noise
python -m gdots input.jpg --width-mm 150 --step-mm 0.7 \
  --mode palette --palette "#d4a373,#87ceeb,#9370db" \
  --dither-method blue-noise --variable-dots \
  --threshold 0.45 --white-threshold 0.8 -o out.svg
```

**Parameters:**
- `--width-mm`: output width in mm (height keeps aspect unless `--height-mm` specified)
- `--step-mm`: grid spacing in mm; smaller yields more points
- `--dot-mm`: dot diameter in mm (default 0.8× step)
- `--mode`: `grayscale` for one-ink dithering, `palette` for multi-ink overlapping color layers
- `--palette`: comma-separated hex colors like `#000,#f00,#0af`
- `--dither-method`: dithering algorithm - `floyd-steinberg` (default), `blue-noise` (stochastic), `ordered` (Bayer), `white-noise` (random)
- `--variable-dots`: enable variable dot sizes based on local tone (darker areas = larger dots)
- `--threshold`: 0-1, controls color selectivity (lower = more overlap, ~0.4-0.5 recommended)
- `--white-threshold`: 0-1, preserves bright areas (0.8-0.85 keeps whites clean like watercolor paper)
- `--order`: `nearest` to reduce travel; `none` keeps raster order

## Features

### Stage 1 (Complete)
- ✅ Floyd-Steinberg error diffusion dithering
- ✅ Multi-color palette mode with overlapping layers
- ✅ White space preservation for watercolor effects
- ✅ Nearest-neighbor path optimization
- ✅ Inkscape-compatible layer groups

### Stage 2 (Complete)
- ✅ **Blue noise (stochastic) dithering** - more natural, random appearance
- ✅ **Ordered (Bayer) dithering** - regular pattern distribution
- ✅ **White noise dithering** - random threshold for artistic effects
- ✅ **Variable dot sizes** - darker areas use larger dots for better tone representation

## Notes

- **Dithering algorithms**: Choose based on desired aesthetic
  - `floyd-steinberg`: Best for detail and edges, some directionality
  - `blue-noise`: Natural stochastic screening, no visible patterns (recommended for photos)
  - `ordered`: Regular Bayer matrix pattern, fast and predictable
  - `white-noise`: Random/grainy, good for artistic effects
- **Variable dots**: Enable `--variable-dots` to modulate dot size by local tone (0.5x to 1.5x base radius)
- **Palette mode creates overlapping layers**: each color is dithered based on its contribution/presence in the image, allowing colors to blend naturally
- Each color is in its own SVG `<g>` layer group with ID like `color_1_7b3f00` for easy selection in Inkscape/AxiDraw
- Circles are exported as stroked outlines with stroke set to the ink color
- For watercolor/ink washes: plot light/transparent colors first; proceed to darker inks; layers will overlay naturally

## Roadmap (Stage 3+)

- Void-and-cluster algorithm for improved blue noise
- Adaptive dot sizing with multiple size classes
- Path optimization across layers
- GUI preview with live parameter adjustment

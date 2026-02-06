# Using PINGMapper Models on GeoTIFF Images

This guide explains how to use the pretrained PINGMapper substrate classification models on your GeoTIFF files.

## Overview

PINGMapper is designed for processing raw sonar recordings, but the underlying segmentation models (based on SegFormer architecture) can be applied to any grayscale or RGB imagery that resembles side-scan sonar data.

## Prerequisites

### 1. Install PINGMapper Environment

```bash
# Create and activate the PINGMapper conda environment
conda env create -f PINGMapper/pingmapper/conda/PINGMapper.yml
conda activate PINGMapper
```

Or if you already have PINGMapper installed:
```bash
conda activate PINGMapper
```

### 2. Required Packages

The script requires these packages (included in PINGMapper environment):
- tensorflow
- transformers
- doodleverse-utils
- rasterio
- numpy
- scikit-image
- tqdm

## Usage

### Quick Start

1. Edit the configuration section at the top of `classify_tif_substrate.py`:

```python
# Input GeoTIFF file
INPUT_TIF = "/Users/quinnfisher/river_substrate/Grand_River_Habitat_Assessment.tif"

# Output directory for results
OUTPUT_DIR = "/Users/quinnfisher/river_substrate/substrate_output"

# Model type: 'raw' or 'egn'
MODEL_TYPE = 'raw'

# Use GPU if available
USE_GPU = False

# Tile size and overlap
TILE_SIZE = 512
TILE_OVERLAP = 64
```

2. Run the script:

```bash
conda activate PINGMapper
python classify_tif_substrate.py
```

### Model Selection

**`MODEL_TYPE = 'raw'`**: Use for imagery that has NOT been empirically gain normalized. This is the default and works well for most general imagery.

**`MODEL_TYPE = 'egn'`**: Use for imagery that HAS been empirically gain normalized (EGN is a specific correction applied in sonar processing).

## Output Files

The script generates several files in the output directory:

| File | Description |
|------|-------------|
| `substrate_classification.tif` | Single-band GeoTIFF with class IDs (0-6) |
| `substrate_classification_colored.tif` | RGB GeoTIFF with colored classes |
| `substrate_classification_colored.png` | PNG preview (downsampled if large) |
| `probability_X_classname.tif` | Per-class probability maps |
| `class_legend.txt` | Class ID to name mapping |

## Substrate Classes

The PINGMapper v2.0 models classify 7 substrate types:

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | Fines (Smooth) | Fine sediments with smooth texture |
| 1 | Fines (Rough) | Fine sediments with rough texture |
| 2 | Sand/Gravel | Sandy or gravelly substrate |
| 3 | Cobble | Cobble-sized rocks |
| 4 | Hard Bottom | Bedrock or consolidated substrate |
| 5 | Wood | Submerged wood/debris |
| 6 | Other | Unclassified or mixed |

*Note: Actual class names depend on the specific model version.*

## Important Considerations

### Image Compatibility

The models were trained on side-scan sonar imagery with specific characteristics:
- Grayscale intensity patterns
- Water column effects removed
- Shadow patterns typical of sonar

**Your TIF will work best if it:**
- Contains side-scan sonar data (or similar acoustic imagery)
- Has similar intensity/texture patterns to training data
- Is grayscale or can be converted to grayscale

**Results may be less accurate if your image:**
- Is optical/photographic (not acoustic)
- Has very different intensity distributions
- Contains features not present in training data

### Memory Usage

For large images like yours (26312 x 11205 pixels), the script processes in tiles to manage memory. Adjust `TILE_SIZE` and `TILE_OVERLAP` if you encounter memory issues:

```python
TILE_SIZE = 256      # Smaller = less memory
TILE_OVERLAP = 32    # Smaller = faster but more edge artifacts
```

### Processing Time

Processing time depends on:
- Image size
- Tile size
- CPU vs GPU
- Available memory

For your image (~295M pixels), expect 30-60 minutes on CPU.

## Troubleshooting

### "ModuleNotFoundError: No module named 'doodleverse_utils'"
Make sure you're in the PINGMapper conda environment:
```bash
conda activate PINGMapper
```

### "Models not found"
The script will automatically download models from Zenodo (~200MB). Ensure you have internet access on first run.

### Out of Memory
Reduce `TILE_SIZE` to 256 or 128:
```python
TILE_SIZE = 256
TILE_OVERLAP = 32
```

### Poor Classification Results
The models are trained specifically on sonar imagery. If your TIF contains significantly different data, consider:
1. Preprocessing to match sonar intensity patterns
2. Training a custom model using [segmentation_gym](https://github.com/Doodleverse/segmentation_gym)

## Advanced: Direct API Usage

For more control, you can use the model functions directly:

```python
import numpy as np
from classify_tif_substrate import init_model, get_model_paths, predict_tile, load_config

# Load model
model_dir = '/path/to/PINGMapperv2.0_SegmentationModelsv1.0'
weights, config_path = get_model_paths(model_dir, 'raw')
model, config = init_model(weights, config_path, use_gpu=False)

# Compile
from classify_tif_substrate import compile_model
model = compile_model(model, config['MODEL'])

# Predict on a tile (H x W x 3 uint8 array)
tile = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
class_pred, logits = predict_tile(model, tile, config)

print(class_pred.shape)  # (512, 512)
print(logits.shape)      # (512, 512, 7)
```

## References

- [PINGMapper GitHub](https://github.com/CameronBodine/PINGMapper)
- [PINGMapper Documentation](https://cameronbodine.github.io/PINGMapper/)
- [Segmentation Models (Zenodo)](https://doi.org/10.5281/zenodo.10093642)
- [Paper: Automated river substrate mapping from sonar imagery with machine learning](https://doi.org/10.1029/2024JH000135)

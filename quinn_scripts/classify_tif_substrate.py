#!/usr/bin/env python3
"""
Standalone script to classify substrate in a GeoTIFF using PINGMapper's pretrained models.

This script applies the PINGMapper segformer substrate classification model to 
an arbitrary GeoTIFF image, processing it in tiles to handle large files.

Usage:
    python classify_tif_substrate.py

Requirements:
    - PINGMapper conda environment activated
    - PINGMapper segmentation models downloaded (will download automatically if missing)

Author: Based on PINGMapper by Cameron S. Bodine
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================================
# CONFIGURATION - Modify these paths as needed
# ============================================================================

# Input GeoTIFF file
INPUT_TIF = "/Users/quinnfisher/river_substrate/Grand_River_Habitat_Assessment.tif"

# Output directory for results
OUTPUT_DIR = "/Users/quinnfisher/river_substrate/substrate_output"

# Model type: 'raw' or 'egn' (empirical gain normalized)
# Use 'raw' if your image hasn't been gain-normalized
MODEL_TYPE = 'raw'

# Use GPU if available (set to False to force CPU)
USE_GPU = False

# Tile size for processing (should match model's expected input roughly)
# Smaller tiles = less memory, more tiles to process
TILE_SIZE = 512

# Overlap between tiles (helps reduce edge artifacts)
TILE_OVERLAP = 64

# ============================================================================
# IMPORTS (after setting TF log level)
# ============================================================================

try:
    import tensorflow as tf
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
    tf.get_logger().setLevel('ERROR')
except ImportError:
    print("Error: TensorFlow not found. Please activate PINGMapper conda environment.")
    print("Run: conda activate PINGMapper")
    sys.exit(1)

try:
    import rasterio
    from rasterio.windows import Window
    from rasterio.enums import Resampling
except ImportError:
    print("Error: rasterio not found. Please install: pip install rasterio")
    sys.exit(1)

from skimage.transform import resize
from skimage.io import imsave
import requests
import zipfile

# ============================================================================
# MODEL LOADING FUNCTIONS (adapted from PINGMapper funcs_model.py)
# ============================================================================

def download_models(model_dir):
    """Download PINGMapper segmentation models if not present."""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    
    # Check if models already exist (could be in subdir or directly in model_dir)
    model_subdir = os.path.join(model_dir, 'PINGMapperv2.0_SegmentationModelsv1.0')
    
    # Check for model files directly in model_dir (zip extracts without wrapper folder)
    raw_model_direct = os.path.join(model_dir, 'Raw_Substrate_Segmentation_segformer_v1.0')
    if os.path.exists(raw_model_direct):
        print(f"Models already exist at: {model_dir}")
        return model_dir
    
    if os.path.exists(model_subdir):
        print(f"Models already exist at: {model_subdir}")
        return model_subdir
    
    url = 'https://zenodo.org/records/10093642/files/PINGMapperv2.0_SegmentationModelsv1.0.zip?download=1'
    print(f'\nDownloading segmentation models from Zenodo...')
    print(f'URL: {url}')
    
    filename = model_dir + '.zip'
    r = requests.get(url, allow_redirects=True, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading') as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    
    print('Extracting models...')
    with zipfile.ZipFile(filename, 'r') as z_fp:
        z_fp.extractall(model_dir)
    os.remove(filename)
    
    print('Model download complete!')
    
    # Return the actual path where models were extracted
    # The zip extracts directly without the wrapper folder
    if os.path.exists(os.path.join(model_dir, 'Raw_Substrate_Segmentation_segformer_v1.0')):
        return model_dir
    return model_subdir


def get_model_paths(model_base_dir, model_type='raw'):
    """Get paths to model weights and config."""
    if model_type.lower() == 'egn':
        model_name = 'EGN_Substrate_Segmentation_segformer_v1.0'
    else:
        model_name = 'Raw_Substrate_Segmentation_segformer_v1.0'
    
    config_path = os.path.join(model_base_dir, model_name, 'config', f'{model_name}.json')
    weights_path = os.path.join(model_base_dir, model_name, 'weights', f'{model_name}_fullmodel.h5')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    return weights_path, config_path


def load_config(config_path):
    """Load model configuration."""
    with open(config_path) as f:
        config = json.load(f)
    return config


def init_model(weights_path, config_path, use_gpu=False):
    """Initialize the segmentation model."""
    SEED = 42
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    config = load_config(config_path)
    
    MODEL = config.get('MODEL', 'segformer')
    TARGET_SIZE = config.get('TARGET_SIZE', [512, 512])
    NCLASSES = config.get('NCLASSES', 7)
    N_DATA_BANDS = config.get('N_DATA_BANDS', 3)
    FILTERS = config.get('FILTERS', 16)
    KERNEL = config.get('KERNEL', 3)
    STRIDE = config.get('STRIDE', 1)
    DROPOUT = config.get('DROPOUT', 0.1)
    DROPOUT_CHANGE_PER_LAYER = config.get('DROPOUT_CHANGE_PER_LAYER', 0.0)
    DROPOUT_TYPE = config.get('DROPOUT_TYPE', 'standard')
    USE_DROPOUT_ON_UPSAMPLING = config.get('USE_DROPOUT_ON_UPSAMPLING', False)
    MY_CLASS_NAMES = config.get('MY_CLASS_NAMES', {})
    
    print(f"\nModel Configuration:")
    print(f"  Architecture: {MODEL}")
    print(f"  Target Size: {TARGET_SIZE}")
    print(f"  Classes: {NCLASSES}")
    print(f"  Class Names: {list(MY_CLASS_NAMES.values()) if MY_CLASS_NAMES else 'N/A'}")
    
    if MODEL == 'resunet':
        try:
            from doodleverse_utils.model_imports import custom_resunet
        except ImportError:
            raise ImportError("doodleverse_utils required for resunet model. Install with: pip install doodleverse-utils")
        
        model = custom_resunet(
            (TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
            FILTERS,
            nclasses=[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
            kernel_size=(KERNEL, KERNEL),
            strides=STRIDE,
            dropout=DROPOUT,
            dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,
            dropout_type=DROPOUT_TYPE,
            use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,
        )
    elif MODEL == 'segformer':
        from transformers import TFSegformerForSemanticSegmentation
        
        # Create segformer model
        id2label = {k: str(k) for k in range(NCLASSES)}
        label2id = {str(k): k for k in range(NCLASSES)}
        
        model = TFSegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=NCLASSES,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
    else:
        raise ValueError(f"Unknown model type: {MODEL}")
    
    model.load_weights(weights_path)
    
    return model, config


# ============================================================================
# PREDICTION FUNCTIONS (adapted from PINGMapper funcs_model.py)
# ============================================================================

def standardize(img):
    """Standardize image for model input."""
    try:
        from doodleverse_utils.prediction_imports import standardize as dv_standardize
        return dv_standardize(img)
    except ImportError:
        # Fallback: simple standardization
        img = np.array(img, dtype=np.float32)
        img = (img - img.mean()) / (img.std() + 1e-8)
        return img


def compile_model(model, model_type):
    """Compile model for inference."""
    # For inference, we don't need full compilation - just return the model
    # The model already has weights loaded
    if isinstance(model, (list, tuple)):
        return model
    return [model]


def predict_tile(model, tile, config):
    """Run prediction on a single tile."""
    MODEL = config.get('MODEL', 'segformer')
    TARGET_SIZE = config.get('TARGET_SIZE', [512, 512])
    NCLASSES = config.get('NCLASSES', 7)
    N_DATA_BANDS = config.get('N_DATA_BANDS', 3)
    
    # Get the actual model object
    if isinstance(model, (list, tuple)):
        model_obj = model[0]
    else:
        model_obj = model
    
    # Ensure tile is the right shape
    if len(tile.shape) == 2:
        # Grayscale - expand to 3 channels for segformer
        tile = np.stack([tile, tile, tile], axis=-1)
    elif tile.shape[-1] == 1:
        tile = np.concatenate([tile, tile, tile], axis=-1)
    elif tile.shape[-1] > 3:
        tile = tile[:, :, :3]
    
    # Store original size
    orig_h, orig_w = tile.shape[:2]
    
    # Resize to model input size
    tile_resized = resize(tile, (TARGET_SIZE[0], TARGET_SIZE[1]), preserve_range=True, clip=True)
    tile_resized = np.array(tile_resized).astype(np.uint8)
    
    # Standardize
    tile_std = standardize(tile_resized).squeeze()
    
    # Add batch dimension
    if MODEL == 'segformer':
        # Segformer expects (batch, channels, height, width)
        tile_input = np.transpose(tile_std, (2, 0, 1))
        tile_input = np.expand_dims(tile_input, 0)
    else:
        tile_input = np.expand_dims(tile_std, 0)
    
    # Run prediction
    pred = model_obj.predict(tile_input, verbose=0)
    
    # Process output
    if MODEL == 'segformer':
        # Segformer output shape: (batch, num_classes, height, width)
        logits = pred.logits[0]  # Remove batch dimension
        logits = np.transpose(logits, (1, 2, 0))  # (H, W, C)
    else:
        logits = pred[0]
    
    # Resize back to original tile size
    logits_resized = resize(logits, (orig_h, orig_w, NCLASSES), preserve_range=True)
    
    # Get class predictions
    class_pred = np.argmax(logits_resized, axis=-1)
    
    return class_pred, logits_resized


# ============================================================================
# TILED PROCESSING
# ============================================================================

def process_tif_tiled(input_path, output_dir, model, config, tile_size=512, overlap=64):
    """Process a large GeoTIFF in tiles."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get class names from config
    class_names = config.get('MY_CLASS_NAMES', {})
    if class_names:
        class_list = [class_names.get(str(i), f'class_{i}') for i in range(len(class_names))]
    else:
        class_list = [f'class_{i}' for i in range(config.get('NCLASSES', 7))]
    
    print(f"\nSubstrate Classes:")
    for i, name in enumerate(class_list):
        print(f"  {i}: {name}")
    
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        width = src.width
        height = src.height
        
        print(f"\nInput image: {width} x {height} pixels")
        print(f"Tile size: {tile_size}, Overlap: {overlap}")
        
        # Calculate number of tiles
        step = tile_size - overlap
        n_tiles_x = int(np.ceil((width - overlap) / step))
        n_tiles_y = int(np.ceil((height - overlap) / step))
        total_tiles = n_tiles_x * n_tiles_y
        
        print(f"Processing {total_tiles} tiles ({n_tiles_x} x {n_tiles_y})")
        
        # Initialize output arrays
        class_output = np.zeros((height, width), dtype=np.uint8)
        count_output = np.zeros((height, width), dtype=np.float32)
        
        # Accumulate logits for averaging in overlap regions
        nclasses = config.get('NCLASSES', 7)
        logits_sum = np.zeros((height, width, nclasses), dtype=np.float32)
        
        # Process tiles
        tile_idx = 0
        with tqdm(total=total_tiles, desc='Processing tiles') as pbar:
            for y in range(0, height, step):
                for x in range(0, width, step):
                    # Calculate window
                    win_width = min(tile_size, width - x)
                    win_height = min(tile_size, height - y)
                    
                    # Read tile
                    window = Window(x, y, win_width, win_height)
                    tile_data = src.read(window=window)
                    
                    # Convert from (bands, height, width) to (height, width, bands)
                    tile_data = np.transpose(tile_data, (1, 2, 0))
                    
                    # Handle different band configurations
                    if tile_data.shape[-1] == 1:
                        # Single band - use as grayscale
                        tile_rgb = np.squeeze(tile_data)
                    elif tile_data.shape[-1] >= 3:
                        # Use first 3 bands as RGB
                        tile_rgb = tile_data[:, :, :3]
                    else:
                        # 2 bands - duplicate first band
                        tile_rgb = np.stack([tile_data[:,:,0]]*3, axis=-1)
                    
                    # Pad if tile is smaller than expected
                    if win_height < tile_size or win_width < tile_size:
                        padded = np.zeros((tile_size, tile_size, 3), dtype=tile_rgb.dtype)
                        padded[:win_height, :win_width, :] = tile_rgb
                        tile_rgb = padded
                    
                    # Run prediction
                    class_pred, logits = predict_tile(model, tile_rgb, config)
                    
                    # Crop to actual window size
                    class_pred = class_pred[:win_height, :win_width]
                    logits = logits[:win_height, :win_width, :]
                    
                    # Accumulate results (for overlap averaging)
                    logits_sum[y:y+win_height, x:x+win_width, :] += logits
                    count_output[y:y+win_height, x:x+win_width] += 1
                    
                    tile_idx += 1
                    pbar.update(1)
        
        # Average overlapping regions
        count_output[count_output == 0] = 1  # Avoid division by zero
        for c in range(nclasses):
            logits_sum[:, :, c] /= count_output
        
        # Final classification
        class_output = np.argmax(logits_sum, axis=-1).astype(np.uint8)
        
        # Save classification output
        out_profile = profile.copy()
        out_profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress='lzw'
        )
        
        output_class_path = os.path.join(output_dir, 'substrate_classification.tif')
        with rasterio.open(output_class_path, 'w', **out_profile) as dst:
            dst.write(class_output, 1)
        
        print(f"\nClassification saved to: {output_class_path}")
        
        # Save probability/logit outputs for each class
        print("\nSaving per-class probability maps...")
        for c in range(nclasses):
            class_name = class_list[c] if c < len(class_list) else f'class_{c}'
            out_prob_profile = profile.copy()
            out_prob_profile.update(
                dtype=rasterio.float32,
                count=1,
                compress='lzw'
            )
            
            prob_path = os.path.join(output_dir, f'probability_{c}_{class_name}.tif')
            
            # Convert logits to probabilities
            probs = tf.nn.softmax(logits_sum, axis=-1).numpy()
            
            with rasterio.open(prob_path, 'w', **out_prob_profile) as dst:
                dst.write(probs[:, :, c].astype(np.float32), 1)
        
        # Create a colored classification map
        create_colored_map(class_output, class_list, output_dir, profile)
        
        # Save class legend
        save_class_legend(class_list, output_dir)
        
        return class_output, class_list


def create_colored_map(class_output, class_list, output_dir, profile):
    """Create a colored RGB classification map."""
    
    # Color palette (same as PINGMapper uses)
    colors = [
        [51, 102, 204],    # Blue
        [220, 57, 18],     # Red
        [255, 153, 0],     # Orange
        [16, 150, 24],     # Green
        [153, 0, 153],     # Purple
        [0, 153, 198],     # Cyan
        [221, 68, 119],    # Pink
        [102, 170, 0],     # Lime
        [184, 46, 46],     # Dark Red
        [49, 99, 149],     # Dark Blue
        [0, 0, 0],         # Black
    ]
    
    height, width = class_output.shape
    rgb_output = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(len(class_list)):
        mask = class_output == i
        color = colors[i % len(colors)]
        for c in range(3):
            rgb_output[:, :, c][mask] = color[c]
    
    # Save as PNG (for quick viewing)
    png_path = os.path.join(output_dir, 'substrate_classification_colored.png')
    
    # Downsample if very large
    if height > 4096 or width > 4096:
        scale = min(4096/height, 4096/width)
        new_h, new_w = int(height * scale), int(width * scale)
        rgb_small = resize(rgb_output, (new_h, new_w, 3), preserve_range=True).astype(np.uint8)
        imsave(png_path, rgb_small)
        print(f"Colored map (downsampled) saved to: {png_path}")
    else:
        imsave(png_path, rgb_output)
        print(f"Colored map saved to: {png_path}")
    
    # Also save as GeoTIFF
    out_profile = profile.copy()
    out_profile.update(
        dtype=rasterio.uint8,
        count=3,
        compress='lzw'
    )
    
    geotiff_path = os.path.join(output_dir, 'substrate_classification_colored.tif')
    with rasterio.open(geotiff_path, 'w', **out_profile) as dst:
        for i in range(3):
            dst.write(rgb_output[:, :, i], i + 1)
    print(f"Colored GeoTIFF saved to: {geotiff_path}")


def save_class_legend(class_list, output_dir):
    """Save a text file with class legend."""
    legend_path = os.path.join(output_dir, 'class_legend.txt')
    
    colors = [
        '#3366CC', '#DC3912', '#FF9900', '#109618', '#990099',
        '#0099C6', '#DD4477', '#66AA00', '#B82E2E', '#316395', '#000000'
    ]
    
    with open(legend_path, 'w') as f:
        f.write("Substrate Classification Legend\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{'Class ID':<10} {'Class Name':<25} {'Color':<10}\n")
        f.write("-" * 50 + "\n")
        for i, name in enumerate(class_list):
            color = colors[i % len(colors)]
            f.write(f"{i:<10} {name:<25} {color:<10}\n")
    
    print(f"Class legend saved to: {legend_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("PINGMapper Substrate Classification for GeoTIFF")
    print("=" * 60)
    
    # Check input file exists
    if not os.path.exists(INPUT_TIF):
        print(f"Error: Input file not found: {INPUT_TIF}")
        sys.exit(1)
    
    print(f"\nInput file: {INPUT_TIF}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Model type: {MODEL_TYPE}")
    
    # Find or download models
    # Try common locations for model directory
    possible_model_dirs = [
        os.path.join(os.environ.get('CONDA_PREFIX', ''), 'pingmapper_config', 'models', 'PINGMapperv2.0_SegmentationModelsv1.0'),
        os.path.join(os.environ.get('CONDA_PREFIX', ''), 'pingmapper_config', 'models'),
        os.path.expanduser('~/.pingmapper/models'),
        os.path.join(os.path.dirname(__file__), 'models'),
        '/Users/quinnfisher/river_substrate/models',
    ]
    
    model_base_dir = None
    for d in possible_model_dirs:
        # Check for model subdirectory directly
        raw_model = os.path.join(d, 'Raw_Substrate_Segmentation_segformer_v1.0')
        if os.path.exists(raw_model):
            model_base_dir = d
            break
        # Also check for wrapper subdirectory
        model_subdir = os.path.join(d, 'PINGMapperv2.0_SegmentationModelsv1.0')
        raw_model_nested = os.path.join(model_subdir, 'Raw_Substrate_Segmentation_segformer_v1.0')
        if os.path.exists(raw_model_nested):
            model_base_dir = model_subdir
            break
    
    if model_base_dir is None:
        # Download models to local directory
        print("\nModels not found locally. Downloading...")
        model_dir = os.path.join(os.path.dirname(OUTPUT_DIR), 'models')
        model_base_dir = download_models(model_dir)
    
    print(f"\nUsing models from: {model_base_dir}")
    
    # Get model paths
    try:
        weights_path, config_path = get_model_paths(model_base_dir, MODEL_TYPE)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"Config: {config_path}")
    print(f"Weights: {weights_path}")
    
    # Load model
    print("\nLoading model...")
    model, config = init_model(weights_path, config_path, USE_GPU)
    
    # Compile model
    MODEL = config.get('MODEL', 'segformer')
    model = compile_model(model, MODEL)
    
    # Process the TIF
    print("\nStarting classification...")
    class_output, class_list = process_tif_tiled(
        INPUT_TIF, 
        OUTPUT_DIR, 
        model, 
        config,
        tile_size=TILE_SIZE,
        overlap=TILE_OVERLAP
    )
    
    print("\n" + "=" * 60)
    print("Classification complete!")
    print("=" * 60)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("\nFiles created:")
    for f in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, f)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  - {f} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()

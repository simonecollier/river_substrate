#!/bin/bash
# Script to install required dependencies for TIF classification

VENV_PATH="/Users/quinnfisher/ontario/lake_ontario/code/salmonid_tracking/myenv312/bin/activate"

echo "Activating virtual environment..."
source "$VENV_PATH"

echo "Installing required packages..."
echo ""

# Check what's needed
echo "Checking current installations..."
pip list | grep -E "(rasterio|gdal|tensorflow|transformers|doodleverse)"

echo ""
echo "Installing missing packages..."
echo ""

# Install rasterio (for reading/writing GeoTIFFs)
echo "Installing rasterio..."
pip install rasterio

# Install TensorFlow
echo "Installing TensorFlow..."
pip install tensorflow

# Install transformers (for SegFormer models)
echo "Installing transformers..."
pip install transformers

# Install doodleverse-utils (required by PINGMapper)
echo "Installing doodleverse-utils..."
pip install doodleverse-utils

# Install other dependencies that might be needed
echo "Installing other dependencies..."
pip install scikit-image geopandas pyproj

echo ""
echo "Checking for GDAL (osgeo)..."
python -c "from osgeo import gdal; print('✓ GDAL is installed')" 2>/dev/null || {
    echo ""
    echo "⚠ WARNING: GDAL (osgeo) is not installed!"
    echo ""
    echo "GDAL is required by PINGMapper. To install it:"
    echo ""
    echo "Option 1 - Using conda (recommended, easiest):"
    echo "  conda install -c conda-forge gdal"
    echo ""
    echo "Option 2 - Using Homebrew + pip (requires Xcode):"
    echo "  brew install gdal"
    echo "  pip install gdal"
    echo ""
    echo "Option 3 - Try pre-built wheel:"
    echo "  pip install --find-links https://girder.github.io/large_image_wheels GDAL"
    echo ""
    echo "After installing GDAL, run this script again or try running the classification."
}

echo ""
echo "Installation complete!"
echo ""
echo "You can now run:"
echo "  ./run_classify.sh Grand_River_Habitat_Assessment.tif"


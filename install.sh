#!/bin/bash
# Installation script for Linux/Mac with CUDA support and virtual environment

echo "==============================================="
echo "SLAM Dataset Annotation Pipeline Installation"
echo "==============================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "Python found:"
python3 --version
echo ""

# Check if already in a virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Already in virtual environment: $VIRTUAL_ENV"
    echo "Proceeding with installation in current environment..."
    echo ""
elif [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Already in conda environment: $CONDA_DEFAULT_ENV"
    echo "Proceeding with installation in current environment..."
    echo ""
else
    # Ask if user wants to create a virtual environment
    echo "Not currently in a virtual environment."
    echo ""
    read -p "Create a new virtual environment? (y/n): " create_venv

    if [ "$create_venv" = "y" ] || [ "$create_venv" = "Y" ]; then
        echo ""
        echo "Creating virtual environment 'venv'..."
        python3 -m venv venv

        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to create virtual environment"
            echo "Make sure you have venv module installed"
            exit 1
        fi

        echo "Activating virtual environment..."
        source venv/bin/activate

        echo ""
        echo "Virtual environment created and activated!"
        echo ""
    else
        echo ""
        echo "WARNING: Installing packages globally."
        echo "It's recommended to use a virtual environment."
        echo ""
    fi
fi

# Ask user for CUDA version
echo "Select PyTorch installation option:"
echo "[1] CUDA 11.8 - RTX 30xx, RTX 40xx, etc."
echo "[2] CUDA 12.1 - Latest GPUs"
echo "[3] CPU only - No GPU"
echo ""
read -p "Enter choice (1-3): " cuda_choice

echo ""

case $cuda_choice in
    1)
        echo "Installing PyTorch with CUDA 11.8 support..."
        pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        ;;
    2)
        echo "Installing PyTorch with CUDA 12.1 support..."
        pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        ;;
    3)
        echo "Installing PyTorch CPU only..."
        pip3 install torch torchvision
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "Installing other dependencies..."
echo ""

# Install from requirements.txt (excluding torch which was already installed)
pip3 install "numpy>=1.24.0"
pip3 install "scipy>=1.10.0"
pip3 install "scikit-learn>=1.2.0"
pip3 install "open3d>=0.17.0"
pip3 install "plyfile>=0.7.4"
pip3 install "pandas>=2.0.0"
pip3 install "h5py>=3.8.0"
pip3 install "pyyaml>=6.0"
pip3 install "matplotlib>=3.7.0"
pip3 install "seaborn>=0.12.0"
pip3 install "tqdm>=4.65.0"
pip3 install "transformers>=4.30.0"
pip3 install "timm>=0.9.0"
pip3 install "einops>=0.6.1"
pip3 install "requests>=2.31.0"
pip3 install "pillow>=9.5.0"
pip3 install "opencv-python>=4.8.0"

echo ""
echo "==============================================="
echo "Installation complete!"
echo "==============================================="
echo ""

if [ -n "$VIRTUAL_ENV" ]; then
    echo "Virtual environment is active at: $VIRTUAL_ENV"
    echo ""
    echo "To activate this environment in the future, run:"
    echo "  source venv/bin/activate"
    echo ""
    echo "To deactivate, run:"
    echo "  deactivate"
    echo ""
fi

echo "Next steps:"
echo "1. Test installation: python3 test_installation.py"
echo "2. Run quick demo:     python3 quick_start.py"
echo "3. Download KITTI:     python3 download_kitti.py"
echo "4. Run full pipeline:  python3 demo.py"
echo ""

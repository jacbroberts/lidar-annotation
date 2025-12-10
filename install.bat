@echo off
REM Installation script for Windows with CUDA support and virtual environment

echo ===============================================
echo SLAM Dataset Annotation Pipeline Installation
echo ===============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

REM Check Python version and warn about 3.13
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Detected Python version: %PYTHON_VERSION%

echo %PYTHON_VERSION% | findstr /C:"3.13" >nul
if not errorlevel 1 (
    echo.
    echo ============================================================
    echo WARNING: Python 3.13 is not yet supported by PyTorch!
    echo ============================================================
    echo.
    echo PyTorch currently supports Python 3.8 - 3.12 only.
    echo.
    echo Please use one of these options:
    echo   1. Install Python 3.11 or 3.12 from python.org
    echo   2. Use: py -3.11 -m venv venv  ^(if you have 3.11 installed^)
    echo   3. Use: py -3.12 -m venv venv  ^(if you have 3.12 installed^)
    echo.
    echo To check available Python versions:
    echo   py --list
    echo.
    pause
    exit /b 1
)

echo.

REM Check if already in a virtual environment
if defined VIRTUAL_ENV (
    echo Already in virtual environment: %VIRTUAL_ENV%
    echo Proceeding with installation in current environment...
    echo.
    goto :install_packages
)

if defined CONDA_DEFAULT_ENV (
    echo Already in conda environment: %CONDA_DEFAULT_ENV%
    echo Proceeding with installation in current environment...
    echo.
    goto :install_packages
)

REM Ask if user wants to create a virtual environment
echo Not currently in a virtual environment.
echo.
set /p create_venv="Create a new virtual environment? (y/n): "

if /i "%create_venv%"=="y" (
    echo.
    echo Creating virtual environment 'venv'...
    python -m venv venv

    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        echo Make sure you have venv module installed
        pause
        exit /b 1
    )

    echo Activating virtual environment...
    call venv\Scripts\activate.bat

    echo.
    echo Virtual environment created and activated!
    echo.
) else (
    echo.
    echo WARNING: Installing packages globally.
    echo It's recommended to use a virtual environment.
    echo.
)

:install_packages

REM Ask user for CUDA version
echo Select PyTorch installation option:
echo [1] CUDA 11.8 - RTX 30xx, RTX 40xx, etc.
echo [2] CUDA 12.1 - Latest GPUs
echo [3] CPU only - No GPU
echo.
set /p cuda_choice="Enter choice (1-3): "

echo.

if "%cuda_choice%"=="1" (
    echo Installing PyTorch with CUDA 11.8 support...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    goto :continue_install
)

if "%cuda_choice%"=="2" (
    echo Installing PyTorch with CUDA 12.1 support...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    goto :continue_install
)

if "%cuda_choice%"=="3" (
    echo Installing PyTorch CPU only...
    pip install torch torchvision
    goto :continue_install
)

echo Invalid choice. Exiting.
pause
exit /b 1

:continue_install

echo.
echo Installing other dependencies...
echo.

REM Use quoted version specifiers to prevent redirect issues
pip install "numpy>=1.24.0"
pip install "scipy>=1.10.0"
pip install "scikit-learn>=1.2.0"
pip install "open3d>=0.17.0"
pip install "plyfile>=0.7.4"
pip install "pandas>=2.0.0"
pip install "h5py>=3.8.0"
pip install "pyyaml>=6.0"
pip install "matplotlib>=3.7.0"
pip install "seaborn>=0.12.0"
pip install "tqdm>=4.65.0"
pip install "transformers>=4.30.0"
pip install "timm>=0.9.0"
pip install "einops>=0.6.1"
pip install "requests>=2.31.0"
pip install "pillow>=9.5.0"
pip install "opencv-python>=4.8.0"

echo.
echo ===============================================
echo Installation complete!
echo ===============================================
echo.

if defined VIRTUAL_ENV (
    echo Virtual environment is active at: %VIRTUAL_ENV%
    echo.
    echo To activate this environment in the future, run:
    echo   venv\Scripts\activate
    echo.
    echo To deactivate, run:
    echo   deactivate
    echo.
)

echo Next steps:
echo 1. Test installation: python test_installation.py
echo 2. Run quick demo:     python quick_start.py
echo 3. Download KITTI:     python download_kitti.py
echo 4. Run full pipeline:  python demo.py
echo.
pause

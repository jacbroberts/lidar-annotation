"""
Installation Test Script

Run this script to verify that all dependencies are correctly installed.
"""

import sys
import io

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    print("-" * 50)

    required_packages = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('open3d', 'Open3D'),
        ('matplotlib', 'Matplotlib'),
        ('scipy', 'SciPy'),
        ('sklearn', 'scikit-learn'),
        ('yaml', 'PyYAML'),
        ('tqdm', 'tqdm'),
        ('requests', 'requests'),
        ('PIL', 'Pillow'),
    ]

    failed = []
    succeeded = []

    for package, name in required_packages:
        try:
            __import__(package)
            succeeded.append(name)
            print(f"  ✓ {name}")
        except ImportError as e:
            failed.append((name, str(e)))
            print(f"  ✗ {name} - FAILED")

    print("-" * 50)

    if failed:
        print(f"\n❌ {len(failed)} package(s) failed to import:")
        for name, error in failed:
            print(f"  - {name}: {error}")
        print("\nPlease install missing packages:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print(f"\n✓ All {len(succeeded)} required packages are installed!")
        return True


def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    print("-" * 50)

    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA is available")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
        else:
            print("  ℹ CUDA is not available (CPU mode only)")
            print("  This is OK, but processing will be slower")
    except Exception as e:
        print(f"  ⚠ Error checking CUDA: {e}")

    print("-" * 50)


def test_project_structure():
    """Test project structure"""
    print("\nTesting project structure...")
    print("-" * 50)

    from pathlib import Path

    required_dirs = [
        'models',
        'utils',
        'configs',
        'data',
        'outputs',
        'logs'
    ]

    required_files = [
        'demo.py',
        'download_kitti.py',
        'quick_start.py',
        'requirements.txt',
        'README.md',
        'configs/default_config.yaml',
    ]

    all_good = True

    print("Checking directories...")
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ - MISSING")
            all_good = False

    print("\nChecking files...")
    for file_name in required_files:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"  ✓ {file_name}")
        else:
            print(f"  ✗ {file_name} - MISSING")
            all_good = False

    print("-" * 50)

    if all_good:
        print("✓ Project structure is complete!")
    else:
        print("❌ Some files/directories are missing")

    return all_good


def test_model_imports():
    """Test project module imports"""
    print("\nTesting project modules...")
    print("-" * 50)

    modules = [
        'utils.point_cloud_utils',
        'utils.visualization',
        'models.segmentation_model',
        'models.nlp_model',
    ]

    all_good = True

    for module_name in modules:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name}")
        except ImportError as e:
            print(f"  ✗ {module_name} - FAILED: {e}")
            all_good = False

    print("-" * 50)

    if all_good:
        print("✓ All project modules loaded successfully!")
    else:
        print("❌ Some project modules failed to load")

    return all_good


def test_simple_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    print("-" * 50)

    try:
        import numpy as np
        from utils.point_cloud_utils import PointCloudProcessor
        from models.segmentation_model import SimpleSegmentationModel
        from models.nlp_model import SimpleNLPModel

        # Create dummy data
        points = np.random.randn(1000, 4).astype(np.float32)

        # Test processor
        processor = PointCloudProcessor()
        filtered = processor.filter_points(points)
        print(f"  ✓ Point cloud filtering: {len(points)} -> {len(filtered)} points")

        # Test segmentation model
        seg_model = SimpleSegmentationModel(num_classes=20, device='cpu')
        labels = seg_model.predict(points[:100])  # Small subset
        print(f"  ✓ Segmentation model: predicted {len(labels)} labels")

        # Test NLP model
        nlp_model = SimpleNLPModel(device='cpu')
        obj = {
            'points': points[:50],
            'semantic_label': 1,
            'instance_id': 0,
            'centroid': np.array([1.0, 2.0, 0.5]),
            'bbox_min': np.array([0.0, 1.0, 0.0]),
            'bbox_max': np.array([2.0, 3.0, 1.0]),
            'num_points': 50
        }
        description = nlp_model.describe_object(obj)
        print(f"  ✓ NLP model: generated description")
        print(f"    \"{description[:60]}...\"")

        print("-" * 50)
        print("✓ Basic functionality tests passed!")
        return True

    except Exception as e:
        print(f"  ✗ Functionality test failed: {e}")
        print("-" * 50)
        print("❌ Basic functionality test failed")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*70)
    print(" Installation Test")
    print("="*70)
    print()

    results = []

    # Run tests
    results.append(("Package Imports", test_imports()))
    test_cuda()  # Informational only
    results.append(("Project Structure", test_project_structure()))
    results.append(("Module Imports", test_model_imports()))
    results.append(("Basic Functionality", test_simple_functionality()))

    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("="*70)

    if all_passed:
        print("\n🎉 All tests passed! Your installation is ready.")
        print("\nNext steps:")
        print("  1. Run quick demo: python quick_start.py")
        print("  2. Download KITTI: python download_kitti.py")
        print("  3. Run full pipeline: python demo.py")
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Check Python version (3.8+ required)")
        print("  3. Ensure you're in the project directory")

    print()

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

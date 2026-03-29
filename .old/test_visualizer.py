#!/usr/bin/env python3
"""
Test script for the visualization utility
This script tests the visualizer package to ensure it works correctly
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all imports work correctly"""
    print("Testing imports...")
    try:
        from visualizer.config import DEFAULT_CONFIG, ATTACKS, CHARACTERISTICS
        print("✓ Config imports work")
        
        from visualizer.utils import parse_arguments, setup_environment
        print("✓ Utils imports work")
        
        from visualizer.data_loaders import load_original_data, check_required_files
        print("✓ Data loaders imports work")
        
        from visualizer.visualizers import AdversarialVisualizer, ModelVisualizer, DetectionVisualizer
        print("✓ Visualizers imports work")
        
        from visualizer.main import main
        print("✓ Main imports work")
        
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_data_availability():
    """Test data availability check"""
    print("\nTesting data availability check...")
    try:
        from visualizer.data_loaders import check_required_files
        
        # Check MNIST data
        status = check_required_files('mnist')
        print(f"MNIST data status:")
        print(f"  Model: {status['model']}")
        
        # Count available adversarial examples
        adv_count = sum(1 for v in status['adversarial'].values() if v)
        print(f"  Adversarial examples: {adv_count}/{len(status['adversarial'])}")
        
        # Count available characteristics
        char_count = sum(1 for char in status['characteristics'] 
                        for v in status['characteristics'][char].values() if v)
        total_chars = sum(len(status['characteristics'][char]) for char in status['characteristics'])
        print(f"  Characteristics: {char_count}/{total_chars}")
        
        return status
    except Exception as e:
        print(f"✗ Data availability check error: {e}")
        return None

def test_config():
    """Test configuration constants"""
    print("\nTesting configuration...")
    try:
        from visualizer.config import ATTACKS, CHARACTERISTICS, MNIST_CONFIG
        
        print(f"Available attacks: {ATTACKS}")
        print(f"Available characteristics: {CHARACTERISTICS}")
        print(f"MNIST config: {MNIST_CONFIG}")
        
        return True
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False

def test_visualizer_creation():
    """Test creating visualizer objects"""
    print("\nTesting visualizer creation...")
    try:
        from visualizer.visualizers import AdversarialVisualizer, ModelVisualizer, DetectionVisualizer
        
        # Test basic creation
        adv_viz = AdversarialVisualizer(dataset='mnist')
        model_viz = ModelVisualizer(dataset='mnist')
        detect_viz = DetectionVisualizer(dataset='mnist')
        
        print("✓ All visualizers created successfully")
        return True
    except Exception as e:
        print(f"✗ Visualizer creation error: {e}")
        return False

def test_argument_parser():
    """Test CLI argument parsing"""
    print("\nTesting argument parser...")
    try:
        # Mock command line arguments
        original_argv = sys.argv
        sys.argv = ['visualizer', '--mode', 'all', '--dataset', 'mnist']
        
        from visualizer.utils import parse_arguments
        args = parse_arguments()
        
        print(f"Parsed args: mode={args.mode}, dataset={args.dataset}")
        
        sys.argv = original_argv
        return True
    except Exception as e:
        print(f"✗ Argument parsing error: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("VISUALIZATION UTILITY TEST SUITE")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Data Availability", test_data_availability),
        ("Visualizer Creation", test_visualizer_creation),
        ("Argument Parser", test_argument_parser),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ Test '{name}' failed with exception: {e}")
            results[name] = False
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:25s}: {status}")
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The visualization utility is ready to use.")
        print("\nQuick start:")
        print("  python -m visualizer.main --mode all --dataset mnist")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
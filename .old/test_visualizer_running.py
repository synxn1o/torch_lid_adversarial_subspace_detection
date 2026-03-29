#!/usr/bin/env python3
"""
Test the visualization utility with actual MNIST data
This script provides a quick validation that the utility works correctly
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_with_real_data():
    """Test the visualizers with actual MNIST data"""
    print("="*70)
    print("TESTING VISUALIZATION UTILITY WITH REAL MNIST DATA")
    print("="*70)
    
    # Check if required files exist
    required_files = [
        "data/model_mnist.pth",
        "data/Adv_mnist_fgsm.npy",
        "data/lid_mnist_fgsm.npy",
        "data/kd_mnist_fgsm.npy",
        "data/bu_mnist_fgsm.npy"
    ]
    
    print("\n1. Checking required data files...")
    missing = []
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✓ {file}")
        else:
            print(f"   ✗ {file} - MISSING")
            missing.append(file)
    
    if missing:
        print(f"\n❌ Missing {len(missing)} required files. Cannot proceed.")
        print("Please run the data generation scripts first:")
        print("  - python train_model.py --dataset mnist")
        print("  - python craft_adv_examples.py --dataset mnist --attack fgsm")
        print("  - python extract_characteristics.py --dataset mnist --attack fgsm")
        return False
    
    print("\n✓ All required files present!")
    
    # Test imports and basic functionality
    print("\n2. Testing imports...")
    try:
        from visualizer.config import ATTACKS, CHARACTERISTICS
        from visualizer.data_loaders import load_adversarial_data, load_characteristics_data
        from visualizer.visualizers import AdversarialVisualizer, ModelVisualizer, DetectionVisualizer
        print("   ✓ All imports successful")
    except Exception as e:
        print(f"   ✗ Import failed: {e}")
        return False
    
    # Test data loading
    print("\n3. Testing data loading...")
    try:
        # Load adversarial data
        adv_data = load_adversarial_data('mnist', 'fgsm', max_samples=100)
        print(f"   ✓ Adversarial data: shape {adv_data.shape}")
        
        # Load characteristics
        lid_data, lid_labels = load_characteristics_data('mnist', 'lid', 'fgsm', max_samples=100)
        print(f"   ✓ LID characteristics: shape {lid_data.shape}, labels: {len(lid_labels)}")
        
        kd_data, kd_labels = load_characteristics_data('mnist', 'kd', 'fgsm', max_samples=100)
        print(f"   ✓ KD characteristics: shape {kd_data.shape}, labels: {len(kd_labels)}")
        
    except Exception as e:
        print(f"   ✗ Data loading failed: {e}")
        return False
    
    # Test visualizer creation
    print("\n4. Testing visualizer creation...")
    try:
        adv_viz = AdversarialVisualizer(dataset='mnist', dpi=72)  # Low DPI for speed
        model_viz = ModelVisualizer(dataset='mnist', dpi=72)
        detect_viz = DetectionVisualizer(dataset='mnist', dpi=72)
        print("   ✓ All visualizers created successfully")
    except Exception as e:
        print(f"   ✗ Visualizer creation failed: {e}")
        return False
    
    # Test specific visualizations
    print("\n5. Testing specific visualizations...")
    
    # Create test output directory
    test_output = Path("visualizer/outputs/test")
    test_output.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Test 1: Adversarial image grid
    print("   Testing adversarial image grid...")
    try:
        adv_viz.create_image_grid_comparison('fgsm', num_samples=4, save=True)
        results['adversarial_grid'] = True
        print("   ✓ Adversarial image grid generated")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        results['adversarial_grid'] = False
    
    # Test 2: Perturbation analysis
    print("   Testing perturbation analysis...")
    try:
        adv_viz.create_perturbation_analysis(attacks=['fgsm'], save=True)
        results['perturbation_analysis'] = True
        print("   ✓ Perturbation analysis generated")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        results['perturbation_analysis'] = False
    
    # Test 3: Attack metrics
    print("   Testing attack metrics...")
    try:
        adv_viz.create_attack_success_metrics(attacks=['fgsm'], save=True)
        results['attack_metrics'] = True
        print("   ✓ Attack metrics generated")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        results['attack_metrics'] = False
    
    # Test 4: Detection ROC comparison
    print("   Testing detection ROC comparison...")
    try:
        detect_viz.create_roc_comparison(attacks=['fgsm'], characteristics=['lid', 'kd'], save=True)
        results['roc_comparison'] = True
        print("   ✓ ROC comparison generated")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        results['roc_comparison'] = False
    
    # Test 5: 3D feature visualization
    print("   Testing 3D feature visualization...")
    try:
        detect_viz.create_3d_feature_space('fgsm', 'lid', save=True)
        results['3d_features'] = True
        print("   ✓ 3D feature plot generated")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        results['3d_features'] = False
    
    # Test 6: Probability distributions
    print("   Testing probability distributions...")
    try:
        detect_viz.create_probability_distributions(attacks=['fgsm'], characteristics=['lid', 'kd'], save=True)
        results['distributions'] = True
        print("   ✓ Probability distributions generated")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        results['distributions'] = False
    
    # Test 7: Metrics comparison
    print("   Testing metrics comparison...")
    try:
        detect_viz.create_metrics_comparison(attacks=['fgsm'], characteristics=['lid', 'kd'], save=True)
        results['metrics_comparison'] = True
        print("   ✓ Metrics comparison generated")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        results['metrics_comparison'] = False
    
    # Test 8: Model visualizations
    print("   Testing model visualizations...")
    try:
        model_viz.create_training_curves(save=True)
        model_viz.create_confusion_matrix(save=True)
        model_viz.create_roc_analysis(save=True)
        results['model_visualizations'] = True
        print("   ✓ Model visualizations generated")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        results['model_visualizations'] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    successful = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:25s}: {status}")
    
    print(f"\nResults: {successful}/{total} tests passed")
    
    if successful == total:
        print("\n🎉 ALL TESTS PASSED!")
        print(f"\nVisualizations saved to: {test_output}")
        print("\nYou can now run the full utility:")
        print("  python -m visualizer.main --mode all --dataset mnist")
        return True
    else:
        print(f"\n⚠️  {total - successful} tests failed. Check errors above.")
        return False

def main():
    """Main test function"""
    try:
        success = test_with_real_data()
        return 0 if success else 1
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
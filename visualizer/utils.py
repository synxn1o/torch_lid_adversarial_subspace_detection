"""
Utility functions for the visualization package
"""

import os
import numpy as np
import torch
from typing import Dict, List, Optional, Any
from pathlib import Path
import argparse
import sys

from visualizer.config import (
    ATTACKS, CHARACTERISTICS, OUTPUT_FORMATS, 
    DEFAULT_CONFIG, validate_dataset, validate_attack, validate_characteristic
)
from visualizer.data_loaders import check_required_files


def setup_environment():
    """Setup paths and environment"""
    # Add parent directory to path
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    # Create output directory
    from visualizer.config import OUTPUT_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Adversarial ML Visualization Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all visualizations for MNIST
  python -m visualizer.main --mode all --dataset mnist
  
  # Generate specific visualizations
  python -m visualizer.main --mode adversarial --dataset mnist --attack fgsm
  python -m visualizer.main --mode model --dataset mnist
  python -m visualizer.main --mode detection --dataset mnist --attack all
  
  # Interactive mode
  python -m visualizer.main --mode interactive --dataset mnist
        """
    )
    
    # Mode selection
    parser.add_argument('--mode', '-m', 
                       choices=['adversarial', 'model', 'detection', 'tda', 'all', 'interactive'],
                       default='all',
                       help='Visualization mode to run')
    
    # Dataset selection
    parser.add_argument('--dataset', '-d',
                       default='mnist',
                       help='Dataset name (mnist, cifar, svhn) - Default: mnist')
    
    # Attack selection
    parser.add_argument('--attack', '-a',
                       help='Attack name (fgsm, bim-a, bim-b, jsma, cw-l2, cw-lid, or all)')
    
    # Characteristic selection
    parser.add_argument('--characteristics', '-c',
                       help='Comma-separated characteristics (lid, kd, bu, km, or all)')
    
    # Output options
    parser.add_argument('--output-dir', '-o',
                       help='Output directory for visualizations')
    
    parser.add_argument('--format', '-f',
                       choices=OUTPUT_FORMATS,
                       default='png',
                       help='Output file format')
    
    parser.add_argument('--dpi', type=int, default=300,
                       help='Resolution for saved figures')
    
    # Performance options
    parser.add_argument('--sample-limit', type=int,
                       help='Maximum samples to load')
    
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching')
    
    # What to generate
    parser.add_argument('--what', '-w',
                       help='Specific visualization to generate (comma-separated)')
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments"""
    if not validate_dataset(args.dataset):
        raise ValueError(f"Invalid dataset: {args.dataset}")
    
    if args.attack and args.attack != 'all':
        if not validate_attack(args.attack):
            raise ValueError(f"Invalid attack: {args.attack}")
    
    if args.characteristics:
        # In TDA mode, characteristics are actually model names, so we skip validation
        if args.mode != 'tda':
            for char in args.characteristics.split(','):
                if not validate_characteristic(char):
                    raise ValueError(f"Invalid characteristic: {char}")


def check_data_availability(dataset: str, mode: str) -> Dict[str, bool]:
    """Check what data is available for visualization"""
    availability = check_required_files(dataset)
    
    results = {
        'model': availability['model'],
        'adversarial': any(availability['adversarial'].values()),
        'characteristics': any(
            any(availability['characteristics'][char].values()) 
            for char in CHARACTERISTICS
        )
    }
    
    if mode == 'all':
        return results
    elif mode == 'adversarial':
        return {'adversarial': results['adversarial']}
    elif mode == 'model':
        return {'model': results['model']}
    elif mode == 'detection':
        return {'characteristics': results['characteristics']}
    
    return results


def print_data_status(dataset: str):
    """Print status of available data files"""
    status = check_required_files(dataset)
    
    print(f"\n{'='*60}")
    print(f"DATA STATUS FOR {dataset.upper()}")
    print(f"{'='*60}")
    
    print(f"\nModel File:")
    print(f"  model_{dataset}.pth: {'✓' if status['model'] else '✗'}")
    
    print(f"\nAdversarial Examples:")
    for attack in ATTACKS:
        available = status['adversarial'][attack]
        print(f"  {attack:8s}: {'✓' if available else '✗'}")
    
    print(f"\nCharacteristics:")
    for char in CHARACTERISTICS:
        print(f"  {char.upper()}:")
        for attack in ATTACKS:
            available = status['characteristics'][char][attack]
            print(f"    {attack:8s}: {'✓' if available else '✗'}")
    
    print(f"\n{'='*60}")


def get_what_to_generate(args, availability: Dict) -> List[str]:
    """Determines which specific visualizations to generate"""
    if not args.what:
        # Return all visualizations for the selected mode
        if args.mode == 'adversarial':
            return ['image_grid', 'perturbation_analysis', 'attack_metrics']
        elif args.mode == 'model':
            return ['training_curves', 'confusion_matrix', 'roc_analysis']
        elif args.mode == 'detection':
            return ['roc_comparison', '3d_features', 'probability_distributions', 'metrics_comparison']
        elif args.mode == 'tda':
            return ['persistence_diagram', 'tda_comparison', 'tda_clean_vs_adv', 'correlation_matrix']
        elif args.mode == 'all':
            return [
                'image_grid', 'perturbation_analysis', 'attack_metrics',
                'training_curves', 'confusion_matrix', 'roc_analysis',
                'roc_comparison', '3d_features', 'probability_distributions', 'metrics_comparison',
                'persistence_diagram', 'tda_comparison', 'tda_clean_vs_adv', 'correlation_matrix'
            ]
    
    # Parse specific what argument
    return [item.strip() for item in args.what.split(',')]


def create_summary_report(results: Dict, dataset: str, mode: str, output_dir: str):
    """Create a summary report of generated visualizations"""
    report_path = os.path.join(output_dir, f"summary_{dataset}_{mode}.txt")
    
    with open(report_path, 'w') as f:
        f.write(f"Adversarial ML Visualization Summary\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Generated: {len(results)} visualizations\n\n")
        
        for viz_name, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            f.write(f"{status}: {viz_name}\n")
        
        f.write(f"\nOutput Directory: {output_dir}\n")
    
    print(f"\nSummary report saved to: {report_path}")


def ensure_dependencies():
    """Check for required dependencies"""
    missing = []
    
    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")
    
    try:
        import seaborn
    except ImportError:
        missing.append("seaborn")
    
    try:
        import sklearn
    except ImportError:
        missing.append("scikit-learn")
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import pandas
    except ImportError:
        missing.append("pandas")
    
    if missing:
        raise ImportError(
            f"Missing required dependencies: {', '.join(missing)}\n"
            f"Please install them using: pip install {' '.join(missing)}"
        )


def format_attack_list(attacks: List[str]) -> str:
    """Format attack list for display"""
    if len(attacks) <= 3:
        return ', '.join(attacks)
    else:
        return ', '.join(attacks[:3]) + f' (+{len(attacks)-3} more)'


def filter_available_attacks(attacks: List[str], availability: Dict) -> List[str]:
    """Filter attacks to only those with available data"""
    return [attack for attack in attacks if availability['adversarial'].get(attack, False)]


def filter_available_characteristics(characteristics: List[str], availability: Dict) -> List[str]:
    """Filter characteristics to only those with available data"""
    available = []
    for char in characteristics:
        if any(availability['characteristics'].get(char, {}).values()):
            available.append(char)
    return available


def print_banner():
    """Print ASCII art banner"""
    banner = """
    ╔════════════════════════════════════════════════════════════╗
    ║   Adversarial ML Visualization Utility                    ║
    ║   MNIST-Focused Analysis & Research Tools                 ║
    ╚════════════════════════════════════════════════════════════╝
    
    Capabilities:
    • Original vs Adversarial Image Comparison
    • Perturbation Analysis & Attack Metrics
    • Model Training & Performance Visualization
    • Adversarial Detection Analysis (LID, KD, BU, KM)
    • Multi-Attack & Multi-Characteristic Comparison
    
    """
    print(banner)
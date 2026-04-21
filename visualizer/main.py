"""
Main CLI interface for the visualization utility
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from visualizer.config import DEFAULT_CONFIG, ATTACKS, CHARACTERISTICS, OUTPUT_DIR
from visualizer.utils import (
    parse_arguments, validate_arguments, print_banner, check_data_availability,
    print_data_status, get_what_to_generate, create_summary_report, ensure_dependencies,
    format_attack_list, filter_available_attacks, filter_available_characteristics
)
from visualizer.data_loaders import check_required_files
from visualizer.visualizers import AdversarialVisualizer, ModelVisualizer, DetectionVisualizer, TDAVisualizer


def run_adversarial_visualizations(dataset: str, attacks: List[str], what: List[str], 
                                 visualizer: AdversarialVisualizer) -> Dict[str, bool]:
    """Run adversarial visualizations"""
    results = {}
    
    if 'image_grid' in what:
        for attack in attacks:
            print(f"  Generating image grid for {attack}...")
            try:
                visualizer.create_image_grid_comparison(attack, num_samples=16, save=True)
                results[f'image_grid_{attack}'] = True
            except Exception as e:
                print(f"    Error: {e}")
                results[f'image_grid_{attack}'] = False
    
    if 'perturbation_analysis' in what:
        print("  Generating perturbation analysis...")
        try:
            visualizer.create_perturbation_analysis(attacks=attacks, save=True)
            results['perturbation_analysis'] = True
        except Exception as e:
            print(f"    Error: {e}")
            results['perturbation_analysis'] = False
    
    if 'attack_metrics' in what:
        print("  Generating attack success metrics...")
        try:
            visualizer.create_attack_success_metrics(attacks=attacks, save=True)
            results['attack_metrics'] = True
        except Exception as e:
            print(f"    Error: {e}")
            results['attack_metrics'] = False
    
    return results


def run_model_visualizations(dataset: str, what: List[str], 
                           visualizer: ModelVisualizer) -> Dict[str, bool]:
    """Run model visualizations"""
    results = {}
    
    if 'training_curves' in what:
        print("  Generating training curves...")
        try:
            visualizer.create_training_curves(save=True)
            results['training_curves'] = True
        except Exception as e:
            print(f"    Error: {e}")
            results['training_curves'] = False
    
    if 'confusion_matrix' in what:
        print("  Generating confusion matrix...")
        try:
            visualizer.create_confusion_matrix(save=True)
            results['confusion_matrix'] = True
        except Exception as e:
            print(f"    Error: {e}")
            results['confusion_matrix'] = False
    
    if 'roc_analysis' in what:
        print("  Generating ROC analysis...")
        try:
            visualizer.create_roc_analysis(save=True)
            results['roc_analysis'] = True
        except Exception as e:
            print(f"    Error: {e}")
            results['roc_analysis'] = False
    
    return results


def run_detection_visualizations(dataset: str, attacks: List[str], characteristics: List[str],
                               what: List[str], visualizer: DetectionVisualizer) -> Dict[str, bool]:
    """Run detection visualizations"""
    results = {}
    
    if 'roc_comparison' in what:
        print("  Generating ROC comparison...")
        try:
            visualizer.create_roc_comparison(attacks=attacks, characteristics=characteristics, save=True)
            results['roc_comparison'] = True
        except Exception as e:
            print(f"    Error: {e}")
            results['roc_comparison'] = False
    
    if '3d_features' in what:
        # Generate 3D plots for first attack and characteristic combination
        if attacks and characteristics:
            print(f"  Generating 3D feature plots...")
            for attack in attacks[:2]:  # Limit to first 2 attacks to avoid too many plots
                for char in characteristics[:2]:  # Limit to first 2 characteristics
                    try:
                        visualizer.create_3d_feature_space(attack=attack, characteristic=char, save=True)
                        results[f'3d_{char}_{attack}'] = True
                    except Exception as e:
                        print(f"    Error for {char}/{attack}: {e}")
                        results[f'3d_{char}_{attack}'] = False
    
    if 'probability_distributions' in what:
        print("  Generating probability distributions...")
        try:
            visualizer.create_probability_distributions(attacks=attacks, characteristics=characteristics, save=True)
            results['probability_distributions'] = True
        except Exception as e:
            print(f"    Error: {e}")
            results['probability_distributions'] = False
    
    if 'metrics_comparison' in what:
        print("  Generating metrics comparison...")
        try:
            visualizer.create_metrics_comparison(attacks=attacks, characteristics=characteristics, save=True)
            results['metrics_comparison'] = True
        except Exception as e:
            print(f"    Error: {e}")
            results['metrics_comparison'] = False
    
    return results


def run_tda_visualizations(dataset: str, what: List[str],
                         visualizer: TDAVisualizer, attacks: Optional[List[str]] = None) -> Dict[str, bool]:
    """Run TDA visualizations"""
    results = {}
    
    tda_attacks = attacks if attacks else ['clean']
    
    if 'persistence_diagram' in what:
        for attack in tda_attacks:
            print(f"  Generating persistence diagram for {attack}...")
            try:
                visualizer.create_persistence_diagram(attack, save=True)
                results[f'persistence_{attack}'] = True
            except Exception as e:
                print(f"    Error: {e}")
                results[f'persistence_{attack}'] = False
    
    if 'tda_comparison' in what:
        print("  Generating TDA feature comparison...")
        try:
            visualizer.create_feature_comparison(tda_attacks, save=True)
            results['tda_comparison'] = True
        except Exception as e:
            print(f"    Error: {e}")
            results['tda_comparison'] = False
    
    if 'tda_clean_vs_adv' in what:
        attack = tda_attacks[0] if tda_attacks else 'fgsm'
        print(f"  Generating TDA clean vs {attack} comparison...")
        try:
            visualizer.create_clean_vs_adversarial_comparison(attack, save=True)
            results['tda_clean_vs_adv'] = True
        except Exception as e:
            print(f"    Error: {e}")
            results['tda_clean_vs_adv'] = False
    
    if 'correlation_matrix' in what:
        for attack in tda_attacks:
            print(f"  Generating correlation matrix for {attack}...")
            try:
                visualizer.create_correlation_matrix_plot(attack, save=True)
                results[f'correlation_{attack}'] = True
            except Exception as e:
                print(f"    Error: {e}")
                results[f'correlation_{attack}'] = False
            
    return results


def main():
    """Main entry point"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Validate arguments
        validate_arguments(args)
        
        # Setup environment
        from visualizer.utils import setup_environment
        setup_environment()

        # Apply data-dir override if provided
        if args.data_dir:
            import visualizer.config as cfg
            cfg.RESULTS_DIR = Path(args.data_dir)
        
        # Check dependencies
        ensure_dependencies()
        
        # Print banner
        print_banner()
        
        # Print current configuration
        print(f"Current Configuration:")
        print(f"  Dataset: {args.dataset}")
        print(f"  Mode: {args.mode}")
        print(f"  Output Format: {args.format}")
        print(f"  DPI: {args.dpi}")
        
        # Check data availability
        print(f"\nChecking data availability for {args.dataset.upper()}...")
        availability = check_required_files(args.dataset)
        
        # Determine attacks and characteristics
        if args.attack:
            if args.attack == 'all':
                attacks = filter_available_attacks(ATTACKS, availability)
            else:
                attacks = [args.attack]
                attacks = filter_available_attacks(attacks, availability)
        else:
            attacks = []
        
        if args.characteristics:
            if args.characteristics == 'all':
                characteristics = filter_available_characteristics(CHARACTERISTICS, availability)
            else:
                characteristics = [c.strip() for c in args.characteristics.split(',')]
                # Skip filtering for TDA mode as these are model names
                if args.mode != 'tda':
                    characteristics = filter_available_characteristics(characteristics, availability)
        else:
            characteristics = []
        
        # Determine what to generate
        what_to_generate = get_what_to_generate(args, availability)
        
        # Check if required data is available
        if args.mode in ['adversarial', 'all']:
            if not availability['model']:
                print("\n⚠️  Warning: Model file not found. Some visualizations may fail.")
            if not attacks:
                print("\n⚠️  Warning: No adversarial data available for specified attacks.")
                if args.mode == 'adversarial':
                    print("  Skipping adversarial visualizations.")
                    return
        
        if args.mode in ['detection', 'all']:
            if not characteristics:
                print("\n⚠️  Warning: No characteristic data available.")
                if args.mode == 'detection':
                    print("  Skipping detection visualizations.")
                    return
        
        if args.mode == 'model' and not availability['model']:
            print("\n⚠️  Warning: Model file not found. Cannot generate model visualizations.")
            return
        
        # Print data status
        print_data_status(args.dataset)
        
        # Initialize output directory
        output_dir = Path(args.output_dir) if args.output_dir else None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results dictionary
        all_results = {}
        
        print(f"\n{'='*60}")
        print(f"GENERATING VISUALIZATIONS")
        print(f"{'='*60}")
        
        # Run visualizations based on mode
        if args.mode == 'adversarial' or args.mode == 'all':
            print(f"\n[ADVERSARIAL ANALYSIS]")
            if attacks:
                print(f"Attacks: {format_attack_list(attacks)}")
                viz = AdversarialVisualizer(
                    dataset=args.dataset,
                    style='presentation',
                    dpi=args.dpi,
                    output_dir=str(output_dir) if output_dir else None
                )
                results = run_adversarial_visualizations(args.dataset, attacks, what_to_generate, viz)
                all_results.update(results)
            else:
                print("  No adversarial data available - skipping")
        
        if args.mode == 'model' or args.mode == 'all':
            print(f"\n[MODEL ANALYSIS]")
            viz = ModelVisualizer(
                dataset=args.dataset,
                style='presentation',
                dpi=args.dpi,
                output_dir=str(output_dir) if output_dir else None
            )
            results = run_model_visualizations(args.dataset, what_to_generate, viz)
            all_results.update(results)
        
        if args.mode == 'detection' or args.mode == 'all':
            print(f"\n[DETECTION ANALYSIS]")
            if attacks and characteristics:
                print(f"Attacks: {format_attack_list(attacks)}")
                print(f"Characteristics: {', '.join(characteristics)}")
                viz = DetectionVisualizer(
                    dataset=args.dataset,
                    style='presentation',
                    dpi=args.dpi,
                    output_dir=str(output_dir) if output_dir else None
                )
                results = run_detection_visualizations(
                    args.dataset, attacks, characteristics, what_to_generate, viz
                )
                all_results.update(results)
            else:
                print("  No detection data available - skipping")
        
        if args.mode == 'tda' or args.mode == 'all':
            print(f"\n[TDA ANALYSIS]")
            viz = TDAVisualizer(
                dataset=args.dataset,
                style='presentation',
                dpi=args.dpi,
                output_dir=str(output_dir) if output_dir else None
            )
            # Check if user provided specific TDA names via --attack or --characteristics
            # For TDA mode, we can repurpose --characteristics to take a list of model names
            tda_attacks = None
            if args.attack:
                if args.attack == 'all':
                    tda_attacks = list(ATTACKS) + ['clean']
                else:
                    tda_attacks = [args.attack]
            elif args.characteristics and args.characteristics != 'all':
                tda_attacks = [c.strip() for c in args.characteristics.split(',')]
            
            results = run_tda_visualizations(args.dataset, what_to_generate, viz, attacks=tda_attacks)
            all_results.update(results)

        # Interactive mode would be implemented here
        if args.mode == 'interactive':
            print("\n[INTERACTIVE MODE]")
            print("Interactive mode would launch a web interface here.")
            print("This requires Streamlit or Plotly Dash.")
            print("For now, use the other modes for static visualizations.")
        
        # Create summary
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        
        successful = sum(1 for v in all_results.values() if v)
        total = len(all_results)
        
        print(f"Generated {successful}/{total} visualizations")
        
        if successful > 0:
            summary_dir = str(output_dir) if output_dir else str(OUTPUT_DIR)
            create_summary_report(all_results, args.dataset, args.mode, summary_dir)
            print(f"\nVisualizations saved to: {summary_dir}")
        
        return 0 if successful > 0 else 1
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
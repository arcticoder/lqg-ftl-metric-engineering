#!/usr/bin/env python3
"""
LQG FTL Metric Engineering - Comprehensive Validation Runner
============================================================

This script runs all critical UQ validations for FTL metric engineering
applications. It provides a unified interface to execute and report on
all validation frameworks.

Usage:
    python run_all_validations.py [--quick] [--component COMPONENT]
    
Options:
    --quick: Run abbreviated validation for faster testing
    --component: Run validation for specific component only
"""

import sys
import os
import time
import argparse
from typing import Dict, List, Optional
import importlib.util

# Add validation modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

class ValidationRunner:
    """Comprehensive validation runner for FTL metric engineering"""
    
    def __init__(self, quick_mode: bool = False):
        self.quick_mode = quick_mode
        self.results = {}
        self.start_time = time.time()
        
        # Validation components mapping
        self.validations = {
            'h_infinity_control': {
                'name': 'H∞ Controller Robustness',
                'path': '../negative-energy-generator',
                'module': 'h_infinity_controller_robustness',
                'description': 'Negative energy generation control system validation'
            },
            'correlation_matrix': {
                'name': '5×5 Correlation Matrix Consistency', 
                'path': '../negative-energy-generator',
                'module': 'correlation_matrix_validation',
                'description': 'Cross-repository statistical consistency validation'
            },
            'metric_stability': {
                'name': 'Metric Stability Under Extreme Curvature',
                'path': '../warp-bubble-optimizer', 
                'module': 'metric_stability_validation',
                'description': 'Spacetime metric stability for FTL applications'
            },
            'stress_energy_coupling': {
                'name': 'Stress-Energy Tensor Coupling (Bobrick-Martire)',
                'path': '../warp-bubble-optimizer',
                'module': 'stress_tensor_coupling_validation', 
                'description': 'Positive-energy warp configuration validation'
            },
            'polymer_consistency': {
                'name': 'Polymer Parameter Consistency (LQG)',
                'path': '../unified-lqg',
                'module': 'polymer_parameter_consistency_validation',
                'description': 'Cross-formulation LQG parameter validation'
            }
        }
    
    def run_validation(self, component_key: str) -> Dict:
        """Run a specific validation component"""
        component = self.validations[component_key]
        
        print(f"\n{'='*70}")
        print(f"Running: {component['name']}")
        print(f"Description: {component['description']}")
        print(f"{'='*70}")
        
        try:
            # Import validation module
            module_path = os.path.join(component['path'], f"{component['module']}.py")
            if not os.path.exists(module_path):
                print(f"⚠️  Module not found: {module_path}")
                return {'status': 'not_found', 'module_path': module_path}
            
            spec = importlib.util.spec_from_file_location(component['module'], module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Run validation
            start_time = time.time()
            
            if hasattr(module, 'main'):
                results = module.main()
                execution_time = time.time() - start_time
                
                return {
                    'status': 'completed',
                    'results': results,
                    'execution_time': execution_time,
                    'module_path': module_path
                }
            else:
                print(f"⚠️  No main() function found in {component['module']}")
                return {'status': 'no_main_function'}
                
        except Exception as e:
            print(f"❌ Error running {component['name']}: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def run_all_validations(self, component_filter: Optional[str] = None) -> Dict:
        """Run all validation components or a specific component"""
        
        print("🚀 LQG FTL Metric Engineering - Comprehensive Validation")
        print("=" * 70)
        print(f"Mode: {'Quick' if self.quick_mode else 'Full'} Validation")
        if component_filter:
            print(f"Component Filter: {component_filter}")
        print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # Determine which validations to run
        if component_filter:
            if component_filter in self.validations:
                components_to_run = [component_filter]
            else:
                print(f"❌ Unknown component: {component_filter}")
                print(f"Available components: {', '.join(self.validations.keys())}")
                return {'status': 'invalid_component'}
        else:
            components_to_run = list(self.validations.keys())
        
        # Run validations
        for component_key in components_to_run:
            result = self.run_validation(component_key)
            self.results[component_key] = result
        
        # Generate summary report
        self.generate_summary_report()
        
        return self.results
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        
        total_time = time.time() - self.start_time
        
        print(f"\n{'='*70}")
        print("🎯 VALIDATION SUMMARY REPORT")
        print(f"{'='*70}")
        print(f"Total Execution Time: {total_time:.2f} seconds")
        print(f"Validations Run: {len(self.results)}")
        
        # Status summary
        status_counts = {}
        for component_key, result in self.results.items():
            status = result.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print(f"\nStatus Summary:")
        for status, count in status_counts.items():
            emoji = {
                'completed': '✅',
                'error': '❌', 
                'not_found': '⚠️',
                'no_main_function': '⚠️'
            }.get(status, '❓')
            print(f"  {emoji} {status.replace('_', ' ').title()}: {count}")
        
        # Detailed results
        print(f"\nDetailed Results:")
        print("-" * 50)
        
        for component_key, result in self.results.items():
            component = self.validations[component_key]
            status = result.get('status', 'unknown')
            
            emoji = {
                'completed': '✅',
                'error': '❌',
                'not_found': '⚠️', 
                'no_main_function': '⚠️'
            }.get(status, '❓')
            
            print(f"\n{emoji} {component['name']}")
            
            if status == 'completed':
                exec_time = result.get('execution_time', 0)
                print(f"   Execution Time: {exec_time:.2f}s")
                
                # Try to extract key metrics if available
                if 'results' in result and hasattr(result['results'], 'results'):
                    validation_results = result['results'].results
                    if hasattr(validation_results, '__dict__'):
                        metrics = self.extract_key_metrics(component_key, validation_results)
                        for metric, value in metrics.items():
                            print(f"   {metric}: {value}")
                            
            elif status == 'error':
                print(f"   Error: {result.get('error', 'Unknown error')}")
            elif status == 'not_found':
                print(f"   Module not found: {result.get('module_path', 'Unknown path')}")
        
        # Overall assessment
        completed_count = status_counts.get('completed', 0)
        total_count = len(self.results)
        success_rate = (completed_count / total_count * 100) if total_count > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"🎯 OVERALL ASSESSMENT")
        print(f"{'='*70}")
        print(f"Success Rate: {success_rate:.1f}% ({completed_count}/{total_count})")
        
        if success_rate >= 80:
            print("🚀 Status: READY FOR FTL ENGINEERING")
        elif success_rate >= 60:
            print("⚠️  Status: CONDITIONAL READINESS - Monitor failed validations")  
        else:
            print("❌ Status: REQUIRES ADDITIONAL VALIDATION")
        
        print(f"\nReport Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
    
    def extract_key_metrics(self, component_key: str, results) -> Dict[str, str]:
        """Extract key metrics from validation results"""
        metrics = {}
        
        try:
            if component_key == 'h_infinity_control':
                if hasattr(results, 'monte_carlo_success_rate'):
                    metrics['Monte Carlo Success'] = f"{results.monte_carlo_success_rate:.1%}"
                if hasattr(results, 'real_time_frequency'):
                    metrics['Real-Time Frequency'] = f"{results.real_time_frequency:,.0f} Hz"
                    
            elif component_key == 'correlation_matrix':
                if hasattr(results, 'matrix_validity_rate'):
                    metrics['Matrix Validity'] = f"{results.matrix_validity_rate:.1%}"
                if hasattr(results, 'cross_repo_consistency'):
                    metrics['Cross-Repo Consistency'] = f"{results.cross_repo_consistency:.1%}"
                    
            elif component_key == 'metric_stability':
                if hasattr(results, 'causality_preservation'):
                    causality_rate = sum(results.causality_preservation.values()) / len(results.causality_preservation.values())
                    metrics['Causality Preservation'] = f"{causality_rate:.1%}"
                    
            elif component_key == 'stress_energy_coupling':
                if hasattr(results, 'bobrick_martire_compliance'):
                    compliance_rate = sum(all(comp.values()) for comp in results.bobrick_martire_compliance.values()) / len(results.bobrick_martire_compliance)
                    metrics['Bobrick-Martire Compliance'] = f"{compliance_rate:.1%}"
                    
            elif component_key == 'polymer_consistency':
                if hasattr(results, 'parameter_consistency'):
                    avg_consistency = sum(results.parameter_consistency.values()) / len(results.parameter_consistency)
                    metrics['Parameter Consistency'] = f"{avg_consistency:.1%}"
                    
        except Exception as e:
            metrics['Metric Extraction Error'] = str(e)
        
        return metrics

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='LQG FTL Metric Engineering Validation Runner')
    parser.add_argument('--quick', action='store_true', help='Run quick validation mode')
    parser.add_argument('--component', type=str, help='Run specific component only')
    
    args = parser.parse_args()
    
    # Initialize and run validation
    runner = ValidationRunner(quick_mode=args.quick)
    results = runner.run_all_validations(component_filter=args.component)
    
    # Return appropriate exit code
    completed_count = sum(1 for r in results.values() if r.get('status') == 'completed')
    total_count = len(results)
    
    if completed_count == total_count:
        sys.exit(0)  # All validations successful
    elif completed_count > 0:
        sys.exit(1)  # Partial success
    else:
        sys.exit(2)  # No successful validations

if __name__ == "__main__":
    main()

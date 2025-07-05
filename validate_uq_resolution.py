#!/usr/bin/env python3
"""
UQ Resolution Validation Script
=============================

Comprehensive validation of uncertainty quantification (UQ) concern resolutions
in the zero exotic energy framework for production-ready FTL metric engineering.
"""

import sys
import os
import numpy as np
from typing import Dict, Any
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from zero_exotic_energy_framework import (
    complete_zero_exotic_energy_analysis,
    EnhancedBobrickMartireFramework,
    ZeroExoticEnergyOptimizationFramework,
    numerical_safety_context,
    CONSERVATION_TOLERANCE,
    NUMERICAL_EPSILON
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_numerical_safety_context():
    """Test the numerical safety context manager."""
    print("\n" + "="*60)
    print("🔍 VALIDATING NUMERICAL SAFETY CONTEXT")
    print("="*60)
    
    try:
        with numerical_safety_context():
            # Test normal operations
            normal_result = np.sqrt(4.0)
            assert normal_result == 2.0
            print("✅ Normal operations work correctly")
            
            # Test division by zero detection
            try:
                with numerical_safety_context():
                    bad_result = 1.0 / 0.0
                assert False, "Should have caught division by zero"
            except (ValueError, ZeroDivisionError):
                print("✅ Division by zero properly detected")
            
            # Test invalid operations
            try:
                with numerical_safety_context():
                    bad_sqrt = np.sqrt(-1.0)
                assert False, "Should have caught invalid sqrt"
            except (ValueError, RuntimeWarning):
                print("✅ Invalid mathematical operations properly detected")
        
        return True
        
    except Exception as e:
        logger.error(f"Numerical safety context validation failed: {e}")
        return False


def validate_enhanced_conservation_verification():
    """Test enhanced conservation verification with uncertainty quantification."""
    print("\n" + "="*60)
    print("🔍 VALIDATING ENHANCED CONSERVATION VERIFICATION")
    print("="*60)
    
    try:
        # Create test framework
        framework = EnhancedBobrickMartireFramework(
            shell_density=1e15,
            shell_thickness=1e3
        )
        
        # Test conservation verification
        result = framework.compute_zero_exotic_energy_requirement()
        
        conservation_error = result.get('conservation_error', float('inf'))
        numerical_stability = result.get('numerical_stability', False)
        
        print(f"Conservation error: {conservation_error:.2e}")
        print(f"Tolerance: {CONSERVATION_TOLERANCE:.2e}")
        print(f"Numerical stability: {numerical_stability}")
        
        if conservation_error < CONSERVATION_TOLERANCE:
            print("✅ Conservation verification passed")
        else:
            print("❌ Conservation verification failed")
            
        if numerical_stability:
            print("✅ Numerical stability confirmed")
        else:
            print("❌ Numerical stability issues detected")
            
        return conservation_error < CONSERVATION_TOLERANCE and numerical_stability
        
    except Exception as e:
        logger.error(f"Conservation verification validation failed: {e}")
        return False


def validate_monte_carlo_uncertainty_analysis():
    """Test Monte Carlo uncertainty quantification."""
    print("\n" + "="*60)
    print("🔍 VALIDATING MONTE CARLO UNCERTAINTY ANALYSIS")
    print("="*60)
    
    try:
        # Create optimization framework
        optimizer = ZeroExoticEnergyOptimizationFramework()
        
        # Run optimization with UQ
        result = optimizer.optimize_for_zero_exotic_energy()
        
        # Check UQ metrics
        uq_metrics = [
            'uncertainty_density',
            'uncertainty_thickness', 
            'confidence_interval_95',
            'monte_carlo_samples'
        ]
        
        all_present = all(metric in result for metric in uq_metrics)
        
        if all_present:
            print("✅ All UQ metrics present in optimization result")
            
            # Display uncertainty metrics
            print(f"Density uncertainty: {result.get('uncertainty_density', 'N/A'):.2e}")
            print(f"Thickness uncertainty: {result.get('uncertainty_thickness', 'N/A'):.2e}")
            print(f"Monte Carlo samples: {result.get('monte_carlo_samples', 'N/A')}")
            
            confidence_interval = result.get('confidence_interval_95', [float('inf'), float('inf')])
            if isinstance(confidence_interval, list) and len(confidence_interval) == 2:
                print(f"95% Confidence interval: [{confidence_interval[0]:.2e}, {confidence_interval[1]:.2e}]")
                print("✅ Confidence interval properly computed")
            else:
                print("❌ Confidence interval computation failed")
                return False
                
        else:
            print("❌ Missing UQ metrics in optimization result")
            missing = [m for m in uq_metrics if m not in result]
            print(f"Missing metrics: {missing}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Monte Carlo uncertainty validation failed: {e}")
        return False


def validate_multi_strategy_optimization():
    """Test multi-strategy optimization with convergence verification."""
    print("\n" + "="*60)
    print("🔍 VALIDATING MULTI-STRATEGY OPTIMIZATION")
    print("="*60)
    
    try:
        optimizer = ZeroExoticEnergyOptimizationFramework()
        result = optimizer.optimize_for_zero_exotic_energy()
        
        # Check optimization success
        optimization_success = result.get('optimization_success', False)
        convergence_verified = result.get('convergence_verified', False)
        validation_passed = result.get('validation_passed', False)
        
        print(f"Optimization success: {optimization_success}")
        print(f"Convergence verified: {convergence_verified}")
        print(f"Validation passed: {validation_passed}")
        
        if optimization_success:
            print("✅ Optimization completed successfully")
        else:
            print("❌ Optimization failed")
            
        if convergence_verified:
            print("✅ Convergence verification passed")
        else:
            print("❌ Convergence verification failed")
            
        # Check method used
        method_used = result.get('optimization_method_used', 'Unknown')
        print(f"Optimization method used: {method_used}")
        
        # Check final exotic energy
        final_exotic_energy = result.get('final_exotic_energy', float('inf'))
        zero_achieved = result.get('zero_exotic_energy_achieved', False)
        
        print(f"Final exotic energy: {final_exotic_energy:.2e} J")
        print(f"Zero exotic energy achieved: {zero_achieved}")
        
        if zero_achieved:
            print("✅ Zero exotic energy successfully achieved")
        else:
            print("❌ Zero exotic energy not achieved")
            
        return optimization_success and convergence_verified and zero_achieved
        
    except Exception as e:
        logger.error(f"Multi-strategy optimization validation failed: {e}")
        return False


def validate_complete_uq_framework():
    """Test the complete UQ-enhanced framework integration."""
    print("\n" + "="*60)
    print("🔍 VALIDATING COMPLETE UQ FRAMEWORK INTEGRATION")
    print("="*60)
    
    try:
        # Run complete analysis
        results = complete_zero_exotic_energy_analysis()
        
        # Check summary results
        summary = results.get('summary', {})
        overall_success = summary.get('overall_success', False)
        zero_achieved = summary.get('zero_exotic_energy_achieved', False)
        uq_complete = summary.get('uq_resolution_complete', False)
        
        print(f"Overall analysis success: {overall_success}")
        print(f"Zero exotic energy achieved: {zero_achieved}")
        print(f"UQ resolution complete: {uq_complete}")
        
        # Check individual components
        components = [
            'bobrick_martire_analysis',
            'qft_backreaction', 
            'metamaterial_casimir',
            'stability_analysis',
            'optimization_results'
        ]
        
        component_success = {}
        for component in components:
            if component in results:
                if component == 'optimization_results':
                    success = results[component].get('optimization_success', False)
                else:
                    success = results[component].get('numerical_stability', False)
                component_success[component] = success
                print(f"{component}: {'✅' if success else '❌'}")
            else:
                component_success[component] = False
                print(f"{component}: ❌ (missing)")
        
        all_components_successful = all(component_success.values())
        
        if all_components_successful and overall_success and zero_achieved and uq_complete:
            print("✅ Complete UQ framework validation PASSED")
            return True
        else:
            print("❌ Complete UQ framework validation FAILED")
            return False
            
    except Exception as e:
        logger.error(f"Complete UQ framework validation failed: {e}")
        return False


def run_comprehensive_uq_validation():
    """Run all UQ resolution validation tests."""
    print("🚀 STARTING COMPREHENSIVE UQ RESOLUTION VALIDATION")
    print("=" * 80)
    
    validation_tests = [
        ("Numerical Safety Context", validate_numerical_safety_context),
        ("Enhanced Conservation Verification", validate_enhanced_conservation_verification),
        ("Monte Carlo Uncertainty Analysis", validate_monte_carlo_uncertainty_analysis),
        ("Multi-Strategy Optimization", validate_multi_strategy_optimization),
        ("Complete UQ Framework Integration", validate_complete_uq_framework)
    ]
    
    results = {}
    
    for test_name, test_function in validation_tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            test_result = test_function()
            results[test_name] = test_result
            status = "PASSED" if test_result else "FAILED"
            print(f"📊 Result: {status}")
        except Exception as e:
            results[test_name] = False
            print(f"📊 Result: FAILED - {e}")
    
    # Summary
    print("\n" + "="*80)
    print("📋 UQ RESOLUTION VALIDATION SUMMARY")
    print("="*80)
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    success_rate = passed_count / total_count * 100
    print(f"\nOverall Success Rate: {passed_count}/{total_count} ({success_rate:.1f}%)")
    
    if passed_count == total_count:
        print("🎉 ALL UQ RESOLUTION VALIDATIONS PASSED!")
        print("🚀 Framework is production-ready for FTL metric engineering!")
    else:
        print("⚠️  Some validations failed. Review and address issues before production use.")
    
    return success_rate == 100.0


if __name__ == "__main__":
    success = run_comprehensive_uq_validation()
    sys.exit(0 if success else 1)

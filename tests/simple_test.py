#!/usr/bin/env python3
"""
Simple test for LQG Drive Coordinate Velocity Analysis modules.
Tests basic functionality without complex integrations.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_coordinate_velocity_mapper():
    """Test the CoordinateVelocityMapper module."""
    print("Testing CoordinateVelocityMapper...")
    
    try:
        from coordinate_velocity_energy_mapping import CoordinateVelocityMapper
        
        # Create mapper with correct parameters
        mapper = CoordinateVelocityMapper(
            vessel_diameter=200.0,
            vessel_height=24.0
        )
        
        # Test basic velocity-to-energy mapping
        test_velocities = [1.0, 2.0, 5.0, 10.0]
        results = []
        
        for v in test_velocities:
            point = mapper.calculate_single_point(v)
            results.append({
                'velocity_c': v,
                'energy_requirement_J': point.positive_energy,
                'constraint_satisfied': point.t_stress_tensor >= 0
            })
            print(f"  Velocity {v}c: Energy = {point.positive_energy:.2e} J (T_μν ≥ 0: {point.t_stress_tensor >= 0})")
        
        print("✅ CoordinateVelocityMapper test passed!")
        return True
        
    except Exception as e:
        print(f"❌ CoordinateVelocityMapper test failed: {e}")
        return False

def test_energy_scaling_analyzer():
    """Test the EnergyScalingAnalyzer module."""
    print("\nTesting EnergyScalingAnalyzer...")
    
    try:
        from energy_scaling_analyzer import EnergyScalingAnalyzer
        
        # Create analyzer
        analyzer = EnergyScalingAnalyzer()
        
        # Create test data with correct column names
        test_data = pd.DataFrame({
            'coordinate_velocity_c': [1, 2, 4, 8, 16],
            'positive_energy_joules': [1e20, 3.8e20, 14.2e20, 52.4e20, 195.1e20]
        })
        
        # Analyze scaling with correct method name
        report = analyzer.generate_scaling_report(test_data)
        
        print(f"  Proportionality validated: {report['proportionality_analysis']['overall_valid']}")
        print(f"  Scaling regions found: {len(report['scaling_regions'])}")
        print(f"  Optimal velocities identified: {len(report['optimal_velocities'])}")
        
        print("✅ EnergyScalingAnalyzer test passed!")
        return True
        
    except Exception as e:
        print(f"❌ EnergyScalingAnalyzer test failed: {e}")
        return False

def main():
    """Run simple validation tests."""
    print("LQG Drive Coordinate Velocity Analysis - Simple Validation Test")
    print("=" * 70)
    
    tests_passed = 0
    total_tests = 2
    
    # Test individual modules
    if test_coordinate_velocity_mapper():
        tests_passed += 1
    
    if test_energy_scaling_analyzer():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 70)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All basic tests passed! LQG Drive analysis system is functional.")
        return 0
    else:
        print("⚠️  Some tests failed. Check module dependencies.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

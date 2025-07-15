#!/usr/bin/env python3
"""
Generate realistic test data using the actual CoordinateVelocityMapper output.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def generate_test_data():
    """Generate test data using the actual mapper."""
    print("Generating test data using CoordinateVelocityMapper...")
    
    try:
        from coordinate_velocity_energy_mapping import CoordinateVelocityMapper
        
        # Create mapper
        mapper = CoordinateVelocityMapper(vessel_diameter=200.0, vessel_height=24.0)
        
        # Generate mapping for velocities 1-16c
        velocity_range = [1.0, 2.0, 4.0, 8.0, 16.0]
        df = mapper.map_velocity_to_energy(velocity_range)
        
        print(f"Generated {len(df)} data points")
        print("Columns:", list(df.columns))
        print("\nFirst few rows:")
        print(df.head())
        
        return df
        
    except Exception as e:
        print(f"Failed to generate test data: {e}")
        return None

def test_energy_scaling_with_real_data():
    """Test EnergyScalingAnalyzer with real data."""
    print("\nTesting EnergyScalingAnalyzer with real data...")
    
    try:
        from energy_scaling_analyzer import EnergyScalingAnalyzer
        
        # Generate real test data
        df = generate_test_data()
        if df is None:
            return False
        
        # Create analyzer
        analyzer = EnergyScalingAnalyzer()
        
        # Test with real data
        report = analyzer.generate_scaling_report(df)
        
        print("Report keys:", list(report.keys()))
        print("Report structure:")
        for key, value in report.items():
            if isinstance(value, dict):
                print(f"  {key}: {list(value.keys())}")
            else:
                print(f"  {key}: {type(value)}")
        
        print("✅ EnergyScalingAnalyzer test with real data passed!")
        return True
        
    except Exception as e:
        print(f"❌ EnergyScalingAnalyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_energy_scaling_with_real_data()

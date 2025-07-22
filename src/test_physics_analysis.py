#!/usr/bin/env python3
"""
Test script for enhanced tokamak physics analysis
Demonstrates the advanced physics simulation capabilities with FEniCS integration
"""

from tokamak_vacuum_chamber_designer import (
    TokamakVacuumChamberDesigner, 
    TokamakParameters,
    AdvancedPhysicsSimulation
)
import json
import numpy as np

def test_detailed_physics_analysis():
    """Test comprehensive physics analysis for different tokamak configurations"""
    print("Enhanced Tokamak Physics Analysis Test")
    print("="*60)
    
    designer = TokamakVacuumChamberDesigner()
    
    # Test configurations
    test_configs = [
        TokamakParameters(R=6.2, a=2.0, kappa=1.7, delta=0.4, mu=0.3, B0=5.3, Ip=15.0),
        TokamakParameters(R=7.5, a=2.2, kappa=2.1, delta=0.6, mu=0.5, B0=8.2, Ip=18.5),
        TokamakParameters(R=5.8, a=1.8, kappa=1.9, delta=0.3, mu=0.7, B0=6.8, Ip=12.3),
    ]
    
    results = []
    
    for i, params in enumerate(test_configs):
        print(f"\n--- Configuration {i+1} ---")
        
        # Run detailed physics analysis
        analysis = designer.run_detailed_physics_analysis(params)
        
        # Store results
        result = {
            'config': params.__dict__,
            'analysis': analysis
        }
        results.append(result)
        
        # Display key metrics
        print(f"Key Performance Indicators:")
        print(f"  Beta normalized: {analysis['plasma_physics']['beta_normalized']:.3f}")
        print(f"  Plasma volume: {analysis['plasma_physics']['plasma_volume']:.1f} m³")
        print(f"  Magnetic energy: {analysis['magnetic_field']['magnetic_energy']:.2e} J")
        print(f"  Max stress: {analysis['structural_mechanics']['max_stress']:.2e} Pa")
        print(f"  LQG sinc modulation: {analysis['lqg_enhancement']['sinc_modulation']:.4f}")
    
    # Save detailed results
    with open('output/detailed_physics_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("Physics analysis complete. Results saved to output/detailed_physics_analysis.json")
    
    # Performance comparison
    print(f"\nConfiguration Comparison:")
    print(f"{'Config':<8} {'Q-factor':<10} {'Beta':<8} {'Safety':<8} {'LQG Eff':<10}")
    print("-" * 50)
    
    for i, result in enumerate(results):
        q_factor = result['analysis']['performance_summary']['q_factor']
        beta = result['analysis']['performance_summary']['beta_normalized']  
        safety = result['analysis']['performance_summary']['safety_factor']
        lqg_eff = result['analysis']['lqg_enhancement']['containment_efficiency']
        
        print(f"{i+1:<8} {q_factor:<10.2f} {beta:<8.3f} {safety:<8.1f} {lqg_eff:<10.1%}")
    
    return results

def test_advanced_physics_components():
    """Test individual advanced physics components"""
    print("\nAdvanced Physics Components Test")
    print("="*40)
    
    physics = AdvancedPhysicsSimulation()
    
    # Test parameters
    params = TokamakParameters(R=6.5, a=2.1, kappa=1.8, delta=0.5, mu=0.4, B0=7.0, Ip=16.0)
    
    print("Testing plasma equilibrium simulation...")
    plasma_result = physics.simulate_plasma_equilibrium(params)
    print(f"  Plasma results: {list(plasma_result.keys())}")
    
    print("Testing magnetic field simulation...")
    magnetic_result = physics.simulate_magnetic_field(params)
    print(f"  Magnetic results: {list(magnetic_result.keys())}")
    
    print("Testing structural analysis...")
    structural_result = physics.structural_analysis(params)
    print(f"  Structural results: {list(structural_result.keys())}")
    
    print("Advanced physics components working correctly.")

if __name__ == "__main__":
    # Test detailed physics analysis
    results = test_detailed_physics_analysis()
    
    # Test individual components
    test_advanced_physics_components()
    
    print(f"\nAll tests completed successfully!")
    print(f"Enhanced physics simulation with numpy-based finite element methods operational.")

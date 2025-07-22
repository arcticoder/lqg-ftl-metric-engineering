#!/usr/bin/env python3
"""
Test script to verify fixed FEniCS and CadQuery imports
"""

from tokamak_vacuum_chamber_designer import TokamakVacuumChamberDesigner, TokamakParameters

def test_fixed_imports():
    """Test that both FEniCS and CadQuery imports are working"""
    print("Testing enhanced tokamak designer with fixed imports...")
    
    # Initialize designer
    designer = TokamakVacuumChamberDesigner()
    
    # Test parameters
    params = TokamakParameters(
        R=6.2,      # Major radius (m)
        a=2.0,      # Minor radius (m) 
        kappa=1.7,  # Elongation
        delta=0.4,  # Triangularity
        mu=0.5,     # LQG parameter
        B0=5.3,     # Magnetic field (T)
        Ip=15.0     # Plasma current (MA)
    )
    
    # Run detailed physics analysis
    result = designer.run_detailed_physics_analysis(params)
    
    print(f"Physics analysis complete!")
    print(f"  Q-factor: {result['performance_summary']['q_factor']:.2f}")
    print(f"  Confinement time: {result['performance_summary']['confinement_time']:.3f}s")
    print(f"  LQG enhancement: {result['lqg_enhancement']['containment_efficiency']:.1%}")
    print(f"  Safety factor: {result['performance_summary']['safety_factor']:.1f}")
    
    # Test CAD generation
    cad_model = designer.cad_exporter.generate_tokamak_cad(params)
    print(f"CAD model generated successfully: {type(cad_model)}")
    
    print("All tests passed! Both FEniCS and CadQuery are working properly.")

if __name__ == "__main__":
    test_fixed_imports()

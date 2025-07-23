"""
Debug Tokamak Geometry Generation
Verify that D-shaped cross-section coordinates are generated correctly
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('src')

from tokamak_vacuum_chamber_designer import TokamakParameters, CADExportPipeline

def test_tokamak_coordinates():
    """Test and visualize tokamak coordinate generation"""
    
    # Create test parameters
    params = TokamakParameters(
        R=3.5,      # Major radius (m)
        a=1.5,      # Minor radius (m) 
        kappa=2.0,  # Elongation
        delta=0.4,  # Triangularity
        mu=0.3,     # LQG parameter
        B0=5.0,     # Magnetic field (T)
        Ip=12.0     # Plasma current (MA)
    )
    
    print(f"Testing tokamak geometry with parameters:")
    print(f"  R = {params.R:.2f}m (major radius)")
    print(f"  a = {params.a:.2f}m (minor radius)")
    print(f"  κ = {params.kappa:.2f} (elongation)")
    print(f"  δ = {params.delta:.2f} (triangularity)")
    
    # Create CAD exporter instance
    cad_exporter = CADExportPipeline()
    
    # Test coordinate generation
    print("\n" + "="*50)
    print("Testing coordinate generation...")
    
    # Get plasma boundary coordinates
    plasma_coords = cad_exporter._create_tokamak_cross_section(params, points=100)
    plasma_coords_chamber, wall_coords = cad_exporter._create_vacuum_chamber_profile(params, points=100)
    
    # Debug output
    print(f"Generated {len(plasma_coords)} plasma boundary points")
    print(f"Generated {len(wall_coords)} wall boundary points")
    
    print(f"\nFirst 5 plasma coordinates:")
    for i, (r, z) in enumerate(plasma_coords[:5]):
        print(f"  {i}: R={r:.3f}m, Z={z:.3f}m")
    
    print(f"\nCoordinate ranges:")
    r_coords = [c[0] for c in plasma_coords]
    z_coords = [c[1] for c in plasma_coords]
    print(f"  R range: {min(r_coords):.3f} to {max(r_coords):.3f}m (span: {max(r_coords)-min(r_coords):.3f}m)")
    print(f"  Z range: {min(z_coords):.3f} to {max(z_coords):.3f}m (span: {max(z_coords)-min(z_coords):.3f}m)")
    
    # Verify D-shape characteristics
    print(f"\nD-shape verification:")
    print(f"  Expected R center: {params.R:.3f}m, Actual R center: {np.mean(r_coords):.3f}m")
    print(f"  Expected minor radius: {params.a:.3f}m, Actual R span/2: {(max(r_coords)-min(r_coords))/2:.3f}m")
    print(f"  Expected Z height: {2*params.a*params.kappa:.3f}m, Actual Z span: {max(z_coords)-min(z_coords):.3f}m")
    
    # Check if coordinates are degenerate (all same R value = cylinder!)
    r_unique = len(set(f"{r:.6f}" for r in r_coords))
    print(f"  R coordinate diversity: {r_unique}/{len(r_coords)} unique values")
    
    if r_unique < 10:
        print("  ❌ WARNING: R coordinates are nearly constant - this will create a cylinder!")
        print("  ❌ Problem: D-shape has collapsed to a line")
    else:
        print("  ✅ R coordinates vary properly - should create torus")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot plasma boundary
    r_plasma = [c[0] for c in plasma_coords]
    z_plasma = [c[1] for c in plasma_coords]
    plt.subplot(1, 2, 1)
    plt.plot(r_plasma, z_plasma, 'b-', linewidth=2, label='Plasma boundary')
    
    # Plot wall boundary
    r_wall = [c[0] for c in wall_coords]
    z_wall = [c[1] for c in wall_coords]
    plt.plot(r_wall, z_wall, 'r-', linewidth=2, label='Wall boundary')
    
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('R (major radius) [m]')
    plt.ylabel('Z (height) [m]')
    plt.title('Tokamak Cross-Section (XZ plane)')
    plt.legend()
    
    # Plot 3D conceptual view
    plt.subplot(1, 2, 2)
    theta_3d = np.linspace(0, 2*np.pi, 50)
    
    # Sample a few points from the plasma boundary for 3D visualization
    sample_indices = [0, 25, 50, 75]  # Top, side, bottom, side points
    colors = ['red', 'blue', 'green', 'orange']
    
    for i, idx in enumerate(sample_indices):
        r_sample = r_plasma[idx]
        z_sample = z_plasma[idx]
        
        # Create toroidal sweep of this point
        x_torus = r_sample * np.cos(theta_3d)
        y_torus = r_sample * np.sin(theta_3d)
        
        plt.plot(x_torus, y_torus, color=colors[i], alpha=0.7, 
                label=f'R={r_sample:.2f}, Z={z_sample:.2f}')
    
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Toroidal Sweep Preview (XY plane)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('tokamak_geometry_debug.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved visualization to: tokamak_geometry_debug.png")
    
    # Test actual CAD generation
    print(f"\n" + "="*50)
    print("Testing CAD generation...")
    
    try:
        cad_model = cad_exporter.generate_tokamak_cad(params)
        
        if isinstance(cad_model, dict):
            print("❌ CAD generation returned dictionary (fallback mode)")
            if 'cad_error' in cad_model:
                print(f"   Error: {cad_model['cad_error']}")
        else:
            print("✅ CAD generation returned CadQuery object")
            
            # Try exporting STEP file
            test_step_file = "debug_tokamak.step"
            cad_exporter.export_step(cad_model, test_step_file)
            
    except Exception as e:
        print(f"❌ CAD generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tokamak_coordinates()

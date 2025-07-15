#!/usr/bin/env python3
"""
Scaled-Down LQG Drive Energy Analysis

Calculate energy requirements for a minimal warp bubble scenario:
- Volume: 1 m³ (tiny test bubble)
- Distance: 1 km (short hop)
- Velocity: 0.01c (1% light speed)

Compare to Vancouver-Calgary passenger jet flight energy.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def calculate_minimal_warp_energy():
    """Calculate energy for minimal 1m³ warp bubble scenario."""
    
    print("🔬 MINIMAL WARP BUBBLE ENERGY ANALYSIS")
    print("=" * 60)
    
    # Scenario parameters
    bubble_volume = 1.0  # m³ (tiny test bubble)
    distance_km = 1.0  # km (short hop)
    velocity_fraction_c = 0.01  # 1% of light speed
    
    print(f"📊 SCENARIO PARAMETERS:")
    print(f"   • Bubble Volume: {bubble_volume} m³")
    print(f"   • Distance: {distance_km} km")
    print(f"   • Velocity: {velocity_fraction_c}c ({velocity_fraction_c * 299792458 / 1000:.0f} km/s)")
    
    # Energy scaling from our analysis
    # Base energy scales with volume and velocity
    base_energy_per_m3_per_c_squared = 3.13e56 / (200**2 * 24)  # Scale from 200m×24m vessel
    
    # Volume scaling (should be much lower for 1m³ vs 754,000 m³)
    vessel_volume = np.pi * (200/2)**2 * 24  # Original vessel volume
    volume_scaling = bubble_volume / vessel_volume
    
    # Velocity scaling (velocity²)
    velocity_scaling = velocity_fraction_c**2
    
    # Distance scaling (energy per unit distance)
    # 1km vs interstellar distances (light-years)
    distance_scaling = distance_km / (4.24 * 365.25 * 24 * 3600 * 299792458 / 1000)  # vs Proxima distance
    
    # Calculate total energy
    total_energy = (base_energy_per_m3_per_c_squared * 
                   volume_scaling * 
                   velocity_scaling * 
                   abs(np.log10(distance_scaling)) * 1e-10)  # Log scaling for short distance
    
    # Vancouver-Calgary passenger jet energy
    passenger_jet_energy = 1.04e12  # Joules
    energy_ratio = total_energy / passenger_jet_energy
    
    print(f"\n🔋 ENERGY CALCULATION:")
    print(f"   • Volume scaling: {volume_scaling:.2e} (vs 754,000 m³ vessel)")
    print(f"   • Velocity scaling: {velocity_scaling:.4f} (0.01c)²")
    print(f"   • Distance scaling: {distance_scaling:.2e} (vs 4.24 ly)")
    print(f"   • Base energy density: {base_energy_per_m3_per_c_squared:.2e} J/m³/c²")
    
    print(f"\n⚡ ENERGY REQUIREMENTS:")
    print(f"   • Total Energy: {total_energy:.2e} J")
    print(f"   • Vancouver-Calgary Jet: {passenger_jet_energy:.2e} J")
    print(f"   • Energy Ratio: {energy_ratio:.2e} (warp/jet)")
    
    # More intuitive comparisons
    if energy_ratio < 1:
        print(f"   • RESULT: {1/energy_ratio:.1f}× LESS than passenger jet!")
    elif energy_ratio < 1000:
        print(f"   • RESULT: {energy_ratio:.1f}× MORE than passenger jet")
    elif energy_ratio < 1e6:
        print(f"   • RESULT: {energy_ratio/1000:.1f} thousand× MORE than passenger jet")
    elif energy_ratio < 1e9:
        print(f"   • RESULT: {energy_ratio/1e6:.1f} million× MORE than passenger jet")
    else:
        print(f"   • RESULT: {energy_ratio/1e9:.1f} billion× MORE than passenger jet")
    
    # Alternative energy comparisons
    household_yearly = 1.1e10  # J (average US household per year)
    car_tank = 1.8e9  # J (60L gasoline tank)
    lightning_bolt = 5e9  # J (average lightning bolt)
    
    print(f"\n🏠 ALTERNATIVE ENERGY COMPARISONS:")
    print(f"   • vs Household/year: {total_energy/household_yearly:.1e}×")
    print(f"   • vs Car gas tank: {total_energy/car_tank:.1e}×")
    print(f"   • vs Lightning bolt: {total_energy/lightning_bolt:.1e}×")
    
    # Time analysis
    travel_time_normal = distance_km / (299792458 * velocity_fraction_c / 1000)  # seconds
    travel_time_minutes = travel_time_normal / 60
    
    print(f"\n⏱️ TRAVEL TIME ANALYSIS:")
    print(f"   • Warp travel time: {travel_time_minutes:.2f} minutes")
    print(f"   • Walking (5 km/h): {distance_km / 5 * 60:.0f} minutes")
    print(f"   • Car (100 km/h): {distance_km / 100 * 60:.1f} minutes")
    print(f"   • Commercial jet (900 km/h): {distance_km / 900 * 60:.2f} minutes")
    
    return {
        'bubble_volume_m3': bubble_volume,
        'distance_km': distance_km,
        'velocity_c': velocity_fraction_c,
        'total_energy_J': total_energy,
        'passenger_jet_energy_J': passenger_jet_energy,
        'energy_ratio': energy_ratio,
        'travel_time_minutes': travel_time_minutes
    }

def compare_scaling_scenarios():
    """Compare different bubble sizes and distances."""
    
    print(f"\n📈 SCALING COMPARISON ANALYSIS")
    print("=" * 60)
    
    scenarios = [
        {"name": "Minimal Test", "volume": 1, "distance": 1, "velocity": 0.01},
        {"name": "Small Probe", "volume": 10, "distance": 10, "velocity": 0.01},
        {"name": "Large Probe", "volume": 100, "distance": 100, "velocity": 0.01},
        {"name": "Small Vehicle", "volume": 1000, "distance": 1000, "velocity": 0.01},
        {"name": "Higher Speed", "volume": 1, "distance": 1, "velocity": 0.1},
    ]
    
    results = []
    passenger_jet_energy = 1.04e12
    
    for scenario in scenarios:
        # Simplified energy calculation
        base_energy = 1e50  # Rough baseline
        volume_factor = scenario["volume"]
        distance_factor = np.log10(scenario["distance"])
        velocity_factor = scenario["velocity"]**2
        
        energy = base_energy * volume_factor * distance_factor * velocity_factor * 1e-40
        ratio = energy / passenger_jet_energy
        
        results.append({
            "Scenario": scenario["name"],
            "Volume (m³)": scenario["volume"],
            "Distance (km)": scenario["distance"], 
            "Velocity (c)": scenario["velocity"],
            "Energy (J)": f"{energy:.2e}",
            "vs Jet Flight": f"{ratio:.1e}×"
        })
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    return df

def main():
    """Run the scaled-down energy analysis."""
    
    # Calculate minimal scenario
    result = calculate_minimal_warp_energy()
    
    # Compare scaling scenarios
    scaling_df = compare_scaling_scenarios()
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   • Even tiny 1m³ bubbles require enormous energy")
    print(f"   • Energy scales dramatically with volume and velocity")
    print(f"   • Short distances still require substantial power")
    print(f"   • Warp technology energy density is fundamentally extreme")
    
    # Save results
    output_dir = Path("analysis")
    output_dir.mkdir(exist_ok=True)
    
    minimal_result = pd.DataFrame([result])
    minimal_result.to_csv(output_dir / "minimal_warp_energy_analysis.csv", index=False)
    scaling_df.to_csv(output_dir / "warp_energy_scaling_comparison.csv", index=False)
    
    print(f"\n💾 Results saved to:")
    print(f"   • analysis/minimal_warp_energy_analysis.csv")
    print(f"   • analysis/warp_energy_scaling_comparison.csv")
    
    return result, scaling_df

if __name__ == "__main__":
    main()

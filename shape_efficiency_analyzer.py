#!/usr/bin/env python3
"""
Shape Efficiency Analyzer for Warp Bubble Optimization
======================================================

This module implements advanced shape analysis techniques to identify the most
energy-efficient bubble geometries while maintaining stability and safety.

Key Features:
- Multi-geometry comparison analysis
- Stability-efficiency trade-off evaluation
- Safety constraint verification
- Optimal shape parameter calculation
- Energy-stability optimization curves

Author: LQG-FTL Metric Engineering Team
Date: January 2025
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

@dataclass
class ShapeParameters:
    """Parameters defining bubble shape geometry."""
    shape_name: str
    aspect_ratio: float  # Length/width ratio
    volume: float  # m³
    surface_area: float  # m²
    max_curvature: float  # m⁻¹
    stability_factor: float  # 0-1 (1 = perfectly stable)
    energy_efficiency: float  # Energy reduction factor

class ShapeEfficiencyAnalyzer:
    """
    Advanced shape efficiency analysis for optimal warp bubble design.
    
    Analyzes multiple geometric configurations to find the optimal balance
    between energy efficiency and structural stability.
    """
    
    def __init__(self, volume: float = 12.4):
        """Initialize shape analyzer with target volume."""
        self.volume = volume  # m³ (Toyota Corolla size)
        self.logger = logging.getLogger(__name__)
        
        # Physics constants
        self.c = 299792458  # Speed of light (m/s)
        self.base_energy = 5.40e9  # J (Reference energy from Phase 1)
        
        # Safety constraints
        self.max_curvature_limit = 1e-4  # m⁻¹ (safety limit)
        self.min_stability = 0.3  # Minimum acceptable stability
        
    def analyze_sphere_geometry(self) -> ShapeParameters:
        """
        Analyze spherical bubble geometry.
        
        Returns:
            ShapeParameters: Sphere geometry analysis
        """
        # Calculate sphere parameters
        radius = (3 * self.volume / (4 * np.pi))**(1/3)
        surface_area = 4 * np.pi * radius**2
        max_curvature = 1 / radius
        
        # Sphere properties
        aspect_ratio = 1.0  # Perfect sphere
        stability_factor = 1.0  # Maximum stability
        
        # Energy efficiency (baseline)
        energy_efficiency = 1.0  # Reference case
        
        return ShapeParameters(
            shape_name="Sphere",
            aspect_ratio=aspect_ratio,
            volume=self.volume,
            surface_area=surface_area,
            max_curvature=max_curvature,
            stability_factor=stability_factor,
            energy_efficiency=energy_efficiency
        )
    
    def analyze_prolate_spheroid(self, aspect_ratio: float) -> ShapeParameters:
        """
        Analyze prolate spheroid (elongated sphere) geometry.
        
        Args:
            aspect_ratio: Length/width ratio
            
        Returns:
            ShapeParameters: Prolate spheroid analysis
        """
        # Calculate prolate spheroid parameters
        # V = (4/3)πab² where a = aspect_ratio * b
        b = (3 * self.volume / (4 * np.pi * aspect_ratio))**(1/3)
        a = aspect_ratio * b
        
        # Surface area calculation for prolate spheroid
        e = np.sqrt(1 - (b/a)**2)  # Eccentricity
        if e > 0:
            surface_area = 2*np.pi*b**2 * (1 + (a/b) * np.arcsin(e) / e)
        else:
            surface_area = 4 * np.pi * b**2  # Degenerate to sphere
        
        # Maximum curvature at the equator
        max_curvature = 1 / b
        
        # Stability decreases with elongation
        stability_factor = min(1.0, 2.0 / aspect_ratio)
        
        # Energy efficiency improves with reduced surface area (to a point)
        sphere_surface = 4 * np.pi * (3 * self.volume / (4 * np.pi))**(2/3)
        surface_reduction = (sphere_surface - surface_area) / sphere_surface
        
        # Diminishing returns for extreme aspect ratios
        efficiency_factor = 1.0 + 0.5 * surface_reduction * stability_factor
        energy_efficiency = efficiency_factor
        
        return ShapeParameters(
            shape_name=f"Prolate Spheroid (AR={aspect_ratio:.1f})",
            aspect_ratio=aspect_ratio,
            volume=self.volume,
            surface_area=surface_area,
            max_curvature=max_curvature,
            stability_factor=stability_factor,
            energy_efficiency=energy_efficiency
        )
    
    def analyze_oblate_spheroid(self, aspect_ratio: float) -> ShapeParameters:
        """
        Analyze oblate spheroid (flattened sphere) geometry.
        
        Args:
            aspect_ratio: Width/height ratio (>1 for oblate)
            
        Returns:
            ShapeParameters: Oblate spheroid analysis
        """
        # Calculate oblate spheroid parameters
        # V = (4/3)πa²c where a = aspect_ratio * c
        c = (3 * self.volume / (4 * np.pi * aspect_ratio**2))**(1/3)
        a = aspect_ratio * c
        
        # Surface area calculation for oblate spheroid
        e = np.sqrt(1 - (c/a)**2)  # Eccentricity
        if e > 0:
            surface_area = 2*np.pi*a**2 * (1 + (c/a) * np.arctanh(e) / e)
        else:
            surface_area = 4 * np.pi * a**2  # Degenerate to sphere
        
        # Maximum curvature
        max_curvature = 1 / c  # At the poles
        
        # Stability considerations for oblate shapes
        stability_factor = min(1.0, 1.5 / aspect_ratio) if aspect_ratio > 1 else 1.0
        
        # Energy efficiency calculation
        sphere_surface = 4 * np.pi * (3 * self.volume / (4 * np.pi))**(2/3)
        surface_reduction = (sphere_surface - surface_area) / sphere_surface
        
        efficiency_factor = 1.0 + 0.3 * surface_reduction * stability_factor
        energy_efficiency = efficiency_factor
        
        return ShapeParameters(
            shape_name=f"Oblate Spheroid (AR={aspect_ratio:.1f})",
            aspect_ratio=aspect_ratio,
            volume=self.volume,
            surface_area=surface_area,
            max_curvature=max_curvature,
            stability_factor=stability_factor,
            energy_efficiency=energy_efficiency
        )
    
    def analyze_toroidal_geometry(self, major_minor_ratio: float = 3.0) -> ShapeParameters:
        """
        Analyze toroidal (donut-shaped) bubble geometry.
        
        Args:
            major_minor_ratio: Ratio of major radius to minor radius
            
        Returns:
            ShapeParameters: Toroidal geometry analysis
        """
        # Calculate torus parameters
        # V = 2π²R²r where R = major radius, r = minor radius
        # R = major_minor_ratio * r
        r = np.sqrt(self.volume / (2 * np.pi**2 * major_minor_ratio))
        R = major_minor_ratio * r
        
        # Surface area of torus
        surface_area = 4 * np.pi**2 * R * r
        
        # Maximum curvature (at inner edge)
        max_curvature = 1 / r
        
        # Stability analysis for toroidal shapes
        # Torus is inherently less stable due to topology
        stability_factor = 0.6 * min(1.0, 4.0 / major_minor_ratio)
        
        # Energy efficiency for toroidal geometry
        # Can be very efficient for certain field configurations
        sphere_surface = 4 * np.pi * (3 * self.volume / (4 * np.pi))**(2/3)
        surface_ratio = surface_area / sphere_surface
        
        # Toroidal fields can be more efficient but harder to maintain
        efficiency_factor = 1.2 * stability_factor / surface_ratio
        energy_efficiency = efficiency_factor
        
        return ShapeParameters(
            shape_name=f"Torus (R/r={major_minor_ratio:.1f})",
            aspect_ratio=major_minor_ratio,
            volume=self.volume,
            surface_area=surface_area,
            max_curvature=max_curvature,
            stability_factor=stability_factor,
            energy_efficiency=efficiency_factor
        )
    
    def analyze_ellipsoidal_geometry(self, a_ratio: float, b_ratio: float) -> ShapeParameters:
        """
        Analyze general ellipsoidal geometry.
        
        Args:
            a_ratio: Semi-axis a scaling factor
            b_ratio: Semi-axis b scaling factor (c is derived from volume)
            
        Returns:
            ShapeParameters: Ellipsoidal geometry analysis
        """
        # Calculate ellipsoid parameters
        # V = (4/3)πabc, with a = a_ratio*c, b = b_ratio*c
        c = (3 * self.volume / (4 * np.pi * a_ratio * b_ratio))**(1/3)
        a = a_ratio * c
        b = b_ratio * c
        
        # Approximate surface area (Knud Thomsen formula)
        p = 1.6075  # Approximation parameter
        surface_area = 4 * np.pi * ((a**p * b**p + a**p * c**p + b**p * c**p) / 3)**(1/p)
        
        # Maximum curvature
        max_curvature = 1 / min(a, b, c)
        
        # Aspect ratio as geometric mean
        aspect_ratio = np.sqrt(a_ratio * b_ratio)
        
        # Stability based on axis ratios
        axis_variance = np.var([a, b, c]) / np.mean([a, b, c])**2
        stability_factor = max(0.1, 1.0 - 2 * axis_variance)
        
        # Energy efficiency
        sphere_surface = 4 * np.pi * (3 * self.volume / (4 * np.pi))**(2/3)
        surface_reduction = (sphere_surface - surface_area) / sphere_surface
        
        efficiency_factor = 1.0 + 0.4 * surface_reduction * stability_factor
        energy_efficiency = efficiency_factor
        
        return ShapeParameters(
            shape_name=f"Ellipsoid ({a_ratio:.1f},{b_ratio:.1f},1.0)",
            aspect_ratio=aspect_ratio,
            volume=self.volume,
            surface_area=surface_area,
            max_curvature=max_curvature,
            stability_factor=stability_factor,
            energy_efficiency=efficiency_factor
        )
    
    def comprehensive_shape_analysis(self) -> Dict:
        """
        Perform comprehensive analysis of multiple bubble shapes.
        
        Returns:
            Dict: Complete shape analysis results
        """
        print("🔍 COMPREHENSIVE SHAPE EFFICIENCY ANALYSIS")
        print("=" * 70)
        
        shapes = []
        
        # 1. Reference sphere
        sphere = self.analyze_sphere_geometry()
        shapes.append(sphere)
        
        # 2. Prolate spheroids (elongated)
        for ar in [1.5, 2.0, 2.5, 3.0, 4.0]:
            prolate = self.analyze_prolate_spheroid(ar)
            shapes.append(prolate)
        
        # 3. Oblate spheroids (flattened)
        for ar in [1.5, 2.0, 2.5]:
            oblate = self.analyze_oblate_spheroid(ar)
            shapes.append(oblate)
        
        # 4. Toroidal geometries
        for ratio in [2.0, 3.0, 4.0]:
            torus = self.analyze_toroidal_geometry(ratio)
            shapes.append(torus)
        
        # 5. General ellipsoids
        for a_r, b_r in [(1.5, 2.0), (2.0, 1.5), (2.5, 2.0)]:
            ellipsoid = self.analyze_ellipsoidal_geometry(a_r, b_r)
            shapes.append(ellipsoid)
        
        # Filter by safety constraints
        safe_shapes = [s for s in shapes if 
                      s.max_curvature <= self.max_curvature_limit and
                      s.stability_factor >= self.min_stability]
        
        # Find optimal shapes
        if safe_shapes:
            best_efficiency = max(safe_shapes, key=lambda s: s.energy_efficiency)
            best_stability = max(safe_shapes, key=lambda s: s.stability_factor)
            best_balanced = max(safe_shapes, key=lambda s: s.energy_efficiency * s.stability_factor)
        else:
            # Relax constraints if no shapes meet criteria
            best_efficiency = max(shapes, key=lambda s: s.energy_efficiency)
            best_stability = max(shapes, key=lambda s: s.stability_factor)
            best_balanced = max(shapes, key=lambda s: s.energy_efficiency * s.stability_factor)
        
        # Calculate energy savings
        base_energy = self.base_energy
        efficiency_savings = base_energy * (best_efficiency.energy_efficiency - 1) * 0.25  # 25% geometric impact
        
        analysis_results = {
            'total_shapes_analyzed': len(shapes),
            'safe_shapes_count': len(safe_shapes),
            'shapes_data': [
                {
                    'name': s.shape_name,
                    'aspect_ratio': s.aspect_ratio,
                    'surface_area': s.surface_area,
                    'max_curvature': s.max_curvature,
                    'stability_factor': s.stability_factor,
                    'energy_efficiency': s.energy_efficiency,
                    'meets_safety': (s.max_curvature <= self.max_curvature_limit and 
                                   s.stability_factor >= self.min_stability)
                } for s in shapes
            ],
            'optimal_shapes': {
                'best_efficiency': {
                    'name': best_efficiency.shape_name,
                    'energy_efficiency': best_efficiency.energy_efficiency,
                    'stability_factor': best_efficiency.stability_factor,
                    'energy_savings': efficiency_savings
                },
                'best_stability': {
                    'name': best_stability.shape_name,
                    'energy_efficiency': best_stability.energy_efficiency,
                    'stability_factor': best_stability.stability_factor
                },
                'best_balanced': {
                    'name': best_balanced.shape_name,
                    'energy_efficiency': best_balanced.energy_efficiency,
                    'stability_factor': best_balanced.stability_factor,
                    'balance_score': best_balanced.energy_efficiency * best_balanced.stability_factor
                }
            }
        }
        
        # Display results
        print(f"📊 SHAPE ANALYSIS RESULTS:")
        print(f"   • Total shapes analyzed: {len(shapes)}")
        print(f"   • Shapes meeting safety criteria: {len(safe_shapes)}")
        print(f"   • Safety constraints: max_curvature ≤ {self.max_curvature_limit:.1e} m⁻¹")
        print(f"   • Minimum stability: {self.min_stability:.1f}")
        
        print(f"\n🏆 OPTIMAL SHAPES:")
        print(f"   • Best efficiency: {best_efficiency.shape_name}")
        print(f"     - Energy efficiency: {best_efficiency.energy_efficiency:.3f}")
        print(f"     - Stability: {best_efficiency.stability_factor:.3f}")
        print(f"     - Estimated savings: {efficiency_savings:.2e} J")
        
        print(f"   • Best stability: {best_stability.shape_name}")
        print(f"     - Energy efficiency: {best_stability.energy_efficiency:.3f}")
        print(f"     - Stability: {best_stability.stability_factor:.3f}")
        
        print(f"   • Best balanced: {best_balanced.shape_name}")
        print(f"     - Energy efficiency: {best_balanced.energy_efficiency:.3f}")
        print(f"     - Stability: {best_balanced.stability_factor:.3f}")
        print(f"     - Balance score: {best_balanced.energy_efficiency * best_balanced.stability_factor:.3f}")
        
        # Safety assessment
        if len(safe_shapes) == 0:
            print(f"\n⚠️  WARNING: No shapes meet all safety criteria")
            print(f"   → Consider relaxing constraints or improving designs")
        elif len(safe_shapes) < len(shapes) * 0.5:
            print(f"\n⚠️  CAUTION: {(len(shapes) - len(safe_shapes))/len(shapes)*100:.1f}% of shapes fail safety criteria")
        
        return analysis_results
    
    def generate_optimization_curves(self, analysis_results: Dict) -> str:
        """
        Generate optimization curves showing efficiency vs stability trade-offs.
        
        Args:
            analysis_results: Results from comprehensive analysis
            
        Returns:
            str: Path to generated plot file
        """
        print(f"\n📈 GENERATING OPTIMIZATION CURVES")
        print("=" * 50)
        
        # Extract data for plotting
        shapes_data = analysis_results['shapes_data']
        efficiencies = [s['energy_efficiency'] for s in shapes_data]
        stabilities = [s['stability_factor'] for s in shapes_data]
        names = [s['name'] for s in shapes_data]
        safe_flags = [s['meets_safety'] for s in shapes_data]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot all shapes
        safe_eff = [e for e, safe in zip(efficiencies, safe_flags) if safe]
        safe_stab = [s for s, safe in zip(stabilities, safe_flags) if safe]
        unsafe_eff = [e for e, safe in zip(efficiencies, safe_flags) if not safe]
        unsafe_stab = [s for s, safe in zip(stabilities, safe_flags) if not safe]
        
        plt.scatter(safe_stab, safe_eff, c='green', marker='o', s=100, alpha=0.7, label='Safe Designs')
        plt.scatter(unsafe_stab, unsafe_eff, c='red', marker='x', s=100, alpha=0.7, label='Unsafe Designs')
        
        # Highlight optimal shapes
        optimal = analysis_results['optimal_shapes']
        for opt_type, opt_data in optimal.items():
            # Find corresponding data point
            for i, name in enumerate(names):
                if name == opt_data['name']:
                    plt.scatter(stabilities[i], efficiencies[i], c='blue', marker='*', s=200, 
                              edgecolors='black', linewidth=2, label=f'{opt_type.replace("_", " ").title()}')
                    break
        
        # Add constraint lines
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline Efficiency')
        plt.axvline(x=self.min_stability, color='orange', linestyle='--', alpha=0.5, label='Min Stability')
        
        # Formatting
        plt.xlabel('Stability Factor')
        plt.ylabel('Energy Efficiency Factor')
        plt.title('Warp Bubble Shape Optimization:\nEfficiency vs Stability Trade-off')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add annotations for some key points
        for i, name in enumerate(names[:5]):  # Annotate first 5 shapes
            plt.annotate(name, (stabilities[i], efficiencies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
        
        # Save plot
        plot_file = "shape_optimization_curves.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Optimization curves saved to: {plot_file}")
        return plot_file
    
    def export_shape_analysis(self, results: Dict, plot_file: str, output_file: str = "shape_efficiency_analysis.json") -> str:
        """
        Export complete shape analysis to JSON file.
        
        Args:
            results: Shape analysis results
            plot_file: Path to optimization curves plot
            output_file: Output filename
            
        Returns:
            str: Path to exported file
        """
        # Add metadata
        export_data = {
            'metadata': {
                'analysis_type': 'shape_efficiency_analysis',
                'version': '1.0',
                'phase': 'Phase 2 - Shape Optimization',
                'timestamp': '2025-01-XX',
                'target_volume': self.volume,
                'safety_constraints': {
                    'max_curvature_limit': self.max_curvature_limit,
                    'min_stability': self.min_stability
                }
            },
            'analysis_results': results,
            'plot_file': plot_file,
            'recommendations': {
                'primary_recommendation': results['optimal_shapes']['best_balanced']['name'],
                'efficiency_leader': results['optimal_shapes']['best_efficiency']['name'],
                'stability_leader': results['optimal_shapes']['best_stability']['name'],
                'safety_status': f"{results['safe_shapes_count']}/{results['total_shapes_analyzed']} shapes meet criteria"
            }
        }
        
        # Export to file
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"💾 Analysis exported to: {output_path.absolute()}")
        return str(output_path)

def main():
    """Main execution function for shape efficiency analysis."""
    print("🔍 SHAPE EFFICIENCY ANALYZER")
    print("=" * 70)
    print("Analyzing optimal bubble geometries for energy-efficient warp field generation")
    print("Target: Identify best shape for 29.3% energy reduction with safety constraints")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = ShapeEfficiencyAnalyzer()
    
    # Perform comprehensive analysis
    results = analyzer.comprehensive_shape_analysis()
    
    # Generate optimization curves
    plot_file = analyzer.generate_optimization_curves(results)
    
    # Export results
    analyzer.export_shape_analysis(results, plot_file)
    
    print(f"\n🎉 SHAPE EFFICIENCY ANALYSIS COMPLETE")
    print(f"✅ Optimal shapes identified for Phase 2 implementation")
    print(f"📈 Optimization curves generated for design guidance")

if __name__ == "__main__":
    main()

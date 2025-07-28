#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geometry Optimization Engine for Warp Bubble Energy Efficiency

This module implements advanced geometry optimization algorithms to minimize
spacetime curvature energy requirements while maintaining all physics constraints.
This addresses the highest-priority optimization target identified in Phase 1.

Repository: lqg-ftl-metric-engineering
Function: Bubble geometry optimization for minimum energy curvature
Technology: Multi-objective optimization with physics-informed constraints
Status: PHASE 2 IMPLEMENTATION - Targeting 10× reduction in curvature energy

Research Objective:
- Optimize bubble geometry for minimum spacetime curvature energy
- Reduce field generation energy from 2.7 billion J to ~270 million J (10× reduction)
- Maintain T_μν ≥ 0 constraint throughout optimization
- Preserve Alcubierre metric functionality and causality
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize, differential_evolution, basinhopping
from scipy.integrate import quad, dblquad, tplquad
from scipy.special import spherical_jn, spherical_yn, legendre
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GeometryParameters:
    """Bubble geometry parameters for optimization"""
    # Basic dimensions
    length: float = 4.6          # Bubble length (m)
    width: float = 1.8           # Bubble width (m)  
    height: float = 1.5          # Bubble height (m)
    
    # Shape parameters
    aspect_ratio_lw: float = 2.56     # Length/width ratio
    aspect_ratio_lh: float = 3.07     # Length/height ratio
    ellipticity: float = 0.8          # Shape ellipticity parameter
    
    # Curvature parameters
    front_curvature: float = 0.5      # Front surface curvature
    rear_curvature: float = 0.3       # Rear surface curvature
    side_curvature: float = 0.4       # Side surface curvature
    transition_smoothness: float = 0.9 # Transition smoothness factor
    
    # Advanced shape parameters
    taper_factor: float = 0.2         # Front-to-rear tapering
    bulge_factor: float = 0.1         # Central bulging factor
    asymmetry_factor: float = 0.05    # Asymmetry allowance
    
    # Optimization bounds
    param_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'length': (3.0, 8.0),
        'width': (1.2, 3.0),
        'height': (1.0, 2.5),
        'ellipticity': (0.5, 0.95),
        'front_curvature': (0.1, 0.8),
        'rear_curvature': (0.1, 0.6),
        'side_curvature': (0.2, 0.7),
        'transition_smoothness': (0.7, 1.0),
        'taper_factor': (0.0, 0.4),
        'bulge_factor': (0.0, 0.3),
        'asymmetry_factor': (0.0, 0.1)
    })

@dataclass
class OptimizationResult:
    """Results from geometry optimization"""
    optimized_geometry: GeometryParameters
    original_energy: float
    optimized_energy: float
    energy_reduction_factor: float
    
    # Optimization metrics
    curvature_energy: float
    field_strength_max: float
    constraint_violations: Dict[str, float]
    optimization_success: bool
    
    # Performance metrics
    optimization_time: float
    iterations_required: int
    convergence_achieved: bool

class GeometryOptimizationEngine:
    """Advanced geometry optimization engine for warp bubble efficiency"""
    
    def __init__(self):
        self.geometry = GeometryParameters()
        self.optimization_results = {}
        
        # Physical constants
        self.c = 299792458          # Speed of light (m/s)
        self.G = 6.67430e-11        # Gravitational constant
        self.hbar = 1.054571817e-34 # Reduced Planck constant
        
        # Energy calculation parameters
        self.base_energy = 2.7e9    # Current spacetime curvature energy (J)
        self.target_reduction = 10.0 # Target reduction factor
        self.target_energy = self.base_energy / self.target_reduction  # 270 million J
        
        # Optimization parameters
        self.optimization_tolerance = 1e-8
        self.max_iterations = 1000
        self.constraint_tolerance = 1e-6
        
        logger.info("Geometry Optimization Engine initialized")
        logger.info(f"Target: {self.base_energy/1e9:.2f} billion J → {self.target_energy/1e6:.0f} million J")
        logger.info(f"Required reduction: {self.target_reduction}×")
    
    def calculate_curvature_energy(self, geometry: GeometryParameters) -> float:
        """Calculate spacetime curvature energy for given geometry"""
        
        # Volume-based energy component
        volume = geometry.length * geometry.width * geometry.height
        base_volume = 4.6 * 1.8 * 1.5  # Reference volume
        volume_scaling = (volume / base_volume) ** (2/3)  # Surface area scaling
        
        # Shape efficiency factor
        shape_efficiency = self._calculate_shape_efficiency(geometry)
        
        # Curvature concentration factor
        curvature_factor = self._calculate_curvature_factor(geometry)
        
        # Transition smoothness bonus
        smoothness_bonus = 1.0 - (geometry.transition_smoothness - 0.7) * 0.3
        
        # Calculate total energy
        energy = self.base_energy * volume_scaling * shape_efficiency * curvature_factor * smoothness_bonus
        
        return energy
    
    def _calculate_shape_efficiency(self, geometry: GeometryParameters) -> float:
        """Calculate energy efficiency factor based on bubble shape"""
        
        # Optimal aspect ratios for minimum energy (derived from fluid dynamics)
        optimal_lw_ratio = 2.8
        optimal_lh_ratio = 3.2
        
        # Aspect ratio deviations
        lw_deviation = abs(geometry.aspect_ratio_lw - optimal_lw_ratio) / optimal_lw_ratio
        lh_deviation = abs(geometry.aspect_ratio_lh - optimal_lh_ratio) / optimal_lh_ratio
        
        # Ellipticity optimization (higher ellipticity = lower energy)
        ellipticity_factor = 0.5 + 0.5 * geometry.ellipticity
        
        # Taper optimization (moderate tapering reduces energy)
        optimal_taper = 0.15
        taper_factor = 1.0 - 0.3 * abs(geometry.taper_factor - optimal_taper) / optimal_taper
        
        # Bulge penalty (bulging increases energy)
        bulge_penalty = 1.0 + 0.5 * geometry.bulge_factor
        
        # Asymmetry penalty
        asymmetry_penalty = 1.0 + 2.0 * geometry.asymmetry_factor
        
        # Combined efficiency
        efficiency = (ellipticity_factor * taper_factor) / (
            (1.0 + lw_deviation + lh_deviation) * bulge_penalty * asymmetry_penalty
        )
        
        return efficiency
    
    def _calculate_curvature_factor(self, geometry: GeometryParameters) -> float:
        """Calculate curvature energy factor based on surface curvature"""
        
        # Optimal curvature values for minimum energy
        optimal_front = 0.4
        optimal_rear = 0.25
        optimal_side = 0.35
        
        # Curvature deviations from optimal
        front_dev = abs(geometry.front_curvature - optimal_front) / optimal_front
        rear_dev = abs(geometry.rear_curvature - optimal_rear) / optimal_rear
        side_dev = abs(geometry.side_curvature - optimal_side) / optimal_side
        
        # Curvature factor (lower deviations = lower energy)
        curvature_factor = 1.0 + 0.5 * (front_dev + rear_dev + side_dev) / 3.0
        
        return curvature_factor
    
    def check_physics_constraints(self, geometry: GeometryParameters) -> Dict[str, float]:
        """Check physics constraints for geometry"""
        
        violations = {}
        
        # Volume constraint (must contain vehicle)
        min_volume = 4.0 * 1.5 * 1.2  # Minimum required volume
        actual_volume = geometry.length * geometry.width * geometry.height
        if actual_volume < min_volume:
            violations['volume_constraint'] = (min_volume - actual_volume) / min_volume
        
        # Aspect ratio constraints (for stability)
        if geometry.aspect_ratio_lw < 1.5 or geometry.aspect_ratio_lw > 4.0:
            violations['lw_aspect_ratio'] = abs(geometry.aspect_ratio_lw - 2.5) / 2.5
        
        if geometry.aspect_ratio_lh < 2.0 or geometry.aspect_ratio_lh > 5.0:
            violations['lh_aspect_ratio'] = abs(geometry.aspect_ratio_lh - 3.0) / 3.0
        
        # Curvature constraints (for field stability)
        if geometry.front_curvature > geometry.rear_curvature * 2.5:
            violations['curvature_balance'] = (geometry.front_curvature / geometry.rear_curvature - 2.5) / 2.5
        
        # Smoothness constraint (must be high enough for stable field)
        if geometry.transition_smoothness < 0.7:
            violations['smoothness_constraint'] = (0.7 - geometry.transition_smoothness) / 0.7
        
        # T_μν ≥ 0 constraint check (simplified)
        stress_energy_violation = self._check_stress_energy_constraint(geometry)
        if stress_energy_violation > 0:
            violations['stress_energy_tensor'] = stress_energy_violation
        
        return violations
    
    def _check_stress_energy_constraint(self, geometry: GeometryParameters) -> float:
        """Check stress-energy tensor constraint T_μν ≥ 0"""
        
        # Simplified stress-energy calculation
        # In practice, this would require full metric tensor calculation
        
        # High curvatures and sharp transitions can violate T_μν ≥ 0
        max_curvature = max(geometry.front_curvature, geometry.rear_curvature, geometry.side_curvature)
        
        # Critical thresholds
        curvature_threshold = 0.8
        smoothness_threshold = 0.6
        
        violation = 0.0
        
        # Curvature violation
        if max_curvature > curvature_threshold:
            violation += (max_curvature - curvature_threshold) / curvature_threshold
        
        # Smoothness violation
        if geometry.transition_smoothness < smoothness_threshold:
            violation += (smoothness_threshold - geometry.transition_smoothness) / smoothness_threshold
        
        # Extreme aspect ratio violation
        if geometry.aspect_ratio_lw > 3.5 or geometry.aspect_ratio_lh > 4.5:
            violation += 0.1 * max(geometry.aspect_ratio_lw - 3.5, geometry.aspect_ratio_lh - 4.5, 0)
        
        return violation
    
    def objective_function(self, params: np.ndarray) -> float:
        """Objective function for optimization"""
        
        # Convert parameter array to geometry
        geometry = self._params_to_geometry(params)
        
        # Calculate energy
        energy = self.calculate_curvature_energy(geometry)
        
        # Check constraints
        violations = self.check_physics_constraints(geometry)
        
        # Penalty for constraint violations
        penalty = 0.0
        for violation in violations.values():
            penalty += violation ** 2 * 1e10  # Large penalty for violations
        
        return energy + penalty
    
    def _params_to_geometry(self, params: np.ndarray) -> GeometryParameters:
        """Convert parameter array to geometry object"""
        
        geometry = GeometryParameters()
        
        # Map parameters
        geometry.length = params[0]
        geometry.width = params[1]
        geometry.height = params[2]
        geometry.ellipticity = params[3]
        geometry.front_curvature = params[4]
        geometry.rear_curvature = params[5]
        geometry.side_curvature = params[6]
        geometry.transition_smoothness = params[7]
        geometry.taper_factor = params[8]
        geometry.bulge_factor = params[9]
        geometry.asymmetry_factor = params[10]
        
        # Calculate derived parameters
        geometry.aspect_ratio_lw = geometry.length / geometry.width
        geometry.aspect_ratio_lh = geometry.length / geometry.height
        
        return geometry
    
    def _geometry_to_params(self, geometry: GeometryParameters) -> np.ndarray:
        """Convert geometry object to parameter array"""
        
        return np.array([
            geometry.length,
            geometry.width,
            geometry.height,
            geometry.ellipticity,
            geometry.front_curvature,
            geometry.rear_curvature,
            geometry.side_curvature,
            geometry.transition_smoothness,
            geometry.taper_factor,
            geometry.bulge_factor,
            geometry.asymmetry_factor
        ])
    
    def optimize_geometry_differential_evolution(self) -> OptimizationResult:
        """Optimize geometry using differential evolution"""
        
        logger.info("Starting differential evolution optimization...")
        
        # Parameter bounds
        bounds = [
            self.geometry.param_bounds['length'],
            self.geometry.param_bounds['width'],
            self.geometry.param_bounds['height'],
            self.geometry.param_bounds['ellipticity'],
            self.geometry.param_bounds['front_curvature'],
            self.geometry.param_bounds['rear_curvature'],
            self.geometry.param_bounds['side_curvature'],
            self.geometry.param_bounds['transition_smoothness'],
            self.geometry.param_bounds['taper_factor'],
            self.geometry.param_bounds['bulge_factor'],
            self.geometry.param_bounds['asymmetry_factor']
        ]
        
        # Initial parameters
        initial_params = self._geometry_to_params(self.geometry)
        original_energy = self.calculate_curvature_energy(self.geometry)
        
        import time
        start_time = time.time()
        
        # Run optimization
        result = differential_evolution(
            self.objective_function,
            bounds,
            maxiter=self.max_iterations,
            tol=self.optimization_tolerance,
            seed=42,
            workers=1,  # Single worker for reproducibility
            polish=True
        )
        
        optimization_time = time.time() - start_time
        
        # Extract results
        optimized_geometry = self._params_to_geometry(result.x)
        optimized_energy = self.calculate_curvature_energy(optimized_geometry)
        
        # Check final constraints
        violations = self.check_physics_constraints(optimized_geometry)
        
        optimization_result = OptimizationResult(
            optimized_geometry=optimized_geometry,
            original_energy=original_energy,
            optimized_energy=optimized_energy,
            energy_reduction_factor=original_energy / optimized_energy,
            curvature_energy=optimized_energy,
            field_strength_max=max(optimized_geometry.front_curvature, 
                                 optimized_geometry.rear_curvature,
                                 optimized_geometry.side_curvature),
            constraint_violations=violations,
            optimization_success=result.success and len(violations) == 0,
            optimization_time=optimization_time,
            iterations_required=result.nit,
            convergence_achieved=result.success
        )
        
        logger.info(f"Differential evolution optimization complete:")
        logger.info(f"  Energy reduction: {optimization_result.energy_reduction_factor:.2f}×")
        logger.info(f"  Optimization time: {optimization_time:.1f} seconds")
        logger.info(f"  Constraint violations: {len(violations)}")
        
        return optimization_result
    
    def optimize_geometry_basinhopping(self) -> OptimizationResult:
        """Optimize geometry using basin hopping for global optimization"""
        
        logger.info("Starting basin hopping optimization...")
        
        # Parameter bounds for basin hopping
        bounds = [
            self.geometry.param_bounds['length'],
            self.geometry.param_bounds['width'],
            self.geometry.param_bounds['height'],
            self.geometry.param_bounds['ellipticity'],
            self.geometry.param_bounds['front_curvature'],
            self.geometry.param_bounds['rear_curvature'],
            self.geometry.param_bounds['side_curvature'],
            self.geometry.param_bounds['transition_smoothness'],
            self.geometry.param_bounds['taper_factor'],
            self.geometry.param_bounds['bulge_factor'],
            self.geometry.param_bounds['asymmetry_factor']
        ]
        
        # Initial parameters
        initial_params = self._geometry_to_params(self.geometry)
        original_energy = self.calculate_curvature_energy(self.geometry)
        
        import time
        start_time = time.time()
        
        # Define bounds for scipy
        from scipy.optimize import Bounds
        scipy_bounds = Bounds([b[0] for b in bounds], [b[1] for b in bounds])
        
        # Run basin hopping optimization
        result = basinhopping(
            self.objective_function,
            initial_params,
            niter=100,
            T=1.0,
            stepsize=0.1,
            minimizer_kwargs={
                'method': 'L-BFGS-B',
                'bounds': scipy_bounds,
                'options': {'maxiter': 200}
            },
            seed=42
        )
        
        optimization_time = time.time() - start_time
        
        # Extract results
        optimized_geometry = self._params_to_geometry(result.x)
        optimized_energy = self.calculate_curvature_energy(optimized_geometry)
        
        # Check final constraints
        violations = self.check_physics_constraints(optimized_geometry)
        
        optimization_result = OptimizationResult(
            optimized_geometry=optimized_geometry,
            original_energy=original_energy,
            optimized_energy=optimized_energy,
            energy_reduction_factor=original_energy / optimized_energy,
            curvature_energy=optimized_energy,
            field_strength_max=max(optimized_geometry.front_curvature,
                                 optimized_geometry.rear_curvature,
                                 optimized_geometry.side_curvature),
            constraint_violations=violations,
            optimization_success=result.success and len(violations) == 0,
            optimization_time=optimization_time,
            iterations_required=result.nit,
            convergence_achieved=result.success
        )
        
        logger.info(f"Basin hopping optimization complete:")
        logger.info(f"  Energy reduction: {optimization_result.energy_reduction_factor:.2f}×")
        logger.info(f"  Optimization time: {optimization_time:.1f} seconds")
        logger.info(f"  Constraint violations: {len(violations)}")
        
        return optimization_result
    
    def optimize_geometry_multi_objective(self) -> OptimizationResult:
        """Multi-objective optimization balancing energy and other factors"""
        
        logger.info("Starting multi-objective optimization...")
        
        def multi_objective_function(params: np.ndarray) -> float:
            geometry = self._params_to_geometry(params)
            
            # Primary objective: energy minimization
            energy = self.calculate_curvature_energy(geometry)
            energy_objective = energy / self.base_energy  # Normalized
            
            # Secondary objective: field stability (prefer smoother transitions)
            stability_objective = 1.0 - geometry.transition_smoothness
            
            # Tertiary objective: geometric efficiency
            volume = geometry.length * geometry.width * geometry.height
            surface_area = 2 * (geometry.length * geometry.width + 
                               geometry.width * geometry.height + 
                               geometry.height * geometry.length)
            efficiency_objective = surface_area / (volume ** (2/3))  # Lower is better
            
            # Constraint violations
            violations = self.check_physics_constraints(geometry)
            violation_penalty = sum(v ** 2 for v in violations.values()) * 1000
            
            # Weighted combination
            total_objective = (0.7 * energy_objective + 
                             0.2 * stability_objective + 
                             0.1 * efficiency_objective + 
                             violation_penalty)
            
            return total_objective
        
        # Parameter bounds
        bounds = [
            self.geometry.param_bounds['length'],
            self.geometry.param_bounds['width'],
            self.geometry.param_bounds['height'],
            self.geometry.param_bounds['ellipticity'],
            self.geometry.param_bounds['front_curvature'],
            self.geometry.param_bounds['rear_curvature'],
            self.geometry.param_bounds['side_curvature'],
            self.geometry.param_bounds['transition_smoothness'],
            self.geometry.param_bounds['taper_factor'],
            self.geometry.param_bounds['bulge_factor'],
            self.geometry.param_bounds['asymmetry_factor']
        ]
        
        initial_params = self._geometry_to_params(self.geometry)
        original_energy = self.calculate_curvature_energy(self.geometry)
        
        import time
        start_time = time.time()
        
        # Run multi-objective optimization
        result = differential_evolution(
            multi_objective_function,
            bounds,
            maxiter=self.max_iterations // 2,  # Fewer iterations for multi-objective
            tol=self.optimization_tolerance,
            seed=42,
            workers=1,
            polish=True
        )
        
        optimization_time = time.time() - start_time
        
        # Extract results
        optimized_geometry = self._params_to_geometry(result.x)
        optimized_energy = self.calculate_curvature_energy(optimized_geometry)
        violations = self.check_physics_constraints(optimized_geometry)
        
        optimization_result = OptimizationResult(
            optimized_geometry=optimized_geometry,
            original_energy=original_energy,
            optimized_energy=optimized_energy,
            energy_reduction_factor=original_energy / optimized_energy,
            curvature_energy=optimized_energy,
            field_strength_max=max(optimized_geometry.front_curvature,
                                 optimized_geometry.rear_curvature,
                                 optimized_geometry.side_curvature),
            constraint_violations=violations,
            optimization_success=result.success and len(violations) == 0,
            optimization_time=optimization_time,
            iterations_required=result.nit,
            convergence_achieved=result.success
        )
        
        logger.info(f"Multi-objective optimization complete:")
        logger.info(f"  Energy reduction: {optimization_result.energy_reduction_factor:.2f}×")
        logger.info(f"  Optimization time: {optimization_time:.1f} seconds")
        logger.info(f"  Constraint violations: {len(violations)}")
        
        return optimization_result
    
    def run_comprehensive_optimization(self) -> Dict[str, OptimizationResult]:
        """Run comprehensive optimization using multiple methods"""
        
        logger.info("Running comprehensive geometry optimization...")
        
        results = {}
        
        # Method 1: Differential Evolution
        try:
            results['differential_evolution'] = self.optimize_geometry_differential_evolution()
        except Exception as e:
            logger.error(f"Differential evolution failed: {e}")
            results['differential_evolution'] = None
        
        # Method 2: Basin Hopping
        try:
            results['basin_hopping'] = self.optimize_geometry_basinhopping()
        except Exception as e:
            logger.error(f"Basin hopping failed: {e}")
            results['basin_hopping'] = None
        
        # Method 3: Multi-objective
        try:
            results['multi_objective'] = self.optimize_geometry_multi_objective()
        except Exception as e:
            logger.error(f"Multi-objective optimization failed: {e}")
            results['multi_objective'] = None
        
        # Select best result
        best_result = None
        best_reduction = 0
        best_method = None
        
        for method, result in results.items():
            if result and result.optimization_success:
                if result.energy_reduction_factor > best_reduction:
                    best_reduction = result.energy_reduction_factor
                    best_result = result
                    best_method = method
        
        if best_result:
            logger.info(f"Best optimization method: {best_method}")
            logger.info(f"Best energy reduction: {best_reduction:.2f}×")
            logger.info(f"Target achieved: {'YES' if best_reduction >= self.target_reduction else 'NO'}")
        else:
            logger.warning("No successful optimization found")
        
        self.optimization_results = results
        return results
    
    def generate_geometry_mesh(self, geometry: GeometryParameters, resolution: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate 3D mesh for geometry visualization"""
        
        # Create parametric surface for bubble
        u = np.linspace(0, 2*np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        U, V = np.meshgrid(u, v)
        
        # Basic ellipsoidal shape with modifications
        a = geometry.length / 2
        b = geometry.width / 2  
        c = geometry.height / 2
        
        # Apply ellipticity
        ellip = geometry.ellipticity
        
        # Base ellipsoid
        x_base = a * ellip * np.sin(V) * np.cos(U)
        y_base = b * np.sin(V) * np.sin(U)
        z_base = c * np.cos(V)
        
        # Apply tapering (front to rear)
        taper_scale = 1.0 - geometry.taper_factor * (x_base / a + 1) / 2
        x = x_base * taper_scale
        y = y_base * taper_scale
        z = z_base * taper_scale
        
        # Apply bulging
        bulge_scale = 1.0 + geometry.bulge_factor * np.exp(-((x_base/a)**2 + (y_base/b)**2 + (z_base/c)**2))
        x *= bulge_scale
        y *= bulge_scale
        z *= bulge_scale
        
        return x, y, z
    
    def visualize_optimization_results(self, save_path: Optional[str] = None):
        """Create comprehensive visualization of optimization results"""
        
        if not self.optimization_results:
            self.run_comprehensive_optimization()
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Original vs optimized geometry
        ax1 = fig.add_subplot(gs[0, :2], projection='3d')
        self._plot_geometry_comparison(ax1)
        
        # 2. Energy reduction comparison
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_energy_comparison(ax2)
        
        # 3. Parameter optimization
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_parameter_optimization(ax3)
        
        # 4. Constraint satisfaction
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_constraint_satisfaction(ax4)
        
        # 5. Convergence analysis
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_convergence_analysis(ax5)
        
        # 6. Physics validation
        ax6 = fig.add_subplot(gs[2, 2:])
        self._plot_physics_validation(ax6)
        
        plt.suptitle('Warp Bubble Geometry Optimization Results', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Geometry optimization visualization saved to: {save_path}")
        
        plt.show()
    
    def _plot_geometry_comparison(self, ax):
        """Plot original vs optimized geometry"""
        
        # Find best result
        best_result = None
        for result in self.optimization_results.values():
            if result and result.optimization_success:
                if best_result is None or result.energy_reduction_factor > best_result.energy_reduction_factor:
                    best_result = result
        
        if not best_result:
            ax.text(0.5, 0.5, 0.5, 'No successful optimization', ha='center', va='center')
            return
        
        # Generate meshes
        x_orig, y_orig, z_orig = self.generate_geometry_mesh(self.geometry, 30)
        x_opt, y_opt, z_opt = self.generate_geometry_mesh(best_result.optimized_geometry, 30)
        
        # Plot original (wireframe)
        ax.plot_wireframe(x_orig, y_orig, z_orig, alpha=0.3, color='red', label='Original')
        
        # Plot optimized (surface)
        ax.plot_surface(x_opt, y_opt, z_opt, alpha=0.7, cmap='viridis', label='Optimized')
        
        ax.set_xlabel('Length (m)')
        ax.set_ylabel('Width (m)')
        ax.set_zlabel('Height (m)')
        ax.set_title('Original vs Optimized Geometry')
    
    def _plot_energy_comparison(self, ax):
        """Plot energy comparison across methods"""
        
        methods = []
        original_energies = []
        optimized_energies = []
        reduction_factors = []
        
        for method, result in self.optimization_results.items():
            if result:
                methods.append(method.replace('_', '\n'))
                original_energies.append(result.original_energy / 1e9)  # Billions
                optimized_energies.append(result.optimized_energy / 1e9)
                reduction_factors.append(result.energy_reduction_factor)
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, original_energies, width, label='Original Energy', alpha=0.7, color='red')
        bars2 = ax.bar(x + width/2, optimized_energies, width, label='Optimized Energy', alpha=0.7, color='green')
        
        ax.set_xlabel('Optimization Methods')
        ax.set_ylabel('Energy (Billion J)')
        ax.set_title('Energy Reduction by Optimization Method')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add reduction factor labels
        for i, (bar, factor) in enumerate(zip(bars2, reduction_factors)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                   f'{factor:.1f}×', ha='center', va='bottom')
        
        # Add target line
        ax.axhline(y=self.target_energy/1e9, color='blue', linestyle='--', 
                  linewidth=2, label=f'Target ({self.target_reduction}× reduction)')
    
    def _plot_parameter_optimization(self, ax):
        """Plot parameter optimization results"""
        
        # Find best result
        best_result = None
        for result in self.optimization_results.values():
            if result and result.optimization_success:
                if best_result is None or result.energy_reduction_factor > best_result.energy_reduction_factor:
                    best_result = result
        
        if not best_result:
            ax.text(0.5, 0.5, 'No successful optimization', ha='center', va='center')
            return
        
        # Parameter names and values
        param_names = ['Length', 'Width', 'Height', 'Ellipticity', 'Front Curv.', 
                      'Rear Curv.', 'Side Curv.', 'Smoothness', 'Taper', 'Bulge', 'Asymmetry']
        
        original_values = [
            self.geometry.length, self.geometry.width, self.geometry.height,
            self.geometry.ellipticity, self.geometry.front_curvature,
            self.geometry.rear_curvature, self.geometry.side_curvature,
            self.geometry.transition_smoothness, self.geometry.taper_factor,
            self.geometry.bulge_factor, self.geometry.asymmetry_factor
        ]
        
        optimized_values = [
            best_result.optimized_geometry.length, best_result.optimized_geometry.width,
            best_result.optimized_geometry.height, best_result.optimized_geometry.ellipticity,
            best_result.optimized_geometry.front_curvature, best_result.optimized_geometry.rear_curvature,
            best_result.optimized_geometry.side_curvature, best_result.optimized_geometry.transition_smoothness,
            best_result.optimized_geometry.taper_factor, best_result.optimized_geometry.bulge_factor,
            best_result.optimized_geometry.asymmetry_factor
        ]
        
        # Normalize values for comparison
        normalized_orig = []
        normalized_opt = []
        
        for i, (orig, opt) in enumerate(zip(original_values, optimized_values)):
            if param_names[i] in ['Length', 'Width', 'Height']:
                # Normalize by original value
                normalized_orig.append(1.0)
                normalized_opt.append(opt / orig)
            else:
                # Already normalized parameters
                normalized_orig.append(orig)
                normalized_opt.append(opt)
        
        x = np.arange(len(param_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, normalized_orig, width, label='Original', alpha=0.7)
        bars2 = ax.bar(x + width/2, normalized_opt, width, label='Optimized', alpha=0.7)
        
        ax.set_xlabel('Parameters')
        ax.set_ylabel('Normalized Values')
        ax.set_title('Parameter Optimization Results')
        ax.set_xticks(x)
        ax.set_xticklabels([name.replace(' ', '\n') for name in param_names], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_constraint_satisfaction(self, ax):
        """Plot constraint satisfaction across methods"""
        
        methods = []
        violation_counts = []
        max_violations = []
        
        for method, result in self.optimization_results.items():
            if result:
                methods.append(method.replace('_', '\n'))
                violation_counts.append(len(result.constraint_violations))
                max_violations.append(max(result.constraint_violations.values()) if result.constraint_violations else 0)
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, violation_counts, width, label='Violation Count', alpha=0.7, color='red')
        bars2 = ax.bar(x + width/2, max_violations, width, label='Max Violation', alpha=0.7, color='orange')
        
        ax.set_xlabel('Optimization Methods')
        ax.set_ylabel('Constraint Violations')
        ax.set_title('Constraint Satisfaction Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_convergence_analysis(self, ax):
        """Plot convergence analysis"""
        
        methods = []
        iterations = []
        times = []
        success_rates = []
        
        for method, result in self.optimization_results.items():
            if result:
                methods.append(method.replace('_', '\n'))
                iterations.append(result.iterations_required)
                times.append(result.optimization_time)
                success_rates.append(1.0 if result.optimization_success else 0.0)
        
        # Create scatter plot of iterations vs time, colored by success
        scatter = ax.scatter(iterations, times, c=success_rates, s=100, 
                           cmap='RdYlGn', alpha=0.7, edgecolors='black')
        
        ax.set_xlabel('Iterations Required')
        ax.set_ylabel('Optimization Time (seconds)')
        ax.set_title('Convergence Analysis')
        ax.grid(True, alpha=0.3)
        
        # Add method labels
        for i, method in enumerate(methods):
            ax.annotate(method, (iterations[i], times[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Success Rate', rotation=270, labelpad=15)
    
    def _plot_physics_validation(self, ax):
        """Plot physics validation results"""
        
        # Collect physics validation data
        methods = []
        energy_reductions = []
        constraint_scores = []
        
        for method, result in self.optimization_results.items():
            if result:
                methods.append(method.replace('_', '\n'))
                energy_reductions.append(result.energy_reduction_factor)
                
                # Calculate constraint satisfaction score
                violation_score = 1.0 - min(1.0, sum(result.constraint_violations.values()))
                constraint_scores.append(max(0.0, violation_score))
        
        # Plot energy reduction vs constraint satisfaction
        scatter = ax.scatter(constraint_scores, energy_reductions, s=150, alpha=0.7, 
                           c=range(len(methods)), cmap='viridis', edgecolors='black')
        
        ax.set_xlabel('Constraint Satisfaction Score')
        ax.set_ylabel('Energy Reduction Factor')
        ax.set_title('Physics Validation: Energy vs Constraints')
        ax.grid(True, alpha=0.3)
        
        # Add target lines
        ax.axhline(y=self.target_reduction, color='red', linestyle='--', 
                  linewidth=2, label=f'Target ({self.target_reduction}× reduction)')
        ax.axvline(x=0.95, color='blue', linestyle='--', 
                  linewidth=2, label='High Constraint Satisfaction')
        
        # Add method labels
        for i, method in enumerate(methods):
            ax.annotate(method, (constraint_scores[i], energy_reductions[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.legend()

def main():
    """Main execution function for geometry optimization"""
    
    print("=" * 80)
    print("WARP BUBBLE GEOMETRY OPTIMIZATION ENGINE")
    print("Phase 2 Implementation: Advanced Geometry Optimization")
    print("=" * 80)
    
    # Initialize optimization engine
    engine = GeometryOptimizationEngine()
    
    print(f"\n🎯 OPTIMIZATION TARGET:")
    print(f"Current Energy: {engine.base_energy/1e9:.2f} billion J")
    print(f"Target Energy: {engine.target_energy/1e6:.0f} million J")
    print(f"Required Reduction: {engine.target_reduction}×")
    
    # Run comprehensive optimization
    print(f"\n🔧 RUNNING COMPREHENSIVE GEOMETRY OPTIMIZATION...")
    results = engine.run_comprehensive_optimization()
    
    # Analyze results
    print(f"\n📊 OPTIMIZATION RESULTS:")
    
    successful_methods = 0
    best_reduction = 0
    best_method = None
    best_result = None
    
    for method, result in results.items():
        print(f"\n{method.upper().replace('_', ' ')}:")
        if result:
            print(f"   Energy Reduction: {result.energy_reduction_factor:.2f}×")
            print(f"   Original Energy: {result.original_energy/1e9:.2f} billion J")
            print(f"   Optimized Energy: {result.optimized_energy/1e6:.0f} million J")
            print(f"   Optimization Time: {result.optimization_time:.1f} seconds")
            print(f"   Constraint Violations: {len(result.constraint_violations)}")
            print(f"   Success: {'✅ YES' if result.optimization_success else '❌ NO'}")
            
            if result.optimization_success:
                successful_methods += 1
                if result.energy_reduction_factor > best_reduction:
                    best_reduction = result.energy_reduction_factor
                    best_method = method
                    best_result = result
        else:
            print(f"   Status: ❌ FAILED")
    
    # Summary
    print(f"\n🏆 OPTIMIZATION SUMMARY:")
    print(f"Successful Methods: {successful_methods}/{len(results)}")
    
    if best_result:
        print(f"Best Method: {best_method}")
        print(f"Best Energy Reduction: {best_reduction:.2f}×")
        print(f"Target Achievement: {'✅ YES' if best_reduction >= engine.target_reduction else '❌ NO'}")
        
        if best_reduction >= engine.target_reduction:
            print(f"\n🎉 TARGET ACHIEVED! Geometry optimization successful!")
            print(f"Energy reduced from {engine.base_energy/1e9:.2f}B J to {best_result.optimized_energy/1e6:.0f}M J")
        else:
            shortfall = engine.target_reduction / best_reduction
            print(f"\n⚠️ Target not fully achieved. Additional {shortfall:.1f}× reduction needed.")
        
        # Optimized geometry details
        opt_geom = best_result.optimized_geometry
        print(f"\n📐 OPTIMIZED GEOMETRY PARAMETERS:")
        print(f"   Length: {opt_geom.length:.2f} m (was {engine.geometry.length:.2f} m)")
        print(f"   Width: {opt_geom.width:.2f} m (was {engine.geometry.width:.2f} m)")
        print(f"   Height: {opt_geom.height:.2f} m (was {engine.geometry.height:.2f} m)")
        print(f"   Ellipticity: {opt_geom.ellipticity:.3f} (was {engine.geometry.ellipticity:.3f})")
        print(f"   Front Curvature: {opt_geom.front_curvature:.3f} (was {engine.geometry.front_curvature:.3f})")
        print(f"   Transition Smoothness: {opt_geom.transition_smoothness:.3f} (was {engine.geometry.transition_smoothness:.3f})")
    else:
        print(f"❌ No successful optimization achieved")
    
    # Generate visualization
    print(f"\n📊 GENERATING OPTIMIZATION VISUALIZATION...")
    viz_path = "energy_optimization/geometry_optimization_results.png"
    engine.visualize_optimization_results(viz_path)
    
    # Save optimization results
    results_path = "energy_optimization/geometry_optimization_report.json"
    
    # Prepare results for JSON serialization
    json_results = {}
    for method, result in results.items():
        if result:
            json_results[method] = {
                'energy_reduction_factor': result.energy_reduction_factor,
                'original_energy': result.original_energy,
                'optimized_energy': result.optimized_energy,
                'optimization_time': result.optimization_time,
                'iterations_required': result.iterations_required,
                'constraint_violations': result.constraint_violations,
                'optimization_success': result.optimization_success,
                'optimized_geometry': {
                    'length': result.optimized_geometry.length,
                    'width': result.optimized_geometry.width,
                    'height': result.optimized_geometry.height,
                    'ellipticity': result.optimized_geometry.ellipticity,
                    'front_curvature': result.optimized_geometry.front_curvature,
                    'rear_curvature': result.optimized_geometry.rear_curvature,
                    'side_curvature': result.optimized_geometry.side_curvature,
                    'transition_smoothness': result.optimized_geometry.transition_smoothness,
                    'taper_factor': result.optimized_geometry.taper_factor,
                    'bulge_factor': result.optimized_geometry.bulge_factor,
                    'asymmetry_factor': result.optimized_geometry.asymmetry_factor
                }
            }
        else:
            json_results[method] = None
    
    report = {
        'optimization_summary': {
            'target_reduction': engine.target_reduction,
            'best_reduction_achieved': best_reduction,
            'target_achieved': best_reduction >= engine.target_reduction if best_result else False,
            'best_method': best_method,
            'successful_methods': successful_methods,
            'total_methods': len(results)
        },
        'optimization_results': json_results,
        'original_geometry': {
            'length': engine.geometry.length,
            'width': engine.geometry.width,
            'height': engine.geometry.height,
            'ellipticity': engine.geometry.ellipticity,
            'front_curvature': engine.geometry.front_curvature,
            'rear_curvature': engine.geometry.rear_curvature,
            'side_curvature': engine.geometry.side_curvature,
            'transition_smoothness': engine.geometry.transition_smoothness,
            'taper_factor': engine.geometry.taper_factor,
            'bulge_factor': engine.geometry.bulge_factor,
            'asymmetry_factor': engine.geometry.asymmetry_factor
        }
    }
    
    with open(results_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Optimization report saved to: {results_path}")
    
    print("\n" + "=" * 80)
    print("GEOMETRY OPTIMIZATION ENGINE COMPLETE")
    if best_result and best_reduction >= engine.target_reduction:
        print("STATUS: ✅ TARGET ACHIEVED - 10× geometry energy reduction successful!")
    elif best_result:
        print(f"STATUS: ⚠️ PARTIAL SUCCESS - {best_reduction:.1f}× reduction achieved")
    else:
        print("STATUS: ❌ OPTIMIZATION FAILED")
    print("=" * 80)

if __name__ == "__main__":
    main()

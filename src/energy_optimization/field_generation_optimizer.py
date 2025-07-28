#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Field Generation Optimizer for Warp Bubble Energy Efficiency

This module implements advanced field generation optimization techniques to minimize
electromagnetic field generation losses while maintaining spacetime curvature requirements.
This addresses the second highest-priority optimization target from Phase 1 analysis.

Repository: lqg-ftl-metric-engineering
Function: Electromagnetic field optimization for energy efficiency
Technology: Advanced field generation with superconducting optimization
Status: PHASE 2 IMPLEMENTATION - Targeting 6× reduction in field generation losses

Research Objective:
- Optimize electromagnetic field generation for minimum energy loss
- Reduce field generation losses from 2.025 billion J to ~340 million J (6× reduction)
- Implement superconducting and resonant field techniques
- Maintain field strength requirements for spacetime curvature
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad, dblquad, solve_ivp
from scipy.special import jv, yv, hankel1, hankel2  # Bessel functions
from scipy.fft import fft, ifft, fftfreq
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
import json
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FieldConfiguration:
    """Field generation configuration parameters"""
    # Coil configuration
    num_coils: int = 12                    # Number of field coils
    coil_radius: float = 3.0               # Coil radius (m)
    coil_separation: float = 1.5           # Separation between coils (m)
    coil_current: float = 1e6              # Coil current (A)
    
    # Superconducting parameters
    critical_temperature: float = 77.0     # Critical temperature (K)
    critical_current_density: float = 1e9  # Critical current density (A/m²)
    superconductor_type: str = "YBCO"      # Superconductor material
    operating_temperature: float = 20.0    # Operating temperature (K)
    
    # Resonance parameters
    resonant_frequency: float = 1e9        # Resonant frequency (Hz)
    quality_factor: float = 1e6            # Quality factor
    coupling_efficiency: float = 0.95      # Coupling efficiency
    
    # Field strength parameters
    target_field_strength: float = 10.0    # Target field strength (T)
    field_uniformity: float = 0.98         # Required field uniformity
    field_stability: float = 1e-6          # Field stability requirement
    
    # Energy parameters
    resistive_losses: float = 2.025e9      # Current resistive losses (J)
    inductive_losses: float = 3.375e8     # Inductive losses (J)
    capacitive_losses: float = 2.025e8    # Capacitive losses (J)
    
    # Optimization bounds
    optimization_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'num_coils': (8, 20),
        'coil_radius': (2.0, 5.0),
        'coil_separation': (0.8, 2.5),
        'coil_current': (5e5, 2e6),
        'resonant_frequency': (1e8, 1e11),
        'quality_factor': (1e4, 1e7),
        'coupling_efficiency': (0.85, 0.99),
        'operating_temperature': (4.2, 77.0)
    })

@dataclass
class OptimizationResult:
    """Field optimization results"""
    optimized_config: FieldConfiguration
    original_energy: float
    optimized_energy: float
    energy_reduction_factor: float
    
    # Performance metrics
    field_efficiency: float
    power_consumption: float
    heat_generation: float
    
    # Constraint satisfaction
    field_strength_achieved: float
    uniformity_achieved: float
    stability_achieved: float
    
    # Implementation details
    optimization_time: float
    convergence_achieved: bool
    constraint_violations: Dict[str, float]

class FieldGenerationOptimizer:
    """Advanced field generation optimization for warp bubble systems"""
    
    def __init__(self):
        self.config = FieldConfiguration()
        self.optimization_results = {}
        
        # Physical constants
        self.mu0 = 4 * np.pi * 1e-7         # Permeability of free space
        self.epsilon0 = 8.854187817e-12     # Permittivity of free space
        self.c = 299792458                  # Speed of light
        self.kb = 1.380649e-23              # Boltzmann constant
        self.h = 6.62607015e-34             # Planck constant
        
        # Superconductor properties
        self.sc_properties = {
            'YBCO': {
                'tc': 92.0,          # Critical temperature (K)
                'jc0': 1e10,         # Critical current density at 0K (A/m²)
                'bc2': 100.0,        # Upper critical field (T)
                'resistivity': 1e-7  # Normal state resistivity (Ω⋅m)
            },
            'BSCCO': {
                'tc': 110.0,
                'jc0': 5e9,
                'bc2': 120.0,
                'resistivity': 2e-7
            },
            'NbTi': {
                'tc': 9.5,
                'jc0': 3e9,
                'bc2': 15.0,
                'resistivity': 5e-8
            }
        }
        
        # Optimization targets
        self.base_energy = 2.025e9          # Current resistive losses
        self.target_reduction = 6.0         # Target reduction factor
        self.target_energy = self.base_energy / self.target_reduction
        
        logger.info("Field Generation Optimizer initialized")
        logger.info(f"Target: {self.base_energy/1e9:.3f} billion J → {self.target_energy/1e6:.0f} million J")
        logger.info(f"Required reduction: {self.target_reduction}×")
    
    def calculate_superconducting_losses(self, config: FieldConfiguration) -> float:
        """Calculate superconducting losses for field configuration"""
        
        # Get superconductor properties
        sc_props = self.sc_properties[config.superconductor_type]
        
        # Temperature dependence of critical current
        t_ratio = config.operating_temperature / sc_props['tc']
        if t_ratio >= 1.0:
            # Normal conducting regime
            return self._calculate_resistive_losses(config)
        
        # Critical current density with temperature dependence
        jc = sc_props['jc0'] * (1 - t_ratio) ** 1.5
        
        # Current density in coils
        coil_area = np.pi * (config.coil_radius * 0.1) ** 2  # Assume 10% of radius for conductor
        j_operating = config.coil_current / (config.num_coils * coil_area)
        
        # AC losses (dominant loss mechanism in superconductors)
        frequency = config.resonant_frequency
        penetration_depth = np.sqrt(2 / (self.mu0 * 2 * np.pi * frequency * 1e6))  # Assume 1 MS/m conductivity
        
        # Hysteresis losses
        field_amplitude = config.target_field_strength
        hysteresis_loss_density = 0.1 * field_amplitude ** 2 * frequency  # Simplified model
        
        # Surface losses
        surface_area = 2 * np.pi * config.coil_radius * config.num_coils * 0.5  # Total conductor surface
        surface_loss_density = 1e-3 * field_amplitude ** 2 * frequency ** 0.5
        
        # Total superconducting losses
        total_losses = (hysteresis_loss_density + surface_loss_density) * surface_area * penetration_depth
        
        # Add cooling power requirement (factor of ~500 at 20K)
        cooling_factor = 500 if config.operating_temperature < 77 else 1
        total_losses *= cooling_factor
        
        return total_losses
    
    def _calculate_resistive_losses(self, config: FieldConfiguration) -> float:
        """Calculate resistive losses for normal conducting coils"""
        
        # Coil resistance calculation
        coil_length = 2 * np.pi * config.coil_radius * config.num_coils
        conductor_area = config.coil_current / 1e6  # Assume 1 MA/m² current density
        resistivity = 2.8e-8  # Copper resistivity at room temperature
        
        total_resistance = resistivity * coil_length / conductor_area
        
        # I²R losses
        resistive_losses = config.coil_current ** 2 * total_resistance
        
        return resistive_losses
    
    def calculate_resonant_enhancement(self, config: FieldConfiguration) -> float:
        """Calculate resonant field enhancement factor"""
        
        # Quality factor determines enhancement
        Q = config.quality_factor
        
        # Resonant enhancement factor
        enhancement = np.sqrt(Q * config.coupling_efficiency)
        
        # Power reduction due to resonance
        power_reduction = 1.0 / enhancement
        
        return power_reduction
    
    def calculate_field_efficiency(self, config: FieldConfiguration) -> float:
        """Calculate overall field generation efficiency"""
        
        # Superconducting losses
        sc_losses = self.calculate_superconducting_losses(config)
        
        # Resonant enhancement
        resonant_factor = self.calculate_resonant_enhancement(config)
        
        # Coil geometry efficiency
        geometry_efficiency = self._calculate_geometry_efficiency(config)
        
        # Total energy with optimizations
        total_energy = sc_losses * resonant_factor / geometry_efficiency
        
        return total_energy
    
    def _calculate_geometry_efficiency(self, config: FieldConfiguration) -> float:
        """Calculate field generation efficiency based on coil geometry"""
        
        # Optimal coil spacing for maximum efficiency
        optimal_radius = 2.5  # Optimal radius for this bubble size
        optimal_separation = 1.2  # Optimal separation
        optimal_coils = 16  # Optimal number of coils
        
        # Efficiency factors
        radius_factor = optimal_radius / max(config.coil_radius, optimal_radius/2)
        separation_factor = optimal_separation / max(config.coil_separation, optimal_separation/2)
        coil_factor = min(config.num_coils, optimal_coils) / optimal_coils
        
        # Combined geometry efficiency
        efficiency = radius_factor * separation_factor * coil_factor * 0.8  # Maximum 80% efficiency
        
        return min(efficiency, 1.0)
    
    def check_field_constraints(self, config: FieldConfiguration) -> Dict[str, float]:
        """Check field generation constraints"""
        
        violations = {}
        
        # Field strength constraint
        achievable_field = self._calculate_achievable_field_strength(config)
        if achievable_field < config.target_field_strength * 0.95:
            violations['field_strength'] = (config.target_field_strength - achievable_field) / config.target_field_strength
        
        # Superconductor critical current constraint
        sc_props = self.sc_properties[config.superconductor_type]
        coil_area = np.pi * (config.coil_radius * 0.1) ** 2
        j_operating = config.coil_current / (config.num_coils * coil_area)
        
        # Temperature-dependent critical current
        t_ratio = config.operating_temperature / sc_props['tc']
        if t_ratio < 1.0:
            jc = sc_props['jc0'] * (1 - t_ratio) ** 1.5
            if j_operating > jc * 0.8:  # 80% of critical current for safety
                violations['critical_current'] = (j_operating - jc * 0.8) / (jc * 0.8)
        
        # Critical field constraint
        if achievable_field > sc_props['bc2'] * 0.8:
            violations['critical_field'] = (achievable_field - sc_props['bc2'] * 0.8) / (sc_props['bc2'] * 0.8)
        
        # Resonance constraint (frequency limits)
        if config.resonant_frequency > 1e11:  # 100 GHz limit
            violations['frequency_limit'] = (config.resonant_frequency - 1e11) / 1e11
        
        # Quality factor realism constraint
        if config.quality_factor > 1e7:  # Realistic Q factor limit
            violations['quality_factor'] = (config.quality_factor - 1e7) / 1e7
        
        # Power constraint (must be achievable)
        total_power = self.calculate_field_efficiency(config)
        if total_power > self.base_energy * 2:  # Cannot exceed 2× original power
            violations['power_limit'] = (total_power - self.base_energy * 2) / (self.base_energy * 2)
        
        return violations
    
    def _calculate_achievable_field_strength(self, config: FieldConfiguration) -> float:
        """Calculate achievable magnetic field strength"""
        
        # Simplified field calculation for solenoid-like arrangement
        # B = μ₀ * n * I for a solenoid
        
        effective_turns = config.num_coils / config.coil_separation
        field_strength = self.mu0 * effective_turns * config.coil_current
        
        # Geometry factor for efficiency
        geometry_factor = 1.0 / (1.0 + (config.coil_radius / config.coil_separation) ** 2)
        
        return field_strength * geometry_factor
    
    def objective_function(self, params: np.ndarray) -> float:
        """Objective function for field optimization"""
        
        # Convert parameters to configuration
        config = self._params_to_config(params)
        
        # Calculate total energy
        total_energy = self.calculate_field_efficiency(config)
        
        # Check constraints
        violations = self.check_field_constraints(config)
        
        # Penalty for constraint violations
        penalty = 0.0
        for violation in violations.values():
            penalty += violation ** 2 * 1e12  # Large penalty for violations
        
        return total_energy + penalty
    
    def _params_to_config(self, params: np.ndarray) -> FieldConfiguration:
        """Convert parameter array to field configuration"""
        
        config = FieldConfiguration()
        
        config.num_coils = int(params[0])
        config.coil_radius = params[1]
        config.coil_separation = params[2]
        config.coil_current = params[3]
        config.resonant_frequency = params[4]
        config.quality_factor = params[5]
        config.coupling_efficiency = params[6]
        config.operating_temperature = params[7]
        
        return config
    
    def _config_to_params(self, config: FieldConfiguration) -> np.ndarray:
        """Convert field configuration to parameter array"""
        
        return np.array([
            config.num_coils,
            config.coil_radius,
            config.coil_separation,
            config.coil_current,
            config.resonant_frequency,
            config.quality_factor,
            config.coupling_efficiency,
            config.operating_temperature
        ])
    
    def optimize_superconducting_configuration(self) -> OptimizationResult:
        """Optimize superconducting field configuration"""
        
        logger.info("Optimizing superconducting field configuration...")
        
        # Parameter bounds
        bounds = [
            self.config.optimization_bounds['num_coils'],
            self.config.optimization_bounds['coil_radius'],
            self.config.optimization_bounds['coil_separation'],
            self.config.optimization_bounds['coil_current'],
            self.config.optimization_bounds['resonant_frequency'],
            self.config.optimization_bounds['quality_factor'],
            self.config.optimization_bounds['coupling_efficiency'],
            self.config.optimization_bounds['operating_temperature']
        ]
        
        # Initial configuration
        initial_params = self._config_to_params(self.config)
        original_energy = self.calculate_field_efficiency(self.config)
        
        import time
        start_time = time.time()
        
        # Run optimization
        result = differential_evolution(
            self.objective_function,
            bounds,
            maxiter=500,
            tol=1e-8,
            seed=42,
            workers=1,
            polish=True
        )
        
        optimization_time = time.time() - start_time
        
        # Extract results
        optimized_config = self._params_to_config(result.x)
        optimized_energy = self.calculate_field_efficiency(optimized_config)
        violations = self.check_field_constraints(optimized_config)
        
        # Calculate performance metrics
        field_strength = self._calculate_achievable_field_strength(optimized_config)
        
        optimization_result = OptimizationResult(
            optimized_config=optimized_config,
            original_energy=original_energy,
            optimized_energy=optimized_energy,
            energy_reduction_factor=original_energy / optimized_energy,
            field_efficiency=self._calculate_geometry_efficiency(optimized_config),
            power_consumption=optimized_energy,
            heat_generation=self.calculate_superconducting_losses(optimized_config),
            field_strength_achieved=field_strength,
            uniformity_achieved=0.98,  # Assume high uniformity with optimization
            stability_achieved=1e-6,   # Assume high stability
            optimization_time=optimization_time,
            convergence_achieved=result.success,
            constraint_violations=violations
        )
        
        logger.info(f"Superconducting optimization complete:")
        logger.info(f"  Energy reduction: {optimization_result.energy_reduction_factor:.2f}×")
        logger.info(f"  Field efficiency: {optimization_result.field_efficiency:.3f}")
        logger.info(f"  Optimization time: {optimization_time:.1f} seconds")
        
        return optimization_result
    
    def optimize_resonant_enhancement(self) -> OptimizationResult:
        """Optimize resonant field enhancement"""
        
        logger.info("Optimizing resonant field enhancement...")
        
        def resonant_objective(params: np.ndarray) -> float:
            config = self._params_to_config(params)
            
            # Focus on resonant parameters
            resonant_reduction = self.calculate_resonant_enhancement(config)
            base_efficiency = self.calculate_field_efficiency(config)
            
            total_energy = base_efficiency * resonant_reduction
            
            # Constraint penalties
            violations = self.check_field_constraints(config)
            penalty = sum(v ** 2 for v in violations.values()) * 1e12
            
            return total_energy + penalty
        
        # Bounds focused on resonant parameters
        bounds = [
            self.config.optimization_bounds['num_coils'],
            self.config.optimization_bounds['coil_radius'],
            self.config.optimization_bounds['coil_separation'],
            self.config.optimization_bounds['coil_current'],
            (1e9, 1e11),  # Focus on high frequency for resonance
            (1e5, 1e7),   # High Q factor for strong resonance
            (0.90, 0.99), # High coupling efficiency
            (4.2, 20.0)   # Low temperature for better superconducting properties
        ]
        
        initial_params = self._config_to_params(self.config)
        original_energy = self.calculate_field_efficiency(self.config)
        
        import time
        start_time = time.time()
        
        # Run resonant optimization
        result = differential_evolution(
            resonant_objective,
            bounds,
            maxiter=300,
            tol=1e-8,
            seed=42,
            workers=1,
            polish=True
        )
        
        optimization_time = time.time() - start_time
        
        # Extract results
        optimized_config = self._params_to_config(result.x)
        optimized_energy = self.calculate_field_efficiency(optimized_config)
        violations = self.check_field_constraints(optimized_config)
        
        field_strength = self._calculate_achievable_field_strength(optimized_config)
        
        optimization_result = OptimizationResult(
            optimized_config=optimized_config,
            original_energy=original_energy,
            optimized_energy=optimized_energy,
            energy_reduction_factor=original_energy / optimized_energy,
            field_efficiency=self._calculate_geometry_efficiency(optimized_config),
            power_consumption=optimized_energy,
            heat_generation=self.calculate_superconducting_losses(optimized_config),
            field_strength_achieved=field_strength,
            uniformity_achieved=0.96,  # Slightly lower due to resonant effects
            stability_achieved=1e-5,   # Resonant systems can be less stable
            optimization_time=optimization_time,
            convergence_achieved=result.success,
            constraint_violations=violations
        )
        
        logger.info(f"Resonant optimization complete:")
        logger.info(f"  Energy reduction: {optimization_result.energy_reduction_factor:.2f}×")
        logger.info(f"  Resonant enhancement: {self.calculate_resonant_enhancement(optimized_config):.2f}")
        logger.info(f"  Optimization time: {optimization_time:.1f} seconds")
        
        return optimization_result
    
    def optimize_hybrid_approach(self) -> OptimizationResult:
        """Optimize hybrid superconducting + resonant approach"""
        
        logger.info("Optimizing hybrid superconducting + resonant approach...")
        
        def hybrid_objective(params: np.ndarray) -> float:
            config = self._params_to_config(params)
            
            # Superconducting losses
            sc_losses = self.calculate_superconducting_losses(config)
            
            # Resonant enhancement
            resonant_factor = self.calculate_resonant_enhancement(config)
            
            # Geometry efficiency
            geometry_eff = self._calculate_geometry_efficiency(config)
            
            # Combined optimization
            total_energy = sc_losses * resonant_factor / geometry_eff
            
            # Multi-objective: balance energy, field strength, and stability
            field_strength = self._calculate_achievable_field_strength(config)
            field_penalty = max(0, (config.target_field_strength - field_strength) / config.target_field_strength) * 1e10
            
            # Constraint violations
            violations = self.check_field_constraints(config)
            constraint_penalty = sum(v ** 2 for v in violations.values()) * 1e12
            
            return total_energy + field_penalty + constraint_penalty
        
        # Full parameter bounds for hybrid optimization
        bounds = [
            self.config.optimization_bounds['num_coils'],
            self.config.optimization_bounds['coil_radius'],
            self.config.optimization_bounds['coil_separation'],
            self.config.optimization_bounds['coil_current'],
            self.config.optimization_bounds['resonant_frequency'],
            self.config.optimization_bounds['quality_factor'],
            self.config.optimization_bounds['coupling_efficiency'],
            self.config.optimization_bounds['operating_temperature']
        ]
        
        initial_params = self._config_to_params(self.config)
        original_energy = self.calculate_field_efficiency(self.config)
        
        import time
        start_time = time.time()
        
        # Run hybrid optimization
        result = differential_evolution(
            hybrid_objective,
            bounds,
            maxiter=800,
            tol=1e-9,
            seed=42,
            workers=1,
            polish=True
        )
        
        optimization_time = time.time() - start_time
        
        # Extract results
        optimized_config = self._params_to_config(result.x)
        optimized_energy = self.calculate_field_efficiency(optimized_config)
        violations = self.check_field_constraints(optimized_config)
        
        field_strength = self._calculate_achievable_field_strength(optimized_config)
        
        optimization_result = OptimizationResult(
            optimized_config=optimized_config,
            original_energy=original_energy,
            optimized_energy=optimized_energy,
            energy_reduction_factor=original_energy / optimized_energy,
            field_efficiency=self._calculate_geometry_efficiency(optimized_config),
            power_consumption=optimized_energy,
            heat_generation=self.calculate_superconducting_losses(optimized_config),
            field_strength_achieved=field_strength,
            uniformity_achieved=0.97,  # Balanced performance
            stability_achieved=5e-6,   # Good stability
            optimization_time=optimization_time,
            convergence_achieved=result.success,
            constraint_violations=violations
        )
        
        logger.info(f"Hybrid optimization complete:")
        logger.info(f"  Energy reduction: {optimization_result.energy_reduction_factor:.2f}×")
        logger.info(f"  Field efficiency: {optimization_result.field_efficiency:.3f}")
        logger.info(f"  Optimization time: {optimization_time:.1f} seconds")
        
        return optimization_result
    
    def run_comprehensive_field_optimization(self) -> Dict[str, OptimizationResult]:
        """Run comprehensive field optimization using multiple approaches"""
        
        logger.info("Running comprehensive field generation optimization...")
        
        results = {}
        
        # Method 1: Superconducting optimization
        try:
            results['superconducting'] = self.optimize_superconducting_configuration()
        except Exception as e:
            logger.error(f"Superconducting optimization failed: {e}")
            results['superconducting'] = None
        
        # Method 2: Resonant enhancement
        try:
            results['resonant'] = self.optimize_resonant_enhancement()
        except Exception as e:
            logger.error(f"Resonant optimization failed: {e}")
            results['resonant'] = None
        
        # Method 3: Hybrid approach
        try:
            results['hybrid'] = self.optimize_hybrid_approach()
        except Exception as e:
            logger.error(f"Hybrid optimization failed: {e}")
            results['hybrid'] = None
        
        # Select best result
        best_result = None
        best_reduction = 0
        best_method = None
        
        for method, result in results.items():
            if result and result.convergence_achieved and len(result.constraint_violations) == 0:
                if result.energy_reduction_factor > best_reduction:
                    best_reduction = result.energy_reduction_factor
                    best_result = result
                    best_method = method
        
        if best_result:
            logger.info(f"Best field optimization method: {best_method}")
            logger.info(f"Best energy reduction: {best_reduction:.2f}×")
            logger.info(f"Target achieved: {'YES' if best_reduction >= self.target_reduction else 'NO'}")
        else:
            logger.warning("No successful field optimization found")
        
        self.optimization_results = results
        return results
    
    def visualize_field_optimization(self, save_path: Optional[str] = None):
        """Create comprehensive visualization of field optimization results"""
        
        if not self.optimization_results:
            self.run_comprehensive_field_optimization()
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Energy reduction comparison
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_energy_reduction_comparison(ax1)
        
        # 2. Field configuration visualization
        ax2 = fig.add_subplot(gs[0, 2:], projection='3d')
        self._plot_field_configuration(ax2)
        
        # 3. Superconducting performance
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_superconducting_performance(ax3)
        
        # 4. Resonant enhancement analysis
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_resonant_enhancement(ax4)
        
        # 5. Parameter optimization
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_parameter_optimization(ax5)
        
        # 6. Constraint satisfaction
        ax6 = fig.add_subplot(gs[2, 2:])
        self._plot_constraint_satisfaction(ax6)
        
        plt.suptitle('Warp Bubble Field Generation Optimization Results', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Field optimization visualization saved to: {save_path}")
        
        plt.show()
    
    def _plot_energy_reduction_comparison(self, ax):
        """Plot energy reduction comparison across methods"""
        
        methods = []
        original_energies = []
        optimized_energies = []
        reduction_factors = []
        
        for method, result in self.optimization_results.items():
            if result:
                methods.append(method.title())
                original_energies.append(result.original_energy / 1e9)
                optimized_energies.append(result.optimized_energy / 1e6)  # Show in millions
                reduction_factors.append(result.energy_reduction_factor)
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, original_energies, width, label='Original Energy (Billion J)', 
                      alpha=0.7, color='red')
        
        # Scale for second y-axis
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, optimized_energies, width, label='Optimized Energy (Million J)', 
                       alpha=0.7, color='green')
        
        ax.set_xlabel('Optimization Methods')
        ax.set_ylabel('Original Energy (Billion J)', color='red')
        ax2.set_ylabel('Optimized Energy (Million J)', color='green')
        ax.set_title('Field Generation Energy Reduction')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        
        # Add reduction factor labels
        for i, factor in enumerate(reduction_factors):
            ax.text(i, max(original_energies) * 1.1, f'{factor:.1f}×', 
                   ha='center', va='bottom', fontweight='bold')
        
        # Add target line
        ax.axhline(y=self.target_energy/1e9, color='blue', linestyle='--', 
                  linewidth=2, label=f'Target ({self.target_reduction}× reduction)')
        
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
    def _plot_field_configuration(self, ax):
        """Plot 3D field configuration"""
        
        # Find best result
        best_result = None
        for result in self.optimization_results.values():
            if result and result.convergence_achieved:
                if best_result is None or result.energy_reduction_factor > best_result.energy_reduction_factor:
                    best_result = result
        
        if not best_result:
            ax.text(0.5, 0.5, 0.5, 'No successful optimization', ha='center', va='center')
            return
        
        config = best_result.optimized_config
        
        # Create coil positions
        angles = np.linspace(0, 2*np.pi, config.num_coils, endpoint=False)
        z_positions = np.linspace(-config.coil_separation/2, config.coil_separation/2, 3)
        
        for z in z_positions:
            x_coils = config.coil_radius * np.cos(angles)
            y_coils = config.coil_radius * np.sin(angles)
            z_coils = np.full_like(x_coils, z)
            
            ax.scatter(x_coils, y_coils, z_coils, s=100, alpha=0.7, 
                      c='blue' if z == 0 else 'lightblue')
        
        # Draw bubble outline
        u = np.linspace(0, 2*np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x_bubble = 2.3 * np.outer(np.sin(v), np.cos(u))
        y_bubble = 0.9 * np.outer(np.sin(v), np.sin(u))
        z_bubble = 0.75 * np.outer(np.cos(v), np.ones(np.size(u)))
        
        ax.plot_surface(x_bubble, y_bubble, z_bubble, alpha=0.3, color='yellow')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Optimized Field Configuration\n{config.num_coils} coils, {config.coil_radius:.1f}m radius')
    
    def _plot_superconducting_performance(self, ax):
        """Plot superconducting performance metrics"""
        
        # Temperature vs critical current for different superconductors
        temperatures = np.linspace(4.2, 100, 100)
        
        for sc_type, props in self.sc_properties.items():
            tc = props['tc']
            jc0 = props['jc0']
            
            # Critical current vs temperature
            jc_values = []
            for T in temperatures:
                if T >= tc:
                    jc_values.append(0)
                else:
                    jc = jc0 * (1 - T/tc) ** 1.5
                    jc_values.append(jc / 1e9)  # Convert to GA/m²
            
            ax.plot(temperatures, jc_values, label=f'{sc_type} (Tc={tc}K)', linewidth=2)
        
        # Mark operating points
        for method, result in self.optimization_results.items():
            if result and result.convergence_achieved:
                config = result.optimized_config
                T_op = config.operating_temperature
                
                # Calculate operating current density
                coil_area = np.pi * (config.coil_radius * 0.1) ** 2
                j_op = config.coil_current / (config.num_coils * coil_area) / 1e9
                
                ax.scatter(T_op, j_op, s=100, label=f'{method} operating point', marker='o')
        
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Critical Current Density (GA/m²)')
        ax.set_title('Superconductor Performance vs Temperature')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
    
    def _plot_resonant_enhancement(self, ax):
        """Plot resonant enhancement analysis"""
        
        methods = []
        frequencies = []
        q_factors = []
        enhancements = []
        
        for method, result in self.optimization_results.items():
            if result:
                methods.append(method.title())
                config = result.optimized_config
                frequencies.append(config.resonant_frequency / 1e9)  # GHz
                q_factors.append(config.quality_factor / 1e6)  # Millions
                enhancement = self.calculate_resonant_enhancement(config)
                enhancements.append(1.0 / enhancement)  # Enhancement factor
        
        # Create scatter plot
        scatter = ax.scatter(frequencies, q_factors, s=[e*50 for e in enhancements], 
                           c=enhancements, cmap='viridis', alpha=0.7, edgecolors='black')
        
        ax.set_xlabel('Resonant Frequency (GHz)')
        ax.set_ylabel('Quality Factor (Millions)')
        ax.set_title('Resonant Enhancement Analysis')
        ax.grid(True, alpha=0.3)
        
        # Add method labels
        for i, method in enumerate(methods):
            ax.annotate(method, (frequencies[i], q_factors[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        # Colorbar for enhancement
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Enhancement Factor', rotation=270, labelpad=15)
    
    def _plot_parameter_optimization(self, ax):
        """Plot parameter optimization results"""
        
        # Find best result
        best_result = None
        for result in self.optimization_results.values():
            if result and result.convergence_achieved:
                if best_result is None or result.energy_reduction_factor > best_result.energy_reduction_factor:
                    best_result = result
        
        if not best_result:
            ax.text(0.5, 0.5, 'No successful optimization', ha='center', va='center')
            return
        
        param_names = ['Coils', 'Radius (m)', 'Separation (m)', 'Current (MA)', 
                      'Freq (GHz)', 'Q (M)', 'Coupling', 'Temp (K)']
        
        original_values = [
            self.config.num_coils,
            self.config.coil_radius,
            self.config.coil_separation,
            self.config.coil_current / 1e6,
            self.config.resonant_frequency / 1e9,
            self.config.quality_factor / 1e6,
            self.config.coupling_efficiency,
            self.config.operating_temperature
        ]
        
        optimized_values = [
            best_result.optimized_config.num_coils,
            best_result.optimized_config.coil_radius,
            best_result.optimized_config.coil_separation,
            best_result.optimized_config.coil_current / 1e6,
            best_result.optimized_config.resonant_frequency / 1e9,
            best_result.optimized_config.quality_factor / 1e6,
            best_result.optimized_config.coupling_efficiency,
            best_result.optimized_config.operating_temperature
        ]
        
        x = np.arange(len(param_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, original_values, width, label='Original', alpha=0.7)
        bars2 = ax.bar(x + width/2, optimized_values, width, label='Optimized', alpha=0.7)
        
        ax.set_xlabel('Parameters')
        ax.set_ylabel('Parameter Values')
        ax.set_title('Field Generation Parameter Optimization')
        ax.set_xticks(x)
        ax.set_xticklabels([name.replace(' ', '\n') for name in param_names], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_constraint_satisfaction(self, ax):
        """Plot constraint satisfaction analysis"""
        
        methods = []
        violation_counts = []
        max_violations = []
        field_achievements = []
        
        for method, result in self.optimization_results.items():
            if result:
                methods.append(method.title())
                violation_counts.append(len(result.constraint_violations))
                max_violations.append(max(result.constraint_violations.values()) 
                                    if result.constraint_violations else 0)
                
                # Field strength achievement percentage
                target_field = 10.0  # Target field strength
                achievement = (result.field_strength_achieved / target_field) * 100
                field_achievements.append(min(achievement, 120))  # Cap at 120%
        
        x = np.arange(len(methods))
        width = 0.25
        
        bars1 = ax.bar(x - width, violation_counts, width, label='Violation Count', alpha=0.7, color='red')
        bars2 = ax.bar(x, max_violations, width, label='Max Violation', alpha=0.7, color='orange')
        bars3 = ax.bar(x + width, [f/100 for f in field_achievements], width, 
                      label='Field Achievement', alpha=0.7, color='green')
        
        ax.set_xlabel('Optimization Methods')
        ax.set_ylabel('Metrics')
        ax.set_title('Constraint Satisfaction and Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add field achievement percentage labels
        for i, achievement in enumerate(field_achievements):
            ax.text(i + width, achievement/100 + 0.05, f'{achievement:.0f}%', 
                   ha='center', va='bottom', fontsize=9)

def main():
    """Main execution function for field generation optimization"""
    
    print("=" * 80)
    print("WARP BUBBLE FIELD GENERATION OPTIMIZATION")
    print("Phase 2 Implementation: Advanced Field Generation Optimization")
    print("=" * 80)
    
    # Initialize field optimizer
    optimizer = FieldGenerationOptimizer()
    
    print(f"\n🎯 OPTIMIZATION TARGET:")
    print(f"Current Energy: {optimizer.base_energy/1e9:.3f} billion J")
    print(f"Target Energy: {optimizer.target_energy/1e6:.0f} million J")
    print(f"Required Reduction: {optimizer.target_reduction}×")
    
    # Run comprehensive field optimization
    print(f"\n⚡ RUNNING COMPREHENSIVE FIELD OPTIMIZATION...")
    results = optimizer.run_comprehensive_field_optimization()
    
    # Analyze results
    print(f"\n📊 FIELD OPTIMIZATION RESULTS:")
    
    successful_methods = 0
    best_reduction = 0
    best_method = None
    best_result = None
    
    for method, result in results.items():
        print(f"\n{method.upper()}:")
        if result:
            print(f"   Energy Reduction: {result.energy_reduction_factor:.2f}×")
            print(f"   Original Energy: {result.original_energy/1e9:.3f} billion J")
            print(f"   Optimized Energy: {result.optimized_energy/1e6:.0f} million J")
            print(f"   Field Efficiency: {result.field_efficiency:.3f}")
            print(f"   Field Strength: {result.field_strength_achieved:.1f} T")
            print(f"   Optimization Time: {result.optimization_time:.1f} seconds")
            print(f"   Constraint Violations: {len(result.constraint_violations)}")
            print(f"   Success: {'✅ YES' if result.convergence_achieved and len(result.constraint_violations) == 0 else '❌ NO'}")
            
            if result.convergence_achieved and len(result.constraint_violations) == 0:
                successful_methods += 1
                if result.energy_reduction_factor > best_reduction:
                    best_reduction = result.energy_reduction_factor
                    best_method = method
                    best_result = result
        else:
            print(f"   Status: ❌ FAILED")
    
    # Summary
    print(f"\n🏆 FIELD OPTIMIZATION SUMMARY:")
    print(f"Successful Methods: {successful_methods}/{len(results)}")
    
    if best_result:
        print(f"Best Method: {best_method}")
        print(f"Best Energy Reduction: {best_reduction:.2f}×")
        print(f"Target Achievement: {'✅ YES' if best_reduction >= optimizer.target_reduction else '❌ NO'}")
        
        if best_reduction >= optimizer.target_reduction:
            print(f"\n🎉 TARGET ACHIEVED! Field optimization successful!")
            print(f"Energy reduced from {optimizer.base_energy/1e9:.3f}B J to {best_result.optimized_energy/1e6:.0f}M J")
        else:
            shortfall = optimizer.target_reduction / best_reduction
            print(f"\n⚠️ Target not fully achieved. Additional {shortfall:.1f}× reduction needed.")
        
        # Optimized configuration details
        opt_config = best_result.optimized_config
        print(f"\n⚙️ OPTIMIZED FIELD CONFIGURATION:")
        print(f"   Number of Coils: {opt_config.num_coils} (was {optimizer.config.num_coils})")
        print(f"   Coil Radius: {opt_config.coil_radius:.2f} m (was {optimizer.config.coil_radius:.2f} m)")
        print(f"   Coil Current: {opt_config.coil_current/1e6:.1f} MA (was {optimizer.config.coil_current/1e6:.1f} MA)")
        print(f"   Operating Temperature: {opt_config.operating_temperature:.1f} K (was {optimizer.config.operating_temperature:.1f} K)")
        print(f"   Resonant Frequency: {opt_config.resonant_frequency/1e9:.1f} GHz (was {optimizer.config.resonant_frequency/1e9:.1f} GHz)")
        print(f"   Quality Factor: {opt_config.quality_factor/1e6:.1f} M (was {optimizer.config.quality_factor/1e6:.1f} M)")
        print(f"   Coupling Efficiency: {opt_config.coupling_efficiency:.3f} (was {optimizer.config.coupling_efficiency:.3f})")
    else:
        print(f"❌ No successful field optimization achieved")
    
    # Generate visualization
    print(f"\n📊 GENERATING FIELD OPTIMIZATION VISUALIZATION...")
    viz_path = "energy_optimization/field_optimization_results.png"
    optimizer.visualize_field_optimization(viz_path)
    
    # Save optimization results
    results_path = "energy_optimization/field_optimization_report.json"
    
    # Prepare results for JSON serialization
    json_results = {}
    for method, result in results.items():
        if result:
            json_results[method] = {
                'energy_reduction_factor': result.energy_reduction_factor,
                'original_energy': result.original_energy,
                'optimized_energy': result.optimized_energy,
                'field_efficiency': result.field_efficiency,
                'field_strength_achieved': result.field_strength_achieved,
                'optimization_time': result.optimization_time,
                'convergence_achieved': result.convergence_achieved,
                'constraint_violations': result.constraint_violations,
                'optimized_config': {
                    'num_coils': result.optimized_config.num_coils,
                    'coil_radius': result.optimized_config.coil_radius,
                    'coil_separation': result.optimized_config.coil_separation,
                    'coil_current': result.optimized_config.coil_current,
                    'resonant_frequency': result.optimized_config.resonant_frequency,
                    'quality_factor': result.optimized_config.quality_factor,
                    'coupling_efficiency': result.optimized_config.coupling_efficiency,
                    'operating_temperature': result.optimized_config.operating_temperature,
                    'superconductor_type': result.optimized_config.superconductor_type
                }
            }
        else:
            json_results[method] = None
    
    report = {
        'optimization_summary': {
            'target_reduction': optimizer.target_reduction,
            'best_reduction_achieved': best_reduction,
            'target_achieved': best_reduction >= optimizer.target_reduction if best_result else False,
            'best_method': best_method,
            'successful_methods': successful_methods,
            'total_methods': len(results)
        },
        'optimization_results': json_results,
        'original_config': {
            'num_coils': optimizer.config.num_coils,
            'coil_radius': optimizer.config.coil_radius,
            'coil_separation': optimizer.config.coil_separation,
            'coil_current': optimizer.config.coil_current,
            'resonant_frequency': optimizer.config.resonant_frequency,
            'quality_factor': optimizer.config.quality_factor,
            'coupling_efficiency': optimizer.config.coupling_efficiency,
            'operating_temperature': optimizer.config.operating_temperature,
            'superconductor_type': optimizer.config.superconductor_type
        }
    }
    
    with open(results_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Field optimization report saved to: {results_path}")
    
    print("\n" + "=" * 80)
    print("FIELD GENERATION OPTIMIZATION COMPLETE")
    if best_result and best_reduction >= optimizer.target_reduction:
        print("STATUS: ✅ TARGET ACHIEVED - 6× field generation energy reduction successful!")
    elif best_result:
        print(f"STATUS: ⚠️ PARTIAL SUCCESS - {best_reduction:.1f}× reduction achieved")
    else:
        print("STATUS: ❌ OPTIMIZATION FAILED")
    print("=" * 80)

if __name__ == "__main__":
    main()

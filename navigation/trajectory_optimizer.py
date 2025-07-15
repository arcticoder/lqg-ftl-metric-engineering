"""
Trajectory Physics Engine for LQG FTL Metric Engineering

This module provides physics-constrained flight path optimization with spacetime
geodesic optimization and energy minimization for LQG FTL navigation.

Author: LQG FTL Navigation Team
License: MIT
"""

import numpy as np
import scipy.optimize
from scipy.integrate import solve_ivp
from typing import Dict, List, Tuple, Optional, Callable
import math
from dataclasses import dataclass
from .flight_path_format import TrajectoryPoint, FlightPathFormat

# Physical constants
C_LIGHT = 299792458.0  # Speed of light in m/s
G_NEWTON = 6.674e-11   # Gravitational constant
PLANCK_LENGTH = 1.616e-35  # Planck length in meters
PLANCK_ENERGY = 1.956e9  # Planck energy in Joules

@dataclass
class GravitationalBody:
    """Gravitational body for trajectory calculations"""
    name: str
    mass: float  # kg
    position: Tuple[float, float, float]  # meters
    radius: float  # meters
    
class SpacetimeMetric:
    """Spacetime metric calculations for LQG FTL navigation"""
    
    def __init__(self, gravitational_bodies: List[GravitationalBody] = None):
        self.gravitational_bodies = gravitational_bodies or []
    
    def calculate_metric_tensor(self, position: np.ndarray) -> np.ndarray:
        """Calculate spacetime metric tensor at given position"""
        # Start with Minkowski metric
        g_metric = np.diag([-1, 1, 1, 1])
        
        # Add gravitational perturbations
        for body in self.gravitational_bodies:
            body_pos = np.array(body.position)
            r_vec = position - body_pos
            r = np.linalg.norm(r_vec)
            
            if r > body.radius:  # Outside the body
                # Schwarzschild metric correction
                rs = 2 * G_NEWTON * body.mass / (C_LIGHT**2)  # Schwarzschild radius
                if r > 1.1 * rs:  # Outside event horizon with safety margin
                    g_metric[0, 0] *= -(1 - rs/r)
                    g_metric[1, 1] *= 1 / (1 - rs/r)
                    g_metric[2, 2] *= r**2
                    g_metric[3, 3] *= r**2 * np.sin(np.arccos(r_vec[2]/r))**2
        
        return g_metric
    
    def calculate_christoffel_symbols(self, position: np.ndarray) -> np.ndarray:
        """Calculate Christoffel symbols for geodesic equations"""
        # Simplified calculation for demonstration
        # In practice, would need numerical derivatives of metric tensor
        gamma = np.zeros((4, 4, 4))
        
        # Add gravitational field effects
        for body in self.gravitational_bodies:
            body_pos = np.array(body.position)
            r_vec = position - body_pos
            r = np.linalg.norm(r_vec)
            
            if r > body.radius:
                rs = 2 * G_NEWTON * body.mass / (C_LIGHT**2)
                if r > 1.1 * rs:
                    # Simplified Christoffel symbols for Schwarzschild metric
                    gamma[0, 0, 1] = rs / (2 * r**2 * (1 - rs/r))
                    gamma[1, 0, 0] = rs * (1 - rs/r) / (2 * r**2)
        
        return gamma

class TrajectoryOptimizer:
    """Physics-constrained trajectory optimizer for LQG FTL missions"""
    
    def __init__(self, spacetime_metric: SpacetimeMetric = None):
        self.spacetime_metric = spacetime_metric or SpacetimeMetric()
        self.energy_budget = 1e20  # Default energy budget in Joules
        self.max_warp_factor = 100.0
        self.safety_margin = 0.1
    
    def calculate_warp_energy_density(self, warp_factor: float) -> float:
        """Calculate energy density required for given warp factor"""
        if warp_factor <= 1.0:
            return 1.23e15  # Base energy density for normal space
        
        # Energy scales with warp factor squared (simplified model)
        base_energy = 1.23e15
        return base_energy * (warp_factor**2.5)  # Slightly superquadratic
    
    def calculate_gravitational_acceleration(self, position: np.ndarray) -> np.ndarray:
        """Calculate gravitational acceleration at given position"""
        acceleration = np.zeros(3)
        
        for body in self.spacetime_metric.gravitational_bodies:
            body_pos = np.array(body.position)
            r_vec = position - body_pos
            r = np.linalg.norm(r_vec)
            
            if r > body.radius:
                # Newtonian gravity
                acc_magnitude = G_NEWTON * body.mass / r**2
                acc_direction = -r_vec / r  # Towards the body
                acceleration += acc_magnitude * acc_direction
        
        return acceleration
    
    def geodesic_equation(self, t: float, state: np.ndarray) -> np.ndarray:
        """Geodesic equation for spacetime trajectory"""
        # state = [x, y, z, vx, vy, vz]
        position = state[:3]
        velocity = state[3:6]
        
        # Calculate gravitational acceleration
        acceleration = self.calculate_gravitational_acceleration(position)
        
        # Return derivatives [vx, vy, vz, ax, ay, az]
        return np.concatenate([velocity, acceleration])
    
    def optimize_trajectory(self, 
                          start_position: np.ndarray,
                          end_position: np.ndarray,
                          mission_duration: float,
                          num_waypoints: int = 100) -> FlightPathFormat:
        """Optimize trajectory between start and end positions"""
        
        print(f"🎯 Optimizing trajectory from {start_position} to {end_position}")
        print(f"Mission duration: {mission_duration/86400:.1f} days")
        
        # Create time points
        t_span = (0, mission_duration)
        t_eval = np.linspace(0, mission_duration, num_waypoints)
        
        # Calculate required average velocity
        distance_vector = end_position - start_position
        distance = np.linalg.norm(distance_vector)
        required_velocity = distance / mission_duration
        
        print(f"Distance: {distance/9.461e15:.2f} light years")
        print(f"Required average velocity: {required_velocity/C_LIGHT:.2f}c")
        
        # Determine warp factor requirements
        if required_velocity > C_LIGHT:
            min_warp_factor = required_velocity / C_LIGHT
            if min_warp_factor > self.max_warp_factor:
                print(f"⚠️  Warning: Required warp factor {min_warp_factor:.1f} exceeds maximum {self.max_warp_factor}")
                min_warp_factor = self.max_warp_factor
        else:
            min_warp_factor = 1.0
        
        # Initial velocity (start from rest)
        initial_velocity = np.array([0.0, 0.0, 0.0])
        initial_state = np.concatenate([start_position, initial_velocity])
        
        # Solve geodesic equation for natural trajectory
        print("🌌 Calculating spacetime geodesic...")
        try:
            solution = solve_ivp(
                self.geodesic_equation,
                t_span,
                initial_state,
                t_eval=t_eval,
                method='RK45',
                rtol=1e-8
            )
            
            if not solution.success:
                print("⚠️  Geodesic integration failed, using direct trajectory")
                positions = self._create_direct_trajectory(start_position, end_position, t_eval)
                velocities = self._calculate_velocities_from_positions(positions, t_eval)
            else:
                positions = solution.y[:3].T
                velocities = solution.y[3:6].T
                
        except Exception as e:
            print(f"⚠️  Geodesic calculation error: {e}, using direct trajectory")
            positions = self._create_direct_trajectory(start_position, end_position, t_eval)
            velocities = self._calculate_velocities_from_positions(positions, t_eval)
        
        # Create optimized flight path
        flight_path = FlightPathFormat()
        flight_path.mission_metadata.update({
            'mission_id': f"OPTIMIZED_{int(t_span[1])}",
            'optimization_method': 'spacetime_geodesic',
            'energy_budget': self.energy_budget,
            'max_warp_factor': self.max_warp_factor
        })
        
        print("🚀 Generating optimized trajectory points...")
        
        # Generate trajectory points with physics constraints
        for i, t in enumerate(t_eval):
            position = positions[i]
            velocity = velocities[i]
            
            # Calculate required warp factor
            velocity_magnitude = np.linalg.norm(velocity)
            if velocity_magnitude > C_LIGHT:
                warp_factor = velocity_magnitude / C_LIGHT
            else:
                warp_factor = 1.0
            
            # Apply warp factor limits
            warp_factor = min(warp_factor, self.max_warp_factor)
            
            # Calculate energy density
            energy_density = self.calculate_warp_energy_density(warp_factor)
            
            # Calculate gravitational field
            gravitational_field = self.calculate_gravitational_acceleration(position)
            
            # Calculate stability factor
            stability_factor = self._calculate_stability_factor(warp_factor, energy_density)
            
            # Create trajectory point
            flight_path.create_trajectory_point(
                timestamp=t,
                position=tuple(position),
                velocity=tuple(velocity),
                warp_factor=warp_factor,
                energy_density=energy_density,
                gravitational_field=tuple(gravitational_field),
                metric_signature='(-,+,+,+)',
                stability_factor=stability_factor
            )
        
        # Validate and optimize energy consumption
        self._optimize_energy_consumption(flight_path)
        
        return flight_path
    
    def _create_direct_trajectory(self, start_pos: np.ndarray, end_pos: np.ndarray, 
                                t_eval: np.ndarray) -> np.ndarray:
        """Create direct trajectory as fallback"""
        positions = np.zeros((len(t_eval), 3))
        total_time = t_eval[-1]
        
        for i, t in enumerate(t_eval):
            # Smooth acceleration/deceleration profile
            if t < total_time * 0.1:  # Acceleration phase
                frac = t / (total_time * 0.1)
                progress = 0.5 * frac**2
            elif t > total_time * 0.9:  # Deceleration phase
                frac = (t - total_time * 0.9) / (total_time * 0.1)
                progress = 0.8 + 0.2 * (2 * frac - frac**2)
            else:  # Cruise phase
                frac = (t - total_time * 0.1) / (total_time * 0.8)
                progress = 0.05 + 0.75 * frac
            
            positions[i] = start_pos + progress * (end_pos - start_pos)
        
        return positions
    
    def _calculate_velocities_from_positions(self, positions: np.ndarray, 
                                           t_eval: np.ndarray) -> np.ndarray:
        """Calculate velocities from position trajectory"""
        velocities = np.zeros_like(positions)
        dt = t_eval[1] - t_eval[0]
        
        # Forward difference for first point
        velocities[0] = (positions[1] - positions[0]) / dt
        
        # Central difference for middle points
        for i in range(1, len(positions) - 1):
            velocities[i] = (positions[i+1] - positions[i-1]) / (2 * dt)
        
        # Backward difference for last point
        velocities[-1] = (positions[-1] - positions[-2]) / dt
        
        return velocities
    
    def _calculate_stability_factor(self, warp_factor: float, energy_density: float) -> float:
        """Calculate trajectory stability factor"""
        # Stability decreases with higher warp factors and energy densities
        warp_stability = max(0.5, 1.0 - (warp_factor - 1.0) / 100.0)
        energy_stability = max(0.5, 1.0 - energy_density / (1e20))
        
        return min(warp_stability, energy_stability)
    
    def _optimize_energy_consumption(self, flight_path: FlightPathFormat) -> None:
        """Optimize energy consumption across trajectory"""
        print("⚡ Optimizing energy consumption...")
        
        total_energy = 0.0
        for point in flight_path.trajectory_points:
            total_energy += point.energy_density
        
        # If over budget, scale down warp factors
        if total_energy > self.energy_budget:
            scale_factor = self.energy_budget / total_energy
            print(f"🔧 Scaling energy consumption by factor {scale_factor:.3f}")
            
            for point in flight_path.trajectory_points:
                # Reduce warp factor to meet energy constraints
                new_warp_factor = point.warp_factor * math.sqrt(scale_factor)
                point.warp_factor = max(1.0, new_warp_factor)
                point.energy_density = self.calculate_warp_energy_density(point.warp_factor)
                point.stability_factor = self._calculate_stability_factor(
                    point.warp_factor, point.energy_density)
    
    def analyze_trajectory_efficiency(self, flight_path: FlightPathFormat) -> Dict[str, float]:
        """Analyze trajectory efficiency metrics"""
        stats = flight_path.get_trajectory_statistics()
        
        # Calculate efficiency metrics
        theoretical_min_energy = (stats['total_distance_m'] / C_LIGHT) * 1e15  # Simplified
        actual_energy = stats['total_energy_j']
        energy_efficiency = theoretical_min_energy / actual_energy if actual_energy > 0 else 0
        
        # Time efficiency (compared to light speed)
        light_travel_time = stats['total_distance_m'] / C_LIGHT
        time_efficiency = light_travel_time / stats['mission_duration_s']
        
        return {
            'energy_efficiency': energy_efficiency,
            'time_efficiency': time_efficiency,
            'warp_utilization': stats['max_warp_factor'] / self.max_warp_factor,
            'average_stability': np.mean([p.stability_factor or 0.5 
                                        for p in flight_path.trajectory_points]),
            'trajectory_smoothness': self._calculate_trajectory_smoothness(flight_path)
        }
    
    def _calculate_trajectory_smoothness(self, flight_path: FlightPathFormat) -> float:
        """Calculate trajectory smoothness metric"""
        if len(flight_path.trajectory_points) < 3:
            return 1.0
        
        acceleration_changes = []
        for i in range(2, len(flight_path.trajectory_points)):
            p1 = flight_path.trajectory_points[i-2]
            p2 = flight_path.trajectory_points[i-1]
            p3 = flight_path.trajectory_points[i]
            
            # Calculate acceleration between points
            dt1 = p2.timestamp - p1.timestamp
            dt2 = p3.timestamp - p2.timestamp
            
            if dt1 > 0 and dt2 > 0:
                v1 = np.array(p2.velocity) - np.array(p1.velocity)
                v2 = np.array(p3.velocity) - np.array(p2.velocity)
                
                a1 = v1 / dt1
                a2 = v2 / dt2
                
                acceleration_change = np.linalg.norm(a2 - a1)
                acceleration_changes.append(acceleration_change)
        
        if not acceleration_changes:
            return 1.0
        
        # Smoothness is inverse of acceleration variation
        avg_acceleration_change = np.mean(acceleration_changes)
        smoothness = 1.0 / (1.0 + avg_acceleration_change / C_LIGHT)
        
        return smoothness

def create_solar_system_bodies() -> List[GravitationalBody]:
    """Create gravitational bodies for Solar System"""
    return [
        GravitationalBody("Sun", 1.989e30, (0, 0, 0), 6.96e8),
        GravitationalBody("Earth", 5.972e24, (1.496e11, 0, 0), 6.371e6),
        GravitationalBody("Jupiter", 1.898e27, (7.785e11, 0, 0), 6.9911e7),
    ]

def main():
    """Demonstration of trajectory optimization"""
    print("🌌 LQG FTL Trajectory Physics Engine Demo")
    print("="*50)
    
    # Create spacetime metric with gravitational bodies
    solar_system = create_solar_system_bodies()
    spacetime_metric = SpacetimeMetric(solar_system)
    
    # Create trajectory optimizer
    optimizer = TrajectoryOptimizer(spacetime_metric)
    optimizer.energy_budget = 1.5e20  # Joules
    optimizer.max_warp_factor = 75.0
    
    # Define mission parameters
    earth_position = np.array([1.496e11, 0, 0])  # Earth's orbital position
    proxima_position = np.array([4.01e16, 0, 0])  # Proxima Centauri position
    mission_duration = 365.25 * 86400  # 1 year
    
    # Optimize trajectory
    optimized_trajectory = optimizer.optimize_trajectory(
        earth_position, 
        proxima_position, 
        mission_duration,
        num_waypoints=50
    )
    
    # Analyze efficiency
    efficiency = optimizer.analyze_trajectory_efficiency(optimized_trajectory)
    
    # Display results
    stats = optimized_trajectory.get_trajectory_statistics()
    validation = optimized_trajectory.validate_trajectory_physics()
    
    print(f"\n📊 Optimized Trajectory Results:")
    print(f"Total Distance: {stats['total_distance_ly']:.2f} light years")
    print(f"Mission Duration: {stats['mission_duration_days']:.1f} days")
    print(f"Max Velocity: {stats['max_velocity_c']:.1f}c")
    print(f"Max Warp Factor: {stats['max_warp_factor']:.1f}")
    print(f"Total Energy: {stats['total_energy_j']:.2e} J")
    
    print(f"\n⚡ Efficiency Analysis:")
    print(f"Energy Efficiency: {efficiency['energy_efficiency']:.4f}")
    print(f"Time Efficiency: {efficiency['time_efficiency']:.4f}")
    print(f"Warp Utilization: {efficiency['warp_utilization']:.2f}")
    print(f"Average Stability: {efficiency['average_stability']:.3f}")
    print(f"Trajectory Smoothness: {efficiency['trajectory_smoothness']:.3f}")
    
    print(f"\n🔬 Physics Validation:")
    for constraint, valid in validation.items():
        status = "✅ PASS" if valid else "❌ FAIL"
        print(f"{constraint}: {status}")
    
    # Export optimized trajectory
    output_file = "optimized_earth_proxima_trajectory.ndjson"
    optimized_trajectory.export_ndjson(output_file)
    print(f"\n💾 Optimized trajectory exported to: {output_file}")

if __name__ == "__main__":
    main()

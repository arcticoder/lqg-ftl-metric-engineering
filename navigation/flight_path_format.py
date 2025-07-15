"""
Flight Path Format Module for LQG FTL Metric Engineering

This module provides standardized trajectory data format for LQG FTL missions
using Newline-delimited JSON (NDJSON) for streaming-compatible real-time updates.

Author: LQG FTL Navigation Team
License: MIT
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Iterator
from dataclasses import dataclass, asdict
import math
from datetime import datetime, timedelta
import io

# Physical constants
C_LIGHT = 299792458.0  # Speed of light in m/s
G_NEWTON = 6.674e-11   # Gravitational constant
PLANCK_LENGTH = 1.616e-35  # Planck length in meters

@dataclass
class TrajectoryPoint:
    """Single trajectory point with spacetime coordinates and warp parameters"""
    timestamp: float                    # Time coordinate (seconds)
    position: Tuple[float, float, float]  # Spatial coordinates (meters)
    velocity: Tuple[float, float, float]  # Velocity vector (m/s)
    warp_factor: float                  # Warp field strength (dimensionless)
    energy_density: float               # Energy density (J/m³)
    gravitational_field: Optional[Tuple[float, float, float]] = None  # g-field (m/s²)
    metric_signature: Optional[str] = None  # Spacetime metric signature
    stability_factor: Optional[float] = None  # Trajectory stability metric
    
    def to_ndjson_line(self) -> str:
        """Convert trajectory point to NDJSON line format"""
        data = asdict(self)
        # Convert tuples to lists for JSON serialization
        data['position'] = list(data['position'])
        data['velocity'] = list(data['velocity'])
        if data['gravitational_field']:
            data['gravitational_field'] = list(data['gravitational_field'])
        return json.dumps(data, separators=(',', ':'))
    
    @classmethod
    def from_ndjson_line(cls, line: str) -> 'TrajectoryPoint':
        """Create trajectory point from NDJSON line"""
        data = json.loads(line.strip())
        # Convert lists back to tuples
        data['position'] = tuple(data['position'])
        data['velocity'] = tuple(data['velocity'])
        if data.get('gravitational_field'):
            data['gravitational_field'] = tuple(data['gravitational_field'])
        return cls(**data)

class FlightPathFormat:
    """NDJSON flight path format handler for LQG FTL missions"""
    
    def __init__(self):
        self.trajectory_points: List[TrajectoryPoint] = []
        self.mission_metadata = {
            'mission_id': None,
            'vessel_id': None,
            'origin': None,
            'destination': None,
            'total_energy_budget': None,
            'mission_duration': None,
            'created_timestamp': datetime.now().isoformat(),
            'format_version': '1.0'
        }
    
    def add_trajectory_point(self, point: TrajectoryPoint) -> None:
        """Add a trajectory point to the flight path"""
        self.trajectory_points.append(point)
    
    def create_trajectory_point(self, 
                              timestamp: float,
                              position: Tuple[float, float, float],
                              velocity: Tuple[float, float, float],
                              warp_factor: float,
                              energy_density: float,
                              **kwargs) -> TrajectoryPoint:
        """Create and add a trajectory point"""
        point = TrajectoryPoint(
            timestamp=timestamp,
            position=position,
            velocity=velocity,
            warp_factor=warp_factor,
            energy_density=energy_density,
            **kwargs
        )
        self.add_trajectory_point(point)
        return point
    
    def validate_trajectory_physics(self) -> Dict[str, bool]:
        """Validate trajectory against physics constraints"""
        validation_results = {
            'energy_conservation': True,
            'causality_preservation': True,
            'velocity_constraints': True,
            'warp_field_limits': True,
            'trajectory_continuity': True
        }
        
        if len(self.trajectory_points) < 2:
            return validation_results
        
        # Check energy conservation
        energy_variations = []
        for i in range(1, len(self.trajectory_points)):
            prev_point = self.trajectory_points[i-1]
            curr_point = self.trajectory_points[i]
            
            # Energy conservation check (simplified)
            energy_change = abs(curr_point.energy_density - prev_point.energy_density)
            relative_change = energy_change / prev_point.energy_density
            if relative_change > 0.1:  # 10% tolerance
                validation_results['energy_conservation'] = False
            
            # Causality check - no backward time travel
            if curr_point.timestamp <= prev_point.timestamp:
                validation_results['causality_preservation'] = False
            
            # Velocity constraint check
            velocity_magnitude = math.sqrt(sum(v**2 for v in curr_point.velocity))
            if velocity_magnitude > C_LIGHT * curr_point.warp_factor:
                validation_results['velocity_constraints'] = False
            
            # Warp field limits
            if curr_point.warp_factor > 1000:  # Arbitrary safety limit
                validation_results['warp_field_limits'] = False
        
        return validation_results
    
    def export_ndjson(self, filename: str) -> None:
        """Export flight path to NDJSON file"""
        with open(filename, 'w') as f:
            # Write metadata header
            f.write(json.dumps({'metadata': self.mission_metadata}) + '\n')
            
            # Write trajectory points
            for point in self.trajectory_points:
                f.write(point.to_ndjson_line() + '\n')
    
    def import_ndjson(self, filename: str) -> None:
        """Import flight path from NDJSON file"""
        self.trajectory_points = []
        
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    if line_num == 0:
                        # First line might be metadata
                        data = json.loads(line)
                        if 'metadata' in data:
                            self.mission_metadata.update(data['metadata'])
                            continue
                    
                    # Regular trajectory point
                    point = TrajectoryPoint.from_ndjson_line(line)
                    self.trajectory_points.append(point)
                    
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON at line {line_num + 1}")
                except Exception as e:
                    print(f"Warning: Error processing line {line_num + 1}: {e}")
    
    def get_trajectory_statistics(self) -> Dict[str, Union[float, int]]:
        """Calculate trajectory statistics"""
        if not self.trajectory_points:
            return {}
        
        total_distance = 0.0
        max_velocity = 0.0
        max_warp_factor = 0.0
        total_energy = 0.0
        
        for i, point in enumerate(self.trajectory_points):
            velocity_magnitude = math.sqrt(sum(v**2 for v in point.velocity))
            max_velocity = max(max_velocity, velocity_magnitude)
            max_warp_factor = max(max_warp_factor, point.warp_factor)
            total_energy += point.energy_density
            
            if i > 0:
                prev_point = self.trajectory_points[i-1]
                dx = point.position[0] - prev_point.position[0]
                dy = point.position[1] - prev_point.position[1] 
                dz = point.position[2] - prev_point.position[2]
                distance = math.sqrt(dx**2 + dy**2 + dz**2)
                total_distance += distance
        
        duration = (self.trajectory_points[-1].timestamp - 
                   self.trajectory_points[0].timestamp)
        
        return {
            'total_points': len(self.trajectory_points),
            'total_distance_m': total_distance,
            'total_distance_ly': total_distance / (9.461e15),  # Convert to light years
            'mission_duration_s': duration,
            'mission_duration_days': duration / 86400,
            'max_velocity_ms': max_velocity,
            'max_velocity_c': max_velocity / C_LIGHT,
            'max_warp_factor': max_warp_factor,
            'average_energy_density': total_energy / len(self.trajectory_points),
            'total_energy_j': total_energy * total_distance  # Simplified
        }
    
    def generate_streaming_iterator(self) -> Iterator[str]:
        """Generate streaming NDJSON iterator for real-time updates"""
        # Yield metadata first
        yield json.dumps({'metadata': self.mission_metadata})
        
        # Yield trajectory points
        for point in self.trajectory_points:
            yield point.to_ndjson_line()

def create_sample_earth_proxima_trajectory() -> FlightPathFormat:
    """Create a sample Earth to Proxima Centauri trajectory"""
    flight_path = FlightPathFormat()
    
    # Mission metadata
    flight_path.mission_metadata.update({
        'mission_id': 'EARTH_PROXIMA_001',
        'vessel_id': 'ENTERPRISE_NX_01',
        'origin': 'Earth, Sol System',
        'destination': 'Proxima Centauri b',
        'total_energy_budget': 1.5e20,  # Joules
        'mission_duration': 3.154e7     # 1 year in seconds
    })
    
    # Proxima Centauri distance: ~4.24 light years
    proxima_distance = 4.24 * 9.461e15  # meters
    
    # Create trajectory points
    # Phase 1: Acceleration (first 30 days)
    for day in range(0, 30):
        timestamp = day * 86400.0
        
        # Gradual acceleration and warp factor increase
        progress = day / 30.0
        warp_factor = 1.0 + progress * 52.0  # Up to warp 53
        velocity_magnitude = progress * 0.9 * C_LIGHT * warp_factor
        
        # Position along trajectory
        position_progress = 0.5 * progress**2  # Accelerating
        distance = position_progress * proxima_distance * 0.1  # 10% during acceleration
        
        flight_path.create_trajectory_point(
            timestamp=timestamp,
            position=(distance, 0.0, 0.0),
            velocity=(velocity_magnitude, 0.0, 0.0),
            warp_factor=warp_factor,
            energy_density=1.23e15 * warp_factor**2,
            metric_signature='(-,+,+,+)',
            stability_factor=0.95
        )
    
    # Phase 2: Cruise (middle 305 days)
    cruise_start_day = 30
    cruise_duration = 305
    for day in range(cruise_duration):
        timestamp = (cruise_start_day + day) * 86400.0
        
        # Constant high warp
        warp_factor = 53.0
        velocity_magnitude = 0.9 * C_LIGHT * warp_factor
        
        # Linear progression during cruise
        cruise_progress = day / cruise_duration
        total_progress = 0.1 + cruise_progress * 0.8  # 10% to 90%
        distance = total_progress * proxima_distance
        
        flight_path.create_trajectory_point(
            timestamp=timestamp,
            position=(distance, 0.0, 0.0),
            velocity=(velocity_magnitude, 0.0, 0.0),
            warp_factor=warp_factor,
            energy_density=1.23e15 * warp_factor**2,
            metric_signature='(-,+,+,+)',
            stability_factor=0.98
        )
    
    # Phase 3: Deceleration (final 30 days)
    decel_start_day = 335
    for day in range(30):
        timestamp = (decel_start_day + day) * 86400.0
        
        # Gradual deceleration
        progress = 1.0 - (day / 30.0)
        warp_factor = 1.0 + progress * 52.0
        velocity_magnitude = progress * 0.9 * C_LIGHT * warp_factor
        
        # Final approach
        decel_progress = day / 30.0
        total_progress = 0.9 + decel_progress * 0.1  # 90% to 100%
        distance = total_progress * proxima_distance
        
        flight_path.create_trajectory_point(
            timestamp=timestamp,
            position=(distance, 0.0, 0.0),
            velocity=(velocity_magnitude, 0.0, 0.0),
            warp_factor=warp_factor,
            energy_density=1.23e15 * warp_factor**2,
            metric_signature='(-,+,+,+)',
            stability_factor=0.95
        )
    
    return flight_path

def main():
    """Demonstration of flight path format functionality"""
    print("🚀 LQG FTL Flight Path Format Demo")
    print("="*50)
    
    # Create sample trajectory
    flight_path = create_sample_earth_proxima_trajectory()
    
    # Display statistics
    stats = flight_path.get_trajectory_statistics()
    print(f"\n📊 Trajectory Statistics:")
    print(f"Total Points: {stats['total_points']}")
    print(f"Total Distance: {stats['total_distance_ly']:.2f} light years")
    print(f"Mission Duration: {stats['mission_duration_days']:.1f} days")
    print(f"Max Velocity: {stats['max_velocity_c']:.1f}c")
    print(f"Max Warp Factor: {stats['max_warp_factor']:.1f}")
    
    # Validate physics
    validation = flight_path.validate_trajectory_physics()
    print(f"\n🔬 Physics Validation:")
    for constraint, valid in validation.items():
        status = "✅ PASS" if valid else "❌ FAIL"
        print(f"{constraint}: {status}")
    
    # Export to NDJSON
    output_file = "earth_proxima_trajectory.ndjson"
    flight_path.export_ndjson(output_file)
    print(f"\n💾 Trajectory exported to: {output_file}")
    
    # Demonstrate streaming
    print(f"\n🌊 Streaming NDJSON Sample (first 3 lines):")
    for i, line in enumerate(flight_path.generate_streaming_iterator()):
        if i >= 3:
            break
        print(f"Line {i+1}: {line[:100]}..." if len(line) > 100 else f"Line {i+1}: {line}")

if __name__ == "__main__":
    main()

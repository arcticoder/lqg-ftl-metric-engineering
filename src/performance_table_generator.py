#!/usr/bin/env python3
"""
Performance Table Generator for LQG Drive Coordinate Velocity Analysis

Generates comprehensive CSV tables with all performance parameters including
coordinate velocity, energy requirements, tidal forces, smear time optimization,
and operational recommendations.

Repository: lqg-ftl-metric-engineering → performance integration module
Technology: Integrated analysis with energy-velocity-tidal-force optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from datetime import datetime
import json
from dataclasses import dataclass

# Import our analysis modules
from .coordinate_velocity_energy_mapping import CoordinateVelocityMapper
from .energy_scaling_analyzer import EnergyScalingAnalyzer
from .proportionality_validator import ProportionalityValidator
from .smear_time_calculator import SmearTimeCalculator
from .tidal_force_calculator import TidalForceCalculator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PerformanceParameters:
    """Comprehensive performance parameters for LQG Drive operations."""
    coordinate_velocity: float          # c
    positive_energy: float              # Joules
    energy_scaling_factor: float        # ratio
    tidal_force_boundary: float         # g
    optimal_smear_time: float           # hours
    optimal_acceleration_rate: float    # c/min
    safety_margin: float                # factor
    comfort_rating: str                 # category
    operational_recommendation: str     # guidance
    mission_profile_suitability: Dict  # mission type ratings

class PerformanceTableGenerator:
    """
    Comprehensive performance table generator for LQG Drive systems.
    
    Features:
    - Integrated analysis across all performance modules
    - Complete CSV table generation with all parameters
    - Mission profile optimization recommendations
    - Safety and comfort assessment integration
    - Operational guidance generation
    """
    
    def __init__(self, vessel_diameter: float = 200.0, vessel_height: float = 24.0):
        """
        Initialize performance table generator.
        
        Args:
            vessel_diameter: Vessel warp shape diameter in meters
            vessel_height: Vessel height in meters
        """
        self.vessel_diameter = vessel_diameter
        self.vessel_height = vessel_height
        
        # Initialize all analysis modules
        self.velocity_mapper = CoordinateVelocityMapper(vessel_diameter, vessel_height)
        self.scaling_analyzer = EnergyScalingAnalyzer(proportionality_limit=4.0)
        self.proportionality_validator = ProportionalityValidator()
        self.smear_calculator = SmearTimeCalculator(vessel_diameter, vessel_height)
        self.tidal_calculator = TidalForceCalculator(vessel_diameter, vessel_height)
        
        # Mission profile definitions
        self.mission_profiles = {
            'cargo_transport': {
                'max_tidal_force': 0.05,    # 0.05g for cargo protection
                'max_energy_scaling': 3.0,  # 3x energy scaling limit
                'preferred_smear_time': 2.0, # 2 hour gradual acceleration
                'priority': 'efficiency'
            },
            'passenger_cruise': {
                'max_tidal_force': 0.02,    # 0.02g for passenger comfort
                'max_energy_scaling': 2.5,  # 2.5x energy scaling limit
                'preferred_smear_time': 4.0, # 4 hour comfortable acceleration
                'priority': 'comfort'
            },
            'scientific_survey': {
                'max_tidal_force': 0.01,    # 0.01g for sensitive instruments
                'max_energy_scaling': 2.0,  # 2x energy scaling limit
                'preferred_smear_time': 6.0, # 6 hour careful acceleration
                'priority': 'precision'
            },
            'emergency_response': {
                'max_tidal_force': 0.2,     # 0.2g acceptable for emergencies
                'max_energy_scaling': 6.0,  # 6x energy scaling acceptable
                'preferred_smear_time': 0.5, # 30 min rapid acceleration
                'priority': 'speed'
            },
            'military_operations': {
                'max_tidal_force': 0.5,     # 0.5g for trained personnel
                'max_energy_scaling': 8.0,  # 8x energy scaling limit
                'preferred_smear_time': 0.25, # 15 min tactical acceleration
                'priority': 'performance'
            }
        }
        
        logger.info(f"Initialized PerformanceTableGenerator for {vessel_diameter}m × {vessel_height}m vessel")
        logger.info(f"Mission profiles configured: {list(self.mission_profiles.keys())}")
        
    def generate_base_performance_data(self, velocity_range: List[float]) -> pd.DataFrame:
        """
        Generate base performance data across all analysis modules.
        
        Args:
            velocity_range: List of coordinate velocities to analyze
            
        Returns:
            DataFrame with integrated performance data
        """
        logger.info(f"Generating base performance data for {len(velocity_range)} velocities")
        
        # 1. Generate velocity-energy mapping
        logger.info("Step 1/5: Calculating velocity-energy mapping...")
        velocity_energy_df = self.velocity_mapper.map_velocity_to_energy(velocity_range)
        
        # 2. Generate tidal force analysis
        logger.info("Step 2/5: Calculating tidal forces...")
        tidal_df = self.tidal_calculator.generate_tidal_force_profile(velocity_range)
        
        # 3. Generate smear time analysis for each velocity
        logger.info("Step 3/5: Calculating optimal smear times...")
        smear_results = []
        
        for velocity in velocity_range:
            try:
                # Calculate optimal smear parameters for this velocity
                velocity_scenarios = [(1, velocity), (velocity/2, velocity), (velocity, velocity*1.5)]
                smear_durations = [0.5, 1.0, 2.0, 4.0]
                acceleration_rates = [0.1, 0.2, 0.5, 1.0]
                
                smear_df = self.smear_calculator.generate_smear_time_table(
                    smear_durations, acceleration_rates, velocity_scenarios[:1]  # Use first scenario
                )
                
                if not smear_df.empty:
                    # Find optimal smear parameters
                    optimal_smear = smear_df.loc[smear_df['safety_margin'].idxmax()]
                    
                    smear_results.append({
                        'coordinate_velocity_c': velocity,
                        'optimal_smear_time_hours': optimal_smear['smear_time_hours'],
                        'optimal_acceleration_rate_c_per_min': optimal_smear['acceleration_rate_c_per_min'],
                        'smear_safety_margin': optimal_smear['safety_margin'],
                        'smear_comfort_rating': optimal_smear['comfort_rating']
                    })
                    
            except Exception as e:
                logger.warning(f"Error calculating smear time for {velocity}c: {e}")
                continue
                
        smear_optimal_df = pd.DataFrame(smear_results)
        
        # 4. Merge all data
        logger.info("Step 4/5: Merging analysis results...")
        
        # Start with velocity-energy data
        integrated_df = velocity_energy_df.copy()
        
        # Merge tidal force data (take boundary values)
        tidal_boundary = tidal_df[tidal_df['radial_position'] >= 1.0].groupby('coordinate_velocity_c').agg({
            'tidal_force_g': 'mean',
            'safety_assessment': 'first',
            'comfort_level': 'first',
            'lqg_correction_factor': 'mean'
        }).reset_index()
        
        integrated_df = integrated_df.merge(tidal_boundary, on='coordinate_velocity_c', how='left')
        
        # Merge smear time data
        integrated_df = integrated_df.merge(smear_optimal_df, on='coordinate_velocity_c', how='left')
        
        # 5. Calculate additional performance metrics
        logger.info("Step 5/5: Calculating additional performance metrics...")
        
        integrated_df['energy_per_distance'] = integrated_df['positive_energy_joules'] / integrated_df['coordinate_velocity_c']
        integrated_df['total_safety_score'] = (
            (1.0 - integrated_df['tidal_force_g'].fillna(0)) * 0.5 +
            integrated_df['smear_safety_margin'].fillna(0.5) * 0.3 +
            (1.0 / integrated_df['scaling_factor']).fillna(0.5) * 0.2
        )
        
        logger.info(f"Generated integrated performance data: {len(integrated_df)} velocity points")
        
        return integrated_df
        
    def assess_mission_profile_suitability(self, performance_row: pd.Series) -> Dict[str, Dict]:
        """
        Assess suitability for different mission profiles.
        
        Args:
            performance_row: Row from performance DataFrame
            
        Returns:
            Dictionary with mission profile suitability assessments
        """
        suitability = {}
        
        for profile_name, requirements in self.mission_profiles.items():
            # Check constraints
            tidal_ok = performance_row.get('tidal_force_g', 0) <= requirements['max_tidal_force']
            scaling_ok = performance_row.get('scaling_factor', 1) <= requirements['max_energy_scaling']
            
            # Calculate suitability score
            tidal_score = max(0, 1 - performance_row.get('tidal_force_g', 0) / requirements['max_tidal_force'])
            scaling_score = max(0, 1 - performance_row.get('scaling_factor', 1) / requirements['max_energy_scaling'])
            
            overall_score = (tidal_score + scaling_score) / 2
            
            # Determine suitability rating
            if tidal_ok and scaling_ok:
                if overall_score >= 0.8:
                    rating = "EXCELLENT"
                elif overall_score >= 0.6:
                    rating = "GOOD"
                else:
                    rating = "ACCEPTABLE"
            else:
                rating = "NOT_SUITABLE"
                
            suitability[profile_name] = {
                'rating': rating,
                'score': overall_score,
                'tidal_constraint_met': tidal_ok,
                'scaling_constraint_met': scaling_ok,
                'recommendation': self._generate_mission_recommendation(rating, requirements['priority'])
            }
            
        return suitability
        
    def _generate_mission_recommendation(self, rating: str, priority: str) -> str:
        """Generate mission-specific recommendations."""
        if rating == "EXCELLENT":
            return f"Optimal for {priority}-focused operations"
        elif rating == "GOOD":
            return f"Well-suited for {priority} missions"
        elif rating == "ACCEPTABLE":
            return f"Usable for {priority} missions with careful planning"
        else:
            return f"Not recommended for {priority} operations"
            
    def generate_operational_guidance(self, performance_row: pd.Series) -> str:
        """
        Generate operational guidance based on performance characteristics.
        
        Args:
            performance_row: Row from performance DataFrame
            
        Returns:
            Operational guidance string
        """
        velocity = performance_row.get('coordinate_velocity_c', 0)
        tidal_force = performance_row.get('tidal_force_g', 0)
        scaling_factor = performance_row.get('scaling_factor', 1)
        safety_score = performance_row.get('total_safety_score', 0.5)
        
        # Generate guidance based on characteristics
        guidance_parts = []
        
        # Velocity-based guidance
        if velocity <= 10:
            guidance_parts.append("LOW VELOCITY: Suitable for all operations")
        elif velocity <= 50:
            guidance_parts.append("MEDIUM VELOCITY: Monitor energy consumption")
        elif velocity <= 100:
            guidance_parts.append("HIGH VELOCITY: Specialized operations only")
        else:
            guidance_parts.append("EXTREME VELOCITY: Expert crews and emergency protocols required")
            
        # Tidal force guidance
        if tidal_force <= 0.01:
            guidance_parts.append("Imperceptible tidal effects")
        elif tidal_force <= 0.05:
            guidance_parts.append("Minimal passenger discomfort")
        elif tidal_force <= 0.1:
            guidance_parts.append("Passenger briefing recommended")
        else:
            guidance_parts.append("Crew-only operations")
            
        # Energy scaling guidance
        if scaling_factor <= 2.0:
            guidance_parts.append("Excellent energy efficiency")
        elif scaling_factor <= 4.0:
            guidance_parts.append("Acceptable energy scaling")
        else:
            guidance_parts.append("High energy cost - limit operational duration")
            
        # Overall recommendation
        if safety_score >= 0.8:
            guidance_parts.append("RECOMMENDED for routine operations")
        elif safety_score >= 0.6:
            guidance_parts.append("ACCEPTABLE with standard precautions")
        else:
            guidance_parts.append("CAUTION: Enhanced safety protocols required")
            
        return " | ".join(guidance_parts)
        
    def generate_comprehensive_performance_table(self, velocity_range: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Generate comprehensive performance table with all parameters.
        
        Args:
            velocity_range: List of velocities to analyze (default: 1c-100c)
            
        Returns:
            Complete performance table DataFrame
        """
        if velocity_range is None:
            velocity_range = list(np.arange(1, 101, 1))  # 1c to 100c in 1c steps
            
        logger.info(f"Generating comprehensive performance table for {len(velocity_range)} velocities")
        
        # Generate base performance data
        base_df = self.generate_base_performance_data(velocity_range)
        
        # Add mission profile suitability assessments
        logger.info("Assessing mission profile suitability...")
        
        mission_assessments = []
        for idx, row in base_df.iterrows():
            suitability = self.assess_mission_profile_suitability(row)
            
            # Flatten mission profile data for CSV
            mission_data = {'coordinate_velocity_c': row['coordinate_velocity_c']}
            for profile, assessment in suitability.items():
                mission_data[f'mission_{profile}_rating'] = assessment['rating']
                mission_data[f'mission_{profile}_score'] = assessment['score']
                
            mission_assessments.append(mission_data)
            
        mission_df = pd.DataFrame(mission_assessments)
        
        # Merge mission assessments
        comprehensive_df = base_df.merge(mission_df, on='coordinate_velocity_c', how='left')
        
        # Add operational guidance
        logger.info("Generating operational guidance...")
        comprehensive_df['operational_guidance'] = comprehensive_df.apply(
            self.generate_operational_guidance, axis=1
        )
        
        # Calculate Earth-Proxima travel times for practical reference
        proxima_distance_ly = 4.24  # light-years
        comprehensive_df['earth_proxima_travel_time_days'] = (
            proxima_distance_ly * 365.25 / comprehensive_df['coordinate_velocity_c']
        )
        
        # Final column organization
        column_order = [
            'coordinate_velocity_c',
            'positive_energy_joules',
            'energy_per_distance',
            'scaling_factor',
            'tidal_force_g',
            'safety_assessment',
            'comfort_level',
            'optimal_smear_time_hours',
            'optimal_acceleration_rate_c_per_min',
            'smear_safety_margin',
            'total_safety_score',
            'earth_proxima_travel_time_days',
            'operational_guidance'
        ]
        
        # Add mission profile columns
        mission_columns = [col for col in comprehensive_df.columns if col.startswith('mission_')]
        column_order.extend(mission_columns)
        
        # Ensure all columns exist
        final_columns = [col for col in column_order if col in comprehensive_df.columns]
        comprehensive_df = comprehensive_df[final_columns]
        
        logger.info(f"Generated comprehensive performance table: {len(comprehensive_df)} rows × {len(final_columns)} columns")
        
        return comprehensive_df
        
    def export_performance_tables(self, df: pd.DataFrame, base_filename: str = "lqg_drive_performance") -> None:
        """
        Export performance tables in multiple formats.
        
        Args:
            df: Performance table DataFrame
            base_filename: Base filename for exports
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export main CSV table
        csv_filename = f"{base_filename}_{timestamp}.csv"
        csv_path = Path(csv_filename)
        
        df.to_csv(csv_path, index=False)
        
        # Add comprehensive header
        with open(csv_path, 'r') as f:
            content = f.read()
            
        header = f"""# LQG Drive Coordinate Velocity Performance Analysis
# Generated: {datetime.now().isoformat()}
# Vessel Configuration: {self.vessel_diameter}m diameter × {self.vessel_height}m height
# Technology: LQG polymer corrections with Bobrick-Martire geometry optimization
# Analysis Scope: Coordinate velocities 1c-{df['coordinate_velocity_c'].max():.0f}c
# 
# PERFORMANCE PARAMETERS:
# - coordinate_velocity_c: Coordinate velocity in units of c
# - positive_energy_joules: Energy requirement maintaining T_μν ≥ 0
# - energy_per_distance: Energy efficiency metric (J per c)
# - scaling_factor: Energy scaling ratio vs previous velocity point
# - tidal_force_g: Average tidal force at warp shape boundary
# - safety_assessment: General safety level classification
# - comfort_level: Passenger comfort assessment
# - optimal_smear_time_hours: Recommended spacetime smearing duration
# - optimal_acceleration_rate_c_per_min: Recommended acceleration rate
# - smear_safety_margin: Safety margin for smearing operations
# - total_safety_score: Composite safety score (0-1 scale)
# - earth_proxima_travel_time_days: Travel time to Proxima Centauri
# - operational_guidance: Comprehensive operational recommendations
# - mission_*_rating: Suitability rating for specific mission profiles
# - mission_*_score: Quantitative suitability score (0-1 scale)
#
# MISSION PROFILES:
# - cargo_transport: Commercial cargo operations
# - passenger_cruise: Civilian passenger transport
# - scientific_survey: Scientific research missions
# - emergency_response: Emergency and rescue operations
# - military_operations: Military and defense applications
#
# SAFETY THRESHOLDS:
# - Tidal forces: <0.1g safe, <0.05g comfortable, <0.01g imperceptible
# - Energy scaling: ≤4x per velocity doubling for proportionality
# - Smear times: Optimized for passenger comfort and energy efficiency
#
"""
        with open(csv_path, 'w') as f:
            f.write(header + content)
            
        logger.info(f"Exported comprehensive performance table to {csv_path}")
        
        # Export summary statistics
        summary_filename = f"{base_filename}_summary_{timestamp}.json"
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'vessel_configuration': {
                'diameter_m': self.vessel_diameter,
                'height_m': self.vessel_height
            },
            'velocity_range': {
                'min_c': float(df['coordinate_velocity_c'].min()),
                'max_c': float(df['coordinate_velocity_c'].max()),
                'total_points': len(df)
            },
            'energy_analysis': {
                'min_energy_j': float(df['positive_energy_joules'].min()),
                'max_energy_j': float(df['positive_energy_joules'].max()),
                'avg_scaling_factor': float(df['scaling_factor'].mean())
            },
            'tidal_force_analysis': {
                'min_tidal_g': float(df['tidal_force_g'].min()),
                'max_tidal_g': float(df['tidal_force_g'].max()),
                'safe_velocities_count': int((df['tidal_force_g'] <= 0.1).sum())
            },
            'mission_profile_suitability': {
                profile: {
                    'excellent_count': int((df[f'mission_{profile}_rating'] == 'EXCELLENT').sum()),
                    'good_count': int((df[f'mission_{profile}_rating'] == 'GOOD').sum()),
                    'acceptable_count': int((df[f'mission_{profile}_rating'] == 'ACCEPTABLE').sum())
                }
                for profile in self.mission_profiles.keys()
            }
        }
        
        with open(summary_filename, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Exported performance summary to {summary_filename}")

def main():
    """Main execution function for performance table generation."""
    logger.info("Starting LQG Drive Performance Table Generation")
    
    # Initialize generator
    generator = PerformanceTableGenerator(vessel_diameter=200.0, vessel_height=24.0)
    
    # Generate performance table (limited range for testing)
    test_velocities = list(range(1, 26))  # 1c to 25c for testing
    
    logger.info(f"Generating performance table for velocities: {test_velocities[0]}c - {test_velocities[-1]}c")
    
    performance_df = generator.generate_comprehensive_performance_table(test_velocities)
    
    # Display summary
    logger.info("=== PERFORMANCE TABLE GENERATION RESULTS ===")
    logger.info(f"Total velocity points analyzed: {len(performance_df)}")
    logger.info(f"Energy range: {performance_df['positive_energy_joules'].min():.2e} - {performance_df['positive_energy_joules'].max():.2e} J")
    logger.info(f"Tidal force range: {performance_df['tidal_force_g'].min():.6f} - {performance_df['tidal_force_g'].max():.6f} g")
    
    # Mission profile summary
    for profile in generator.mission_profiles.keys():
        excellent_count = (performance_df[f'mission_{profile}_rating'] == 'EXCELLENT').sum()
        logger.info(f"{profile}: {excellent_count} excellent velocity points")
        
    # Export results
    generator.export_performance_tables(performance_df, "test_lqg_drive_performance")
    
    logger.info("Performance table generation complete!")
    
    return performance_df

if __name__ == "__main__":
    performance_df = main()

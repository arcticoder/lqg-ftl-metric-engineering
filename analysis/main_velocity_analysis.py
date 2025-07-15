#!/usr/bin/env python3
"""
LQG Drive Coordinate Velocity and Energy Requirements Analysis - Main Integration Script

Complete implementation of the LQG Drive Coordinate Velocity and Energy Requirements Analysis
as specified in future-directions.md. Integrates all analysis modules to generate comprehensive
performance tables and optimization recommendations.

Repository: lqg-ftl-metric-engineering → complete velocity analysis system
Technology: LQG polymer corrections with Bobrick-Martire geometry optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from pathlib import Path
import logging
from datetime import datetime
import argparse
import sys

# Import all analysis modules
from src.coordinate_velocity_energy_mapping import CoordinateVelocityMapper
from src.energy_scaling_analyzer import EnergyScalingAnalyzer
from src.proportionality_validator import ProportionalityValidator
from src.smear_time_calculator import SmearTimeCalculator
from src.tidal_force_calculator import TidalForceCalculator
from src.performance_table_generator import PerformanceTableGenerator
from src.csv_export_system import CSVExportSystem
from src.optimization_recommender import OptimizationRecommender, OptimizationObjective

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lqg_drive_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class LQGDriveAnalysisController:
    """
    Main controller for comprehensive LQG Drive coordinate velocity analysis.
    
    Implements complete analysis pipeline as specified in future-directions.md:
    - Coordinate velocity energy mapping (1c-9999c)  
    - Energy scaling analysis and proportionality validation
    - Smear time optimization framework
    - Tidal force calculation and safety assessment
    - Performance table generation with mission profiles
    - Optimization recommendations for operational guidance
    """
    
    def __init__(self, vessel_diameter: float = 200.0, vessel_height: float = 24.0):
        """
        Initialize LQG Drive analysis controller.
        
        Args:
            vessel_diameter: Warp shape diameter in meters (default: 200m)
            vessel_height: Vessel height in meters (default: 24m)
        """
        self.vessel_diameter = vessel_diameter
        self.vessel_height = vessel_height
        
        # Analysis configuration
        self.analysis_config = {
            'max_coordinate_velocity': 9999.0,    # Sky is the limit
            'velocity_increment': 0.1,            # 0.1c increments for comprehensive mapping
            'proportionality_limit': 4.0,         # ≤4x energy increase per velocity doubling
            'max_scaling_per_step': 8.0,          # ≤8x energy increase per 1c step
            'safety_tidal_threshold': 0.1,        # 0.1g safety threshold
            'comfort_tidal_threshold': 0.05       # 0.05g comfort threshold
        }
        
        # Initialize all analysis modules
        logger.info("Initializing LQG Drive Analysis Controller...")
        
        self.velocity_mapper = CoordinateVelocityMapper(vessel_diameter, vessel_height)
        self.scaling_analyzer = EnergyScalingAnalyzer(proportionality_limit=self.analysis_config['proportionality_limit'])
        self.proportionality_validator = ProportionalityValidator()
        self.smear_calculator = SmearTimeCalculator(vessel_diameter, vessel_height)
        self.tidal_calculator = TidalForceCalculator(vessel_diameter, vessel_height)
        self.performance_generator = PerformanceTableGenerator(vessel_diameter, vessel_height)
        self.csv_exporter = CSVExportSystem(vessel_diameter, vessel_height)
        self.optimizer = OptimizationRecommender()
        
        logger.info(f"Initialized for {vessel_diameter}m × {vessel_height}m vessel configuration")
        logger.info("All analysis modules loaded successfully")
        
    def run_velocity_energy_mapping(self, max_velocity: Optional[float] = None) -> pd.DataFrame:
        """
        Phase 1: Generate coordinate velocity to energy mapping.
        
        Args:
            max_velocity: Maximum velocity to analyze (default: from config)
            
        Returns:
            Velocity-energy mapping DataFrame
        """
        logger.info("=== PHASE 1: COORDINATE VELOCITY ENERGY MAPPING ===")
        
        if max_velocity is None:
            max_velocity = self.analysis_config['max_coordinate_velocity']
            
        # Start with conservative range, then expand if feasible
        initial_max = min(max_velocity, 100.0)  # Start with 100c
        increment = self.analysis_config['velocity_increment']
        
        logger.info(f"Starting velocity mapping: 1c to {initial_max}c (increment: {increment}c)")
        
        # Generate initial velocity range
        velocities = []
        velocity = 1.0
        
        while velocity <= initial_max:
            velocities.append(velocity)
            
            # Adaptive increment - smaller steps at higher velocities
            if velocity <= 10:
                step = increment
            elif velocity <= 50:
                step = increment * 2  # 0.2c steps
            else:
                step = increment * 5  # 0.5c steps
                
            velocity += step
            
        logger.info(f"Generated {len(velocities)} velocity points")
        
        # Generate velocity-energy mapping
        velocity_energy_df = self.velocity_mapper.map_velocity_to_energy(velocities)
        
        # Validate results and potentially extend range
        if not velocity_energy_df.empty:
            max_achieved = velocity_energy_df['coordinate_velocity_c'].max()
            logger.info(f"Successfully mapped velocities up to {max_achieved:.1f}c")
            
            # Check if we can extend to higher velocities
            if max_achieved >= initial_max * 0.9 and max_velocity > initial_max:
                logger.info("Attempting to extend velocity range...")
                
                # Try extending to higher velocities
                extended_velocities = list(np.arange(max_achieved + 1, min(max_velocity, 500), 5))[:20]  # Limit extension
                
                if extended_velocities:
                    try:
                        extended_df = self.velocity_mapper.map_velocity_to_energy(extended_velocities)
                        if not extended_df.empty:
                            velocity_energy_df = pd.concat([velocity_energy_df, extended_df], ignore_index=True)
                            logger.info(f"Extended analysis to {velocity_energy_df['coordinate_velocity_c'].max():.1f}c")
                    except Exception as e:
                        logger.warning(f"Could not extend velocity range: {e}")
                        
        return velocity_energy_df
        
    def run_energy_scaling_analysis(self, velocity_energy_df: pd.DataFrame) -> Dict:
        """
        Phase 2: Analyze energy scaling patterns and validate proportionality.
        
        Args:
            velocity_energy_df: Velocity-energy mapping data
            
        Returns:
            Energy scaling analysis results
        """
        logger.info("=== PHASE 2: ENERGY SCALING ANALYSIS ===")
        
        # Generate comprehensive scaling analysis
        scaling_report = self.scaling_analyzer.generate_scaling_report(velocity_energy_df)
        
        # Validate proportionality constraints
        proportionality_results = self.proportionality_validator.comprehensive_validation(velocity_energy_df)
        
        # Combine results
        analysis_results = {
            'scaling_analysis': scaling_report,
            'proportionality_validation': proportionality_results,
            'compliance_summary': {
                'proportionality_compliance': proportionality_results['compliance_percentage'],
                'scaling_regions': len(scaling_report.get('scaling_regions', [])),
                'optimal_velocity_ranges': len(scaling_report.get('optimal_velocities', {})),
                'critical_violations': len(proportionality_results.get('critical_violations', []))
            }
        }
        
        logger.info(f"Proportionality compliance: {proportionality_results['compliance_percentage']:.1f}%")
        logger.info(f"Identified {len(scaling_report.get('scaling_regions', []))} scaling regions")
        
        return analysis_results
        
    def run_smear_time_optimization(self, velocity_energy_df: pd.DataFrame) -> pd.DataFrame:
        """
        Phase 3: Generate smear time optimization analysis.
        
        Args:
            velocity_energy_df: Velocity-energy mapping data
            
        Returns:
            Smear time analysis DataFrame
        """
        logger.info("=== PHASE 3: SMEAR TIME OPTIMIZATION ===")
        
        # Define smear time analysis parameters
        smear_durations = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]  # hours
        acceleration_rates = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]  # c per minute
        
        # Create velocity scenarios based on mapped data
        velocity_scenarios = []
        max_velocity = velocity_energy_df['coordinate_velocity_c'].max()
        
        # Low speed scenarios
        velocity_scenarios.extend([(1, 5), (2, 10), (5, 15)])
        
        # Medium speed scenarios  
        if max_velocity >= 20:
            velocity_scenarios.extend([(10, 20), (15, 30), (20, 40)])
            
        # High speed scenarios
        if max_velocity >= 50:
            velocity_scenarios.extend([(25, 50), (40, 80)])
            
        # Very high speed scenarios
        if max_velocity >= 100:
            velocity_scenarios.extend([(50, 100), (75, min(max_velocity, 150))])
            
        logger.info(f"Analyzing {len(velocity_scenarios)} velocity scenarios with {len(smear_durations)} durations")
        
        # Generate smear time table
        smear_df = self.smear_calculator.generate_smear_time_table(
            smear_durations, acceleration_rates, velocity_scenarios
        )
        
        # Find optimal parameters
        optimal_params = self.smear_calculator.find_optimal_smear_parameters(smear_df)
        
        logger.info("Smear time optimization profiles identified:")
        for profile, params in optimal_params.items():
            if 'smear_time_hours' in params:
                logger.info(f"  {profile}: {params['smear_time_hours']}h, {params['acceleration_rate_c_per_min']}c/min")
                
        return smear_df
        
    def run_tidal_force_analysis(self, velocity_energy_df: pd.DataFrame) -> pd.DataFrame:
        """
        Phase 4: Calculate tidal forces at warp shape boundary.
        
        Args:
            velocity_energy_df: Velocity-energy mapping data
            
        Returns:
            Tidal force analysis DataFrame
        """
        logger.info("=== PHASE 4: TIDAL FORCE ANALYSIS ===")
        
        # Extract velocity range for tidal analysis
        velocities = velocity_energy_df['coordinate_velocity_c'].tolist()
        
        # Limit tidal analysis for performance (every 5th point for high velocities)
        if len(velocities) > 100:
            # Take every point up to 50c, then sample
            analysis_velocities = [v for v in velocities if v <= 50]
            high_velocity_sample = [v for v in velocities if v > 50][::5]  # Every 5th point
            analysis_velocities.extend(high_velocity_sample)
        else:
            analysis_velocities = velocities
            
        logger.info(f"Analyzing tidal forces for {len(analysis_velocities)} velocity points")
        
        # Generate tidal force profile
        tidal_df = self.tidal_calculator.generate_tidal_force_profile(analysis_velocities)
        
        # Find maximum safe velocity
        max_safe = self.tidal_calculator.find_maximum_safe_velocity(
            tidal_df, safety_threshold=self.analysis_config['safety_tidal_threshold']
        )
        
        if max_safe.get('max_safe_velocity_c'):
            logger.info(f"Maximum safe velocity: {max_safe['max_safe_velocity_c']:.1f}c")
            logger.info(f"Tidal force at maximum: {max_safe['tidal_force_at_max_g']:.6f}g")
        else:
            logger.warning("No velocities meet safety threshold - all operations require enhanced protocols")
            
        return tidal_df
        
    def run_performance_integration(self, velocity_energy_df: pd.DataFrame) -> pd.DataFrame:
        """
        Phase 5: Generate comprehensive performance tables.
        
        Args:
            velocity_energy_df: Velocity-energy mapping data
            
        Returns:
            Comprehensive performance table DataFrame
        """
        logger.info("=== PHASE 5: PERFORMANCE TABLE GENERATION ===")
        
        # Extract velocity range (limit for performance)
        velocities = velocity_energy_df['coordinate_velocity_c'].tolist()
        
        # For comprehensive tables, limit to reasonable range
        if len(velocities) > 200:
            # Take every point up to 100c, then every 5th point
            table_velocities = [v for v in velocities if v <= 100]
            high_velocity_sample = [v for v in velocities if v > 100][::5]
            table_velocities.extend(high_velocity_sample[:50])  # Limit to 50 additional points
        else:
            table_velocities = velocities
            
        logger.info(f"Generating performance tables for {len(table_velocities)} velocity points")
        
        # Generate comprehensive performance table
        performance_df = self.performance_generator.generate_comprehensive_performance_table(table_velocities)
        
        logger.info(f"Generated performance table: {len(performance_df)} rows × {len(performance_df.columns)} columns")
        
        return performance_df
        
    def run_optimization_analysis(self, performance_df: pd.DataFrame) -> Dict:
        """
        Phase 6: Generate optimization recommendations.
        
        Args:
            performance_df: Performance table DataFrame
            
        Returns:
            Optimization recommendations dictionary
        """
        logger.info("=== PHASE 6: OPTIMIZATION RECOMMENDATIONS ===")
        
        # Generate comprehensive optimization recommendations
        recommendations = self.optimizer.generate_comprehensive_recommendations(performance_df)
        
        logger.info("Generated optimization recommendations:")
        for objective, rec in recommendations.items():
            logger.info(f"  {objective}: {rec.recommended_velocity:.1f}c (confidence: {rec.confidence_score:.3f})")
            
        return recommendations
        
    def export_all_results(self, performance_df: pd.DataFrame, 
                          scaling_analysis: Dict, smear_df: pd.DataFrame, 
                          tidal_df: pd.DataFrame, recommendations: Dict) -> None:
        """
        Export all analysis results to files.
        
        Args:
            performance_df: Main performance table
            scaling_analysis: Energy scaling analysis results
            smear_df: Smear time analysis data
            tidal_df: Tidal force analysis data
            recommendations: Optimization recommendations
        """
        logger.info("=== EXPORTING ANALYSIS RESULTS ===")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export main performance tables in multiple formats
        export_results = self.csv_exporter.export_multiple_formats(
            performance_df, f"lqg_drive_coordinate_velocity_performance_{timestamp}"
        )
        
        # Export individual analysis components
        component_exports = [
            (smear_df, f"lqg_drive_smear_time_analysis_{timestamp}.csv"),
            (tidal_df, f"lqg_drive_tidal_force_analysis_{timestamp}.csv")
        ]
        
        for df, filename in component_exports:
            if not df.empty:
                df.to_csv(filename, index=False)
                logger.info(f"Exported {filename}")
                
        # Export optimization recommendations
        self.optimizer.export_recommendations(
            recommendations, f"lqg_drive_optimization_recommendations_{timestamp}.json"
        )
        
        # Export energy scaling analysis
        import json
        with open(f"lqg_drive_scaling_analysis_{timestamp}.json", 'w') as f:
            json.dump(scaling_analysis, f, indent=2, default=str)
            
        # Generate comprehensive README
        self.csv_exporter.generate_readme(
            export_results, f"README_LQG_Drive_Analysis_{timestamp}.md"
        )
        
        logger.info("All analysis results exported successfully")
        
    def run_complete_analysis(self, max_velocity: Optional[float] = None) -> Dict:
        """
        Run complete LQG Drive coordinate velocity and energy requirements analysis.
        
        Args:
            max_velocity: Maximum coordinate velocity to analyze
            
        Returns:
            Dictionary with all analysis results
        """
        logger.info("🚀 STARTING COMPLETE LQG DRIVE COORDINATE VELOCITY ANALYSIS 🚀")
        logger.info(f"Target: Comprehensive performance mapping for {self.vessel_diameter}m starship")
        logger.info(f"Technology: LQG polymer corrections with Bobrick-Martire geometry optimization")
        logger.info(f"Objective: Generate CSV tables with coordinate velocities 1c-{max_velocity or self.analysis_config['max_coordinate_velocity']}c")
        
        try:
            # Phase 1: Velocity-Energy Mapping
            velocity_energy_df = self.run_velocity_energy_mapping(max_velocity)
            
            if velocity_energy_df.empty:
                raise RuntimeError("Velocity-energy mapping failed - no valid data points generated")
                
            # Phase 2: Energy Scaling Analysis
            scaling_analysis = self.run_energy_scaling_analysis(velocity_energy_df)
            
            # Phase 3: Smear Time Optimization
            smear_df = self.run_smear_time_optimization(velocity_energy_df)
            
            # Phase 4: Tidal Force Analysis
            tidal_df = self.run_tidal_force_analysis(velocity_energy_df)
            
            # Phase 5: Performance Integration
            performance_df = self.run_performance_integration(velocity_energy_df)
            
            # Phase 6: Optimization Recommendations
            recommendations = self.run_optimization_analysis(performance_df)
            
            # Export all results
            self.export_all_results(performance_df, scaling_analysis, smear_df, tidal_df, recommendations)
            
            # Compile final results
            results = {
                'velocity_energy_mapping': velocity_energy_df,
                'performance_table': performance_df,
                'scaling_analysis': scaling_analysis,
                'smear_time_analysis': smear_df,
                'tidal_force_analysis': tidal_df,
                'optimization_recommendations': recommendations,
                'analysis_summary': {
                    'total_velocity_points': len(velocity_energy_df),
                    'max_velocity_achieved': velocity_energy_df['coordinate_velocity_c'].max(),
                    'energy_range': (velocity_energy_df['positive_energy_joules'].min(), 
                                   velocity_energy_df['positive_energy_joules'].max()),
                    'proportionality_compliance': scaling_analysis['proportionality_validation']['compliance_percentage'],
                    'safe_velocity_count': (tidal_df['tidal_force_g'] <= 0.1).sum() if not tidal_df.empty else 0
                }
            }
            
            logger.info("✅ COMPLETE LQG DRIVE ANALYSIS FINISHED SUCCESSFULLY ✅")
            logger.info(f"📊 Analyzed {results['analysis_summary']['total_velocity_points']} velocity points")
            logger.info(f"🚀 Maximum velocity: {results['analysis_summary']['max_velocity_achieved']:.1f}c")
            logger.info(f"⚡ Energy range: {results['analysis_summary']['energy_range'][0]:.2e} - {results['analysis_summary']['energy_range'][1]:.2e} J")
            logger.info(f"✅ Proportionality compliance: {results['analysis_summary']['proportionality_compliance']:.1f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ ANALYSIS FAILED: {e}")
            raise

def main():
    """Main execution function with command line interface."""
    parser = argparse.ArgumentParser(
        description="LQG Drive Coordinate Velocity and Energy Requirements Analysis",
        epilog="Complete implementation of future-directions.md LQG Drive analysis requirements"
    )
    
    parser.add_argument(
        '--max-velocity', 
        type=float, 
        default=100.0,
        help='Maximum coordinate velocity to analyze in units of c (default: 100.0)'
    )
    
    parser.add_argument(
        '--vessel-diameter',
        type=float,
        default=200.0,
        help='Vessel warp shape diameter in meters (default: 200.0)'
    )
    
    parser.add_argument(
        '--vessel-height',
        type=float,
        default=24.0,
        help='Vessel height in meters (default: 24.0)'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with limited velocity range (1c-25c)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Adjust max velocity for quick test
    if args.quick_test:
        max_velocity = 25.0
        logger.info("🧪 QUICK TEST MODE: Limited to 25c for rapid validation")
    else:
        max_velocity = args.max_velocity
        
    # Initialize and run analysis
    controller = LQGDriveAnalysisController(
        vessel_diameter=args.vessel_diameter,
        vessel_height=args.vessel_height
    )
    
    results = controller.run_complete_analysis(max_velocity=max_velocity)
    
    return results

if __name__ == "__main__":
    results = main()

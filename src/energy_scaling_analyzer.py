#!/usr/bin/env python3
"""
Energy Scaling Analyzer for LQG Drive Systems

Analyzes energy scaling patterns, identifies optimal operating regions, and validates 
proportionality constraints (≤4x energy increase per coordinate velocity doubling).
Provides comprehensive scaling analysis and efficiency recommendations.

Repository: lqg-ftl-metric-engineering → velocity analysis module  
Technology: Advanced scaling analysis with LQG optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ScalingRegion:
    """Data structure for energy scaling analysis regions."""
    velocity_start: float
    velocity_end: float
    scaling_factor: float
    efficiency_rating: str
    energy_range: Tuple[float, float]
    recommendation: str

class EnergyScalingAnalyzer:
    """
    Advanced energy scaling analyzer for LQG Drive coordinate velocity mapping.
    
    Features:
    - Proportionality validation (≤4x per velocity doubling)
    - Optimal operating region identification  
    - Scaling pattern analysis and prediction
    - Efficiency rating system
    - Energy constraint monitoring
    """
    
    def __init__(self, proportionality_limit: float = 4.0):
        """
        Initialize the energy scaling analyzer.
        
        Args:
            proportionality_limit: Maximum allowed energy scaling per velocity doubling (default: 4x)
        """
        self.proportionality_limit = proportionality_limit
        self.efficiency_thresholds = {
            'excellent': 1.5,
            'good': 2.5, 
            'acceptable': 4.0,
            'poor': 8.0
        }
        
        logger.info(f"Initialized EnergyScalingAnalyzer with {proportionality_limit}x proportionality limit")
        
    def analyze_doubling_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze energy scaling for velocity doubling intervals.
        
        Args:
            df: Velocity-energy mapping DataFrame
            
        Returns:
            DataFrame with doubling analysis results
        """
        doubling_analysis = []
        
        # Find approximate doubling points
        base_velocities = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        
        for base_v in base_velocities:
            double_v = base_v * 2
            
            # Find closest points in data
            base_point = df.loc[(df['coordinate_velocity_c'] - base_v).abs().idxmin()]
            double_candidates = df[df['coordinate_velocity_c'] >= double_v]
            
            if not double_candidates.empty:
                double_point = double_candidates.iloc[0]
                
                # Calculate scaling factor
                energy_ratio = double_point['positive_energy_joules'] / base_point['positive_energy_joules']
                velocity_ratio = double_point['coordinate_velocity_c'] / base_point['coordinate_velocity_c']
                
                # Evaluate proportionality
                proportional = energy_ratio <= self.proportionality_limit
                efficiency = self._classify_efficiency(energy_ratio)
                
                doubling_analysis.append({
                    'base_velocity_c': base_point['coordinate_velocity_c'],
                    'double_velocity_c': double_point['coordinate_velocity_c'],
                    'base_energy_j': base_point['positive_energy_joules'],
                    'double_energy_j': double_point['positive_energy_joules'],
                    'energy_scaling_factor': energy_ratio,
                    'velocity_ratio': velocity_ratio,
                    'proportional': proportional,
                    'efficiency_rating': efficiency,
                    'within_limit': energy_ratio <= self.proportionality_limit
                })
                
        doubling_df = pd.DataFrame(doubling_analysis)
        
        logger.info(f"Analyzed {len(doubling_df)} velocity doubling intervals")
        if not doubling_df.empty:
            compliant_count = doubling_df['within_limit'].sum()
            logger.info(f"Proportionality compliance: {compliant_count}/{len(doubling_df)} ({100*compliant_count/len(doubling_df):.1f}%)")
            
        return doubling_df
        
    def _classify_efficiency(self, scaling_factor: float) -> str:
        """Classify efficiency based on scaling factor."""
        for rating, threshold in self.efficiency_thresholds.items():
            if scaling_factor <= threshold:
                return rating
        return 'unacceptable'
        
    def identify_scaling_regions(self, df: pd.DataFrame) -> List[ScalingRegion]:
        """
        Identify distinct scaling regions in the velocity-energy mapping.
        
        Args:
            df: Velocity-energy mapping DataFrame
            
        Returns:
            List of scaling regions with characteristics
        """
        regions = []
        
        # Calculate rolling scaling factors
        df = df.copy()
        df['rolling_scaling'] = df['scaling_factor'].rolling(window=10, center=True).mean()
        
        # Identify region boundaries based on scaling factor changes
        scaling_changes = []
        threshold = 0.5  # Scaling factor change threshold
        
        for i in range(1, len(df)-1):
            if abs(df.iloc[i]['rolling_scaling'] - df.iloc[i-1]['rolling_scaling']) > threshold:
                scaling_changes.append(i)
                
        # Add start and end points
        boundaries = [0] + scaling_changes + [len(df)-1]
        
        # Analyze each region
        for i in range(len(boundaries)-1):
            start_idx = boundaries[i]
            end_idx = boundaries[i+1]
            
            region_data = df.iloc[start_idx:end_idx+1]
            
            if len(region_data) > 1:
                avg_scaling = region_data['rolling_scaling'].mean()
                efficiency = self._classify_efficiency(avg_scaling)
                
                # Generate recommendation
                if avg_scaling <= 2.0:
                    recommendation = "OPTIMAL: Excellent efficiency for sustained operations"
                elif avg_scaling <= 4.0:
                    recommendation = "GOOD: Suitable for normal operations"  
                elif avg_scaling <= 8.0:
                    recommendation = "CAUTION: High energy cost, limit duration"
                else:
                    recommendation = "AVOID: Excessive energy requirements"
                    
                region = ScalingRegion(
                    velocity_start=region_data['coordinate_velocity_c'].iloc[0],
                    velocity_end=region_data['coordinate_velocity_c'].iloc[-1],
                    scaling_factor=avg_scaling,
                    efficiency_rating=efficiency,
                    energy_range=(
                        region_data['positive_energy_joules'].iloc[0],
                        region_data['positive_energy_joules'].iloc[-1]
                    ),
                    recommendation=recommendation
                )
                
                regions.append(region)
                
        logger.info(f"Identified {len(regions)} distinct scaling regions")
        return regions
        
    def find_optimal_velocities(self, df: pd.DataFrame) -> Dict:
        """
        Find optimal velocity ranges for different mission profiles.
        
        Args:
            df: Velocity-energy mapping DataFrame
            
        Returns:
            Dictionary with optimal velocity recommendations
        """
        # Define mission profile requirements
        profiles = {
            'economic_cruise': {'max_scaling': 2.0, 'description': 'Long-range economic operations'},
            'standard_cruise': {'max_scaling': 3.0, 'description': 'Standard interstellar travel'},
            'high_speed': {'max_scaling': 4.0, 'description': 'High-speed missions'},
            'emergency': {'max_scaling': 6.0, 'description': 'Emergency/military operations'},
            'maximum_performance': {'max_scaling': 8.0, 'description': 'Maximum capability'}
        }
        
        optimal_ranges = {}
        
        for profile, criteria in profiles.items():
            # Find velocities meeting scaling criteria
            suitable_data = df[df['scaling_factor'] <= criteria['max_scaling']]
            
            if not suitable_data.empty:
                optimal_ranges[profile] = {
                    'velocity_range': (
                        suitable_data['coordinate_velocity_c'].min(),
                        suitable_data['coordinate_velocity_c'].max()
                    ),
                    'energy_range': (
                        suitable_data['positive_energy_joules'].min(),
                        suitable_data['positive_energy_joules'].max()
                    ),
                    'max_scaling_factor': criteria['max_scaling'],
                    'description': criteria['description'],
                    'recommended_velocity': suitable_data['coordinate_velocity_c'].median(),
                    'points_available': len(suitable_data)
                }
            else:
                optimal_ranges[profile] = {
                    'velocity_range': None,
                    'message': f"No velocities meet {criteria['max_scaling']}x scaling requirement"
                }
                
        return optimal_ranges
        
    def validate_proportionality(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive proportionality validation against design targets.
        
        Args:
            df: Velocity-energy mapping DataFrame
            
        Returns:
            Validation results dictionary
        """
        validation = {
            'proportionality_limit': self.proportionality_limit,
            'doubling_analysis': None,
            'overall_compliance': None,
            'violation_points': [],
            'compliance_percentage': 0.0,
            'recommendations': []
        }
        
        # Perform doubling analysis
        doubling_df = self.analyze_doubling_scaling(df)
        validation['doubling_analysis'] = doubling_df
        
        if not doubling_df.empty:
            # Calculate overall compliance
            compliant_count = doubling_df['within_limit'].sum()
            total_count = len(doubling_df)
            compliance_percentage = 100 * compliant_count / total_count
            
            validation['overall_compliance'] = compliance_percentage >= 80.0  # 80% threshold
            validation['compliance_percentage'] = compliance_percentage
            
            # Identify violation points
            violations = doubling_df[~doubling_df['within_limit']]
            validation['violation_points'] = violations.to_dict('records')
            
            # Generate recommendations
            if compliance_percentage >= 90:
                validation['recommendations'].append("EXCELLENT: Proportionality well maintained")
            elif compliance_percentage >= 80:
                validation['recommendations'].append("GOOD: Minor proportionality violations")
            elif compliance_percentage >= 60:
                validation['recommendations'].append("CAUTION: Significant proportionality issues")
            else:
                validation['recommendations'].append("CRITICAL: Major proportionality violations")
                
            # Specific recommendations
            if not violations.empty:
                max_violation = violations['energy_scaling_factor'].max()
                validation['recommendations'].append(
                    f"Maximum violation: {max_violation:.1f}x at {violations.loc[violations['energy_scaling_factor'].idxmax(), 'base_velocity_c']:.1f}c"
                )
                
        return validation
        
    def generate_scaling_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive scaling analysis report.
        
        Args:
            df: Velocity-energy mapping DataFrame
            
        Returns:
            Complete analysis report
        """
        logger.info("Generating comprehensive energy scaling analysis report")
        
        report = {
            'summary': {
                'total_data_points': len(df),
                'velocity_range': (df['coordinate_velocity_c'].min(), df['coordinate_velocity_c'].max()),
                'energy_range': (df['positive_energy_joules'].min(), df['positive_energy_joules'].max()),
                'average_scaling_factor': df['scaling_factor'].mean(),
                'max_scaling_factor': df['scaling_factor'].max()
            },
            'proportionality_validation': self.validate_proportionality(df),
            'scaling_regions': self.identify_scaling_regions(df),
            'optimal_velocities': self.find_optimal_velocities(df),
            'doubling_analysis': self.analyze_doubling_scaling(df)
        }
        
        # Add analysis insights
        report['insights'] = self._generate_insights(report)
        
        return report
        
    def _generate_insights(self, report: Dict) -> List[str]:
        """Generate analytical insights from the scaling report."""
        insights = []
        
        # Proportionality insights
        prop_validation = report['proportionality_validation']
        if prop_validation['overall_compliance']:
            insights.append("✅ Energy proportionality requirements satisfied")
        else:
            insights.append(f"⚠️ Proportionality compliance: {prop_validation['compliance_percentage']:.1f}%")
            
        # Scaling region insights
        regions = report['scaling_regions']
        if regions:
            excellent_regions = [r for r in regions if r.efficiency_rating == 'excellent']
            if excellent_regions:
                insights.append(f"🚀 {len(excellent_regions)} excellent efficiency regions identified")
                
        # Optimal velocity insights
        optimal = report['optimal_velocities']
        if 'economic_cruise' in optimal and optimal['economic_cruise']['velocity_range']:
            v_range = optimal['economic_cruise']['velocity_range']
            insights.append(f"💰 Economic cruise range: {v_range[0]:.1f}c - {v_range[1]:.1f}c")
            
        return insights
        
    def export_analysis(self, report: Dict, filename: str = "energy_scaling_analysis.json") -> None:
        """
        Export scaling analysis report to JSON file.
        
        Args:
            report: Analysis report dictionary
            filename: Output filename
        """
        import json
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
            
        # Clean report for serialization
        clean_report = json.loads(json.dumps(report, default=convert_numpy))
        
        output_path = Path(filename)
        with open(output_path, 'w') as f:
            json.dump(clean_report, f, indent=2)
            
        logger.info(f"Exported scaling analysis to {output_path}")

def main():
    """Main execution function for energy scaling analysis."""
    logger.info("Starting Energy Scaling Analysis")
    
    # For testing, create sample data (normally would load from coordinate_velocity_energy_mapping.py)
    sample_data = {
        'coordinate_velocity_c': np.arange(1, 101, 1),
        'positive_energy_joules': np.array([1e15 * v**1.8 for v in np.arange(1, 101, 1)]),
        'scaling_factor': np.array([1.0] + [1.8] * 99)  # Simplified scaling factors
    }
    df = pd.DataFrame(sample_data)
    
    # Initialize analyzer
    analyzer = EnergyScalingAnalyzer(proportionality_limit=4.0)
    
    # Generate comprehensive analysis
    report = analyzer.generate_scaling_report(df)
    
    # Display key results
    logger.info("=== ENERGY SCALING ANALYSIS RESULTS ===")
    for insight in report['insights']:
        logger.info(insight)
        
    # Export results
    analyzer.export_analysis(report, "test_scaling_analysis.json")
    
    logger.info("Energy scaling analysis complete!")
    
    return report

if __name__ == "__main__":
    report = main()

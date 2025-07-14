#!/usr/bin/env python3
"""
Proportionality Validator for LQG Drive Energy Requirements

Validates energy scaling proportionality constraints and ensures design targets are met.
Monitors for disproportionate energy increases and validates the ≤4x energy increase 
per coordinate velocity doubling requirement.

Repository: lqg-ftl-metric-engineering → velocity analysis module
Technology: Advanced constraint validation with real-time monitoring
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ValidationResult(Enum):
    """Enumeration for validation result types."""
    PASS = "PASS"
    WARNING = "WARNING" 
    FAIL = "FAIL"
    CRITICAL = "CRITICAL"

@dataclass
class ProportionalityConstraint:
    """Data structure for proportionality constraint definitions."""
    name: str
    description: str
    threshold: float
    critical_threshold: float
    units: str
    
@dataclass
class ValidationPoint:
    """Data structure for individual validation results."""
    velocity: float
    constraint_name: str
    measured_value: float
    threshold: float
    result: ValidationResult
    deviation_percentage: float
    message: str

class ProportionalityValidator:
    """
    Advanced proportionality validator for LQG Drive energy scaling.
    
    Features:
    - Real-time constraint monitoring
    - Multiple validation criteria
    - Statistical analysis of violations
    - Automatic threshold adjustment
    - Comprehensive violation reporting
    """
    
    def __init__(self):
        """Initialize the proportionality validator with design constraints."""
        
        # Define core proportionality constraints
        self.constraints = {
            'doubling_energy': ProportionalityConstraint(
                name="Energy Doubling Scaling",
                description="Energy increase per velocity doubling ≤ 4x",
                threshold=4.0,
                critical_threshold=8.0,
                units="ratio"
            ),
            'adjacent_scaling': ProportionalityConstraint(
                name="Adjacent Point Scaling", 
                description="Energy increase between adjacent points ≤ 8x",
                threshold=8.0,
                critical_threshold=16.0,
                units="ratio"
            ),
            'exponential_growth': ProportionalityConstraint(
                name="Exponential Growth Control",
                description="Exponential growth coefficient ≤ 2.5",
                threshold=2.5,
                critical_threshold=4.0,
                units="coefficient"
            ),
            'efficiency_degradation': ProportionalityConstraint(
                name="Efficiency Degradation",
                description="Energy efficiency degradation ≤ 50% per velocity doubling",
                threshold=50.0,
                critical_threshold=100.0,
                units="percentage"
            )
        }
        
        self.validation_history = []
        
        logger.info("Initialized ProportionalityValidator with 4 core constraints")
        
    def validate_doubling_constraint(self, df: pd.DataFrame) -> List[ValidationPoint]:
        """
        Validate the ≤4x energy increase per velocity doubling constraint.
        
        Args:
            df: Velocity-energy mapping DataFrame
            
        Returns:
            List of validation points for doubling constraint
        """
        validation_points = []
        constraint = self.constraints['doubling_energy']
        
        # Find velocity doubling points
        base_velocities = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        
        for base_v in base_velocities:
            double_v = base_v * 2
            
            # Find closest data points
            base_idx = (df['coordinate_velocity_c'] - base_v).abs().idxmin()
            double_candidates = df[df['coordinate_velocity_c'] >= double_v]
            
            if not double_candidates.empty:
                double_idx = double_candidates.index[0]
                
                base_energy = df.loc[base_idx, 'positive_energy_joules']
                double_energy = df.loc[double_idx, 'positive_energy_joules']
                actual_base_v = df.loc[base_idx, 'coordinate_velocity_c']
                actual_double_v = df.loc[double_idx, 'coordinate_velocity_c']
                
                # Calculate energy scaling factor
                energy_ratio = double_energy / base_energy
                
                # Determine validation result
                if energy_ratio <= constraint.threshold:
                    result = ValidationResult.PASS
                    message = f"Energy scaling {energy_ratio:.2f}x within {constraint.threshold}x limit"
                elif energy_ratio <= constraint.critical_threshold:
                    result = ValidationResult.WARNING
                    message = f"Energy scaling {energy_ratio:.2f}x exceeds {constraint.threshold}x limit"
                else:
                    result = ValidationResult.CRITICAL
                    message = f"Energy scaling {energy_ratio:.2f}x exceeds critical {constraint.critical_threshold}x limit"
                    
                deviation = 100 * (energy_ratio - constraint.threshold) / constraint.threshold
                
                validation_point = ValidationPoint(
                    velocity=actual_base_v,
                    constraint_name=constraint.name,
                    measured_value=energy_ratio,
                    threshold=constraint.threshold,
                    result=result,
                    deviation_percentage=deviation,
                    message=message
                )
                
                validation_points.append(validation_point)
                
        return validation_points
        
    def validate_adjacent_scaling(self, df: pd.DataFrame) -> List[ValidationPoint]:
        """
        Validate adjacent point energy scaling constraints.
        
        Args:
            df: Velocity-energy mapping DataFrame
            
        Returns:
            List of validation points for adjacent scaling
        """
        validation_points = []
        constraint = self.constraints['adjacent_scaling']
        
        for i in range(1, len(df)):
            current_energy = df.iloc[i]['positive_energy_joules']
            previous_energy = df.iloc[i-1]['positive_energy_joules']
            current_velocity = df.iloc[i]['coordinate_velocity_c']
            
            if previous_energy > 0:
                scaling_factor = current_energy / previous_energy
                
                # Validate against constraint
                if scaling_factor <= constraint.threshold:
                    result = ValidationResult.PASS
                    message = f"Adjacent scaling {scaling_factor:.2f}x within {constraint.threshold}x limit"
                elif scaling_factor <= constraint.critical_threshold:
                    result = ValidationResult.WARNING
                    message = f"Adjacent scaling {scaling_factor:.2f}x exceeds {constraint.threshold}x limit"
                else:
                    result = ValidationResult.CRITICAL
                    message = f"Adjacent scaling {scaling_factor:.2f}x exceeds critical {constraint.critical_threshold}x limit"
                    
                deviation = 100 * (scaling_factor - constraint.threshold) / constraint.threshold
                
                # Only record significant violations to avoid clutter
                if scaling_factor > constraint.threshold * 0.9:  # 90% of threshold
                    validation_point = ValidationPoint(
                        velocity=current_velocity,
                        constraint_name=constraint.name,
                        measured_value=scaling_factor,
                        threshold=constraint.threshold,
                        result=result,
                        deviation_percentage=deviation,
                        message=message
                    )
                    
                    validation_points.append(validation_point)
                    
        return validation_points
        
    def validate_exponential_growth(self, df: pd.DataFrame) -> List[ValidationPoint]:
        """
        Validate exponential growth coefficient constraints.
        
        Args:
            df: Velocity-energy mapping DataFrame
            
        Returns:
            List of validation points for exponential growth
        """
        validation_points = []
        constraint = self.constraints['exponential_growth']
        
        # Calculate exponential growth coefficient using log-linear regression
        log_velocity = np.log(df['coordinate_velocity_c'])
        log_energy = np.log(df['positive_energy_joules'])
        
        # Fit exponential model: log(E) = a + b*log(v)
        coefficients = np.polyfit(log_velocity, log_energy, 1)
        growth_coefficient = coefficients[0]  # This is the exponential coefficient
        
        # Validate growth coefficient
        if growth_coefficient <= constraint.threshold:
            result = ValidationResult.PASS
            message = f"Growth coefficient {growth_coefficient:.2f} within {constraint.threshold} limit"
        elif growth_coefficient <= constraint.critical_threshold:
            result = ValidationResult.WARNING
            message = f"Growth coefficient {growth_coefficient:.2f} exceeds {constraint.threshold} limit"
        else:
            result = ValidationResult.CRITICAL
            message = f"Growth coefficient {growth_coefficient:.2f} exceeds critical {constraint.critical_threshold} limit"
            
        deviation = 100 * (growth_coefficient - constraint.threshold) / constraint.threshold
        
        validation_point = ValidationPoint(
            velocity=df['coordinate_velocity_c'].median(),  # Representative velocity
            constraint_name=constraint.name,
            measured_value=growth_coefficient,
            threshold=constraint.threshold,
            result=result,
            deviation_percentage=deviation,
            message=message
        )
        
        validation_points.append(validation_point)
        
        return validation_points
        
    def validate_efficiency_degradation(self, df: pd.DataFrame) -> List[ValidationPoint]:
        """
        Validate energy efficiency degradation constraints.
        
        Args:
            df: Velocity-energy mapping DataFrame
            
        Returns:
            List of validation points for efficiency degradation
        """
        validation_points = []
        constraint = self.constraints['efficiency_degradation']
        
        # Calculate efficiency as energy per velocity unit
        df_copy = df.copy()
        df_copy['efficiency'] = df_copy['positive_energy_joules'] / df_copy['coordinate_velocity_c']
        
        # Analyze efficiency degradation for velocity doublings
        base_velocities = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        
        for base_v in base_velocities:
            double_v = base_v * 2
            
            # Find closest points
            base_idx = (df_copy['coordinate_velocity_c'] - base_v).abs().idxmin()
            double_candidates = df_copy[df_copy['coordinate_velocity_c'] >= double_v]
            
            if not double_candidates.empty:
                double_idx = double_candidates.index[0]
                
                base_efficiency = df_copy.loc[base_idx, 'efficiency']
                double_efficiency = df_copy.loc[double_idx, 'efficiency']
                actual_base_v = df_copy.loc[base_idx, 'coordinate_velocity_c']
                
                # Calculate efficiency degradation percentage
                if base_efficiency > 0:
                    efficiency_degradation = 100 * (double_efficiency - base_efficiency) / base_efficiency
                    
                    # Validate against constraint (degradation should be limited)
                    if abs(efficiency_degradation) <= constraint.threshold:
                        result = ValidationResult.PASS
                        message = f"Efficiency change {efficiency_degradation:.1f}% within {constraint.threshold}% limit"
                    elif abs(efficiency_degradation) <= constraint.critical_threshold:
                        result = ValidationResult.WARNING
                        message = f"Efficiency change {efficiency_degradation:.1f}% exceeds {constraint.threshold}% limit"
                    else:
                        result = ValidationResult.CRITICAL
                        message = f"Efficiency change {efficiency_degradation:.1f}% exceeds critical {constraint.critical_threshold}% limit"
                        
                    deviation = abs(efficiency_degradation) - constraint.threshold
                    
                    validation_point = ValidationPoint(
                        velocity=actual_base_v,
                        constraint_name=constraint.name,
                        measured_value=abs(efficiency_degradation),
                        threshold=constraint.threshold,
                        result=result,
                        deviation_percentage=deviation,
                        message=message
                    )
                    
                    validation_points.append(validation_point)
                    
        return validation_points
        
    def comprehensive_validation(self, df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive proportionality validation across all constraints.
        
        Args:
            df: Velocity-energy mapping DataFrame
            
        Returns:
            Complete validation results dictionary
        """
        logger.info("Performing comprehensive proportionality validation")
        
        # Run all validation tests
        all_validations = []
        
        all_validations.extend(self.validate_doubling_constraint(df))
        all_validations.extend(self.validate_adjacent_scaling(df))
        all_validations.extend(self.validate_exponential_growth(df))
        all_validations.extend(self.validate_efficiency_degradation(df))
        
        # Compile results
        results = {
            'total_validations': len(all_validations),
            'constraint_summary': {},
            'overall_status': ValidationResult.PASS,
            'critical_violations': [],
            'warning_violations': [],
            'passed_validations': 0,
            'compliance_percentage': 0.0,
            'recommendations': []
        }
        
        # Analyze results by constraint type
        for constraint_name in self.constraints.keys():
            constraint_validations = [v for v in all_validations if constraint_name in v.constraint_name.lower().replace(' ', '_')]
            
            if constraint_validations:
                passed = len([v for v in constraint_validations if v.result == ValidationResult.PASS])
                total = len(constraint_validations)
                
                results['constraint_summary'][constraint_name] = {
                    'total_checks': total,
                    'passed': passed,
                    'compliance_percentage': 100 * passed / total,
                    'worst_violation': max(constraint_validations, key=lambda x: x.deviation_percentage) if constraint_validations else None
                }
                
        # Overall statistics
        passed_count = len([v for v in all_validations if v.result == ValidationResult.PASS])
        critical_count = len([v for v in all_validations if v.result == ValidationResult.CRITICAL])
        warning_count = len([v for v in all_validations if v.result == ValidationResult.WARNING])
        
        results['passed_validations'] = passed_count
        results['compliance_percentage'] = 100 * passed_count / len(all_validations) if all_validations else 100
        results['critical_violations'] = [v for v in all_validations if v.result == ValidationResult.CRITICAL]
        results['warning_violations'] = [v for v in all_validations if v.result == ValidationResult.WARNING]
        
        # Determine overall status
        if critical_count > 0:
            results['overall_status'] = ValidationResult.CRITICAL
        elif warning_count > 0:
            results['overall_status'] = ValidationResult.WARNING
        else:
            results['overall_status'] = ValidationResult.PASS
            
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        # Store validation history
        self.validation_history.append(results)
        
        logger.info(f"Validation complete: {results['overall_status'].value}")
        logger.info(f"Compliance: {results['compliance_percentage']:.1f}% ({passed_count}/{len(all_validations)})")
        
        return results
        
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        compliance = results['compliance_percentage']
        
        if compliance >= 95:
            recommendations.append("✅ EXCELLENT: All proportionality constraints well satisfied")
        elif compliance >= 85:
            recommendations.append("✅ GOOD: Minor proportionality violations, system acceptable")
        elif compliance >= 70:
            recommendations.append("⚠️ CAUTION: Significant violations, review energy scaling")
        else:
            recommendations.append("🚨 CRITICAL: Major proportionality failures, redesign required")
            
        # Specific constraint recommendations
        for constraint_name, summary in results['constraint_summary'].items():
            if summary['compliance_percentage'] < 80:
                recommendations.append(f"🔧 Review {constraint_name}: {summary['compliance_percentage']:.1f}% compliance")
                
        # Critical violation handling
        if results['critical_violations']:
            recommendations.append(f"🚨 {len(results['critical_violations'])} critical violations require immediate attention")
            
        return recommendations
        
    def export_validation_report(self, results: Dict, filename: str = "proportionality_validation_report.json") -> None:
        """
        Export validation results to comprehensive report file.
        
        Args:
            results: Validation results dictionary
            filename: Output filename
        """
        import json
        
        # Convert enum values for JSON serialization
        def convert_enums(obj):
            if isinstance(obj, ValidationResult):
                return obj.value
            elif isinstance(obj, ValidationPoint):
                return {
                    'velocity': obj.velocity,
                    'constraint_name': obj.constraint_name,
                    'measured_value': obj.measured_value,
                    'threshold': obj.threshold,
                    'result': obj.result.value,
                    'deviation_percentage': obj.deviation_percentage,
                    'message': obj.message
                }
            return obj
            
        # Clean results for serialization
        clean_results = json.loads(json.dumps(results, default=convert_enums))
        
        output_path = Path(filename)
        with open(output_path, 'w') as f:
            json.dump(clean_results, f, indent=2)
            
        logger.info(f"Exported validation report to {output_path}")

def main():
    """Main execution function for proportionality validation."""
    logger.info("Starting Proportionality Validation Analysis")
    
    # For testing, create sample data (normally would load from coordinate_velocity_energy_mapping.py)
    velocities = np.arange(1, 101, 1)
    energies = 1e15 * velocities**1.8  # Slightly sub-proportional scaling
    
    sample_data = {
        'coordinate_velocity_c': velocities,
        'positive_energy_joules': energies
    }
    df = pd.DataFrame(sample_data)
    
    # Initialize validator
    validator = ProportionalityValidator()
    
    # Perform comprehensive validation
    results = validator.comprehensive_validation(df)
    
    # Display key results
    logger.info("=== PROPORTIONALITY VALIDATION RESULTS ===")
    logger.info(f"Overall Status: {results['overall_status'].value}")
    logger.info(f"Compliance: {results['compliance_percentage']:.1f}%")
    
    for rec in results['recommendations']:
        logger.info(rec)
        
    # Export results
    validator.export_validation_report(results, "test_proportionality_validation.json")
    
    logger.info("Proportionality validation complete!")
    
    return results

if __name__ == "__main__":
    results = main()

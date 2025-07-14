"""
LQG FTL Metric Engineering Core Framework
=========================================

LQG Drive Coordinate Velocity and Energy Requirements Analysis
Complete implementation of coordinate velocity performance mapping with non-exotic energy optimization.
"""

__version__ = "2.0.0"
__author__ = "LQG FTL Metric Engineering Team"

# Import our actual coordinate velocity analysis modules
try:
    from .coordinate_velocity_energy_mapping import CoordinateVelocityMapper
    from .energy_scaling_analyzer import EnergyScalingAnalyzer
    from .proportionality_validator import ProportionalityValidator
    from .smear_time_calculator import SmearTimeCalculator
    from .tidal_force_calculator import TidalForceCalculator
    from .performance_table_generator import PerformanceTableGenerator
    from .csv_export_system import CSVExportSystem
    from .optimization_recommender import OptimizationRecommender
    
    __all__ = [
        'CoordinateVelocityMapper',
        'EnergyScalingAnalyzer',
        'ProportionalityValidator',
        'SmearTimeCalculator',
        'TidalForceCalculator',
        'PerformanceTableGenerator',
        'CSVExportSystem',
        'OptimizationRecommender'
    ]
except ImportError as e:
    # Graceful degradation if modules aren't available
    print(f"Warning: Some LQG Drive analysis modules not available: {e}")
    __all__ = []

# Core constants for LQG Drive analysis
EXACT_BACKREACTION_FACTOR = 4.2e-9  # From cross-repository validation
LQG_ALPHA_PARAMETER = 0.2375  # Dimensionless LQG coupling

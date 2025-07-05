"""
LQG FTL Metric Engineering Core Framework
=========================================

First steps towards achieving traversable geometries with finite or zero exotic energy requirements.
Implements validated mathematical frameworks from cross-repository analysis.
"""

__version__ = "1.0.0"
__author__ = "LQG FTL Metric Engineering Team"

from traversable_geometries import (
    TraversableGeometryFramework,
    LQGWormholeImplementation,
    BobrickMartirePositiveEnergyShapes,
    MorrisThorneFiniteEnergyDesign
)

from constants import (
    EXACT_BACKREACTION_FACTOR,
    LQG_ALPHA_PARAMETER,
    VALIDATED_CONSTANTS
)

__all__ = [
    'TraversableGeometryFramework',
    'LQGWormholeImplementation', 
    'BobrickMartirePositiveEnergyShapes',
    'MorrisThorneFiniteEnergyDesign',
    'EXACT_BACKREACTION_FACTOR',
    'LQG_ALPHA_PARAMETER',
    'VALIDATED_CONSTANTS'
]

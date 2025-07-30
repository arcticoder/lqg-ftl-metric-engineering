"""
V&V #EnergyDistribution:
  Verify EnergyComponentAnalyzer.analyze_energy_distribution()

Checks that the returned dict has the expected structure and
that the recoverable_energy for one component matches a spot‐check.
"""
import numpy as np
import pytest
from energy_optimization.energy_component_analyzer import EnergyComponentAnalyzer

def test_analyze_energy_distribution_output():
    e = EnergyComponentAnalyzer()
    result = e.analyze_energy_distribution()
    print(f"\n=== Energy Distribution Analysis Output ===\n{result}\n")

    # structure checks
    assert isinstance(result, dict)
    assert 'components' in result and isinstance(result['components'], dict)

    comps = result['components']
    # spot‐check that spacetime_curvature appears
    assert 'spacetime_curvature' in comps

    # spot‐check one numeric value
    recov = comps['spacetime_curvature']['recoverable_energy']
    # expected: initial 2.7e9 J minus 12% loss
    expected = 2.7e9 * (1 - 0.12)
    assert np.isclose(recov, expected, atol=1e3), (
        f"recoverable_energy {recov} != {expected}"
    )

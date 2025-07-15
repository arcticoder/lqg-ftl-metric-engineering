#!/usr/bin/env python3
"""
Phase 1 Energy Analysis Test Suite

Comprehensive test suite for the energy optimization analysis framework
validating all components of the 100× energy reduction methodology.
"""

import sys
import os
import unittest
import json
import tempfile
from unittest.mock import patch, MagicMock

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from energy_component_analyzer import EnergyComponentAnalyzer, EnergyComponent, EnergyAnalysis
    from optimization_target_identifier import OptimizationTargetIdentifier, OptimizationTarget, OptimizationPriority
    from energy_loss_evaluator import EnergyLossEvaluator, EnergyLoss, LossAnalysis
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    print("Make sure to run this test from the lqg-ftl-metric-engineering directory")
    sys.exit(1)

class TestEnergyComponentAnalyzer(unittest.TestCase):
    """Test suite for EnergyComponentAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = EnergyComponentAnalyzer()
    
    def test_initialization(self):
        """Test analyzer initialization."""
        self.assertEqual(self.analyzer.c, 299792458)
        self.assertEqual(self.analyzer.G, 6.674e-11)
        self.assertIsInstance(self.analyzer.corolla_baseline, dict)
        self.assertIn('energy_J', self.analyzer.corolla_baseline)
    
    def test_analyze_corolla_sized_bubble(self):
        """Test Corolla-sized bubble analysis."""
        analysis = self.analyzer.analyze_corolla_sized_bubble()
        
        # Validate analysis structure
        self.assertIsInstance(analysis, EnergyAnalysis)
        self.assertGreater(analysis.total_energy, 0)
        self.assertIsInstance(analysis.components, list)
        self.assertGreater(len(analysis.components), 0)
        
        # Validate components
        for component in analysis.components:
            self.assertIsInstance(component, EnergyComponent)
            self.assertGreater(component.energy_joules, 0)
            self.assertGreaterEqual(component.percentage, 0)
            self.assertLessEqual(component.percentage, 100)
            self.assertGreaterEqual(component.optimization_potential, 0)
            self.assertLessEqual(component.optimization_potential, 1)
        
        # Check total percentages sum to ~100%
        total_percentage = sum(comp.percentage for comp in analysis.components)
        self.assertAlmostEqual(total_percentage, 100.0, delta=5.0)
    
    def test_energy_component_calculations(self):
        """Test individual energy component calculations."""
        radius = 1.44
        velocity = 8.33
        
        # Test raw Alcubierre energy calculation
        raw_energy = self.analyzer._calculate_raw_alcubierre_energy(radius, velocity)
        self.assertGreater(raw_energy, 0)
        self.assertIsInstance(raw_energy, float)
        
        # Test curvature energy calculation
        curvature_energy = self.analyzer._calculate_curvature_energy(radius, velocity)
        self.assertGreater(curvature_energy, 0)
        
        # Test wall maintenance energy
        wall_energy = self.analyzer._calculate_wall_maintenance_energy(radius, 240)
        self.assertGreater(wall_energy, 0)
    
    def test_optimization_opportunities(self):
        """Test optimization opportunity identification."""
        analysis = self.analyzer.analyze_corolla_sized_bubble()
        opportunities = self.analyzer.identify_optimization_opportunities(analysis)
        
        self.assertIsInstance(opportunities, dict)
        for comp_name, opp in opportunities.items():
            self.assertIn('current_energy', opp)
            self.assertIn('optimization_potential', opp)
            self.assertIn('strategies', opp)
            self.assertIn('theoretical_reduction_factor', opp)
            self.assertGreater(opp['current_energy'], 0)
            self.assertGreater(opp['theoretical_reduction_factor'], 1.0)
    
    def test_export_functionality(self):
        """Test analysis export functionality."""
        analysis = self.analyzer.analyze_corolla_sized_bubble()
        roadmap = self.analyzer.generate_optimization_roadmap(analysis)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filename = f.name
        
        try:
            self.analyzer.export_analysis(analysis, roadmap, temp_filename)
            
            # Verify file was created and contains valid JSON
            self.assertTrue(os.path.exists(temp_filename))
            with open(temp_filename, 'r') as f:
                data = json.load(f)
            
            self.assertIn('total_energy_J', data)
            self.assertIn('components', data)
            self.assertIn('roadmap', data)
            
        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

class TestOptimizationTargetIdentifier(unittest.TestCase):
    """Test suite for OptimizationTargetIdentifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.identifier = OptimizationTargetIdentifier()
    
    def test_geometric_optimization_targets(self):
        """Test geometric optimization target identification."""
        targets = self.identifier.identify_geometric_optimization_targets()
        
        self.assertIsInstance(targets, list)
        self.assertGreater(len(targets), 0)
        
        for target in targets:
            self.assertIsInstance(target, OptimizationTarget)
            self.assertGreater(target.reduction_factor, 1.0)
            self.assertGreaterEqual(target.expected_energy_impact, 0)
            self.assertLessEqual(target.expected_energy_impact, 1.0)
            self.assertIn(target.implementation_difficulty, ["Low", "Medium", "High"])
            self.assertIn(target.physics_risk, ["Low", "Medium", "High"])
    
    def test_temporal_optimization_targets(self):
        """Test temporal optimization target identification."""
        targets = self.identifier.identify_temporal_optimization_targets()
        
        self.assertIsInstance(targets, list)
        self.assertGreater(len(targets), 0)
        
        # Check for expected temporal targets
        target_names = [target.name for target in targets]
        self.assertIn("Variable Smearing Optimization", target_names)
        self.assertIn("Optimal Acceleration Profile", target_names)
    
    def test_advanced_optimization_targets(self):
        """Test advanced optimization target identification."""
        targets = self.identifier.identify_advanced_optimization_targets()
        
        self.assertIsInstance(targets, list)
        self.assertGreater(len(targets), 0)
        
        # Check for expected advanced targets
        target_names = [target.name for target in targets]
        self.assertIn("Warp Field Energy Recycling", target_names)
        self.assertIn("Quantum Field Enhancement", target_names)
    
    def test_target_prioritization(self):
        """Test optimization target prioritization."""
        # Create mock targets for testing
        targets = self.identifier.identify_geometric_optimization_targets()
        priorities = self.identifier.prioritize_optimization_targets(targets)
        
        self.assertIsInstance(priorities, list)
        self.assertEqual(len(priorities), len(targets))
        
        # Check prioritization order (should be sorted by priority score)
        for i in range(len(priorities) - 1):
            self.assertGreaterEqual(priorities[i].priority_score, priorities[i+1].priority_score)
        
        # Check priority structure
        for priority in priorities:
            self.assertIsInstance(priority, OptimizationPriority)
            self.assertGreater(priority.priority_score, 0)
            self.assertGreater(priority.implementation_order, 0)
    
    def test_optimization_potential_calculation(self):
        """Test total optimization potential calculation."""
        targets = self.identifier.identify_geometric_optimization_targets()
        priorities = self.identifier.prioritize_optimization_targets(targets)
        potential = self.identifier.calculate_total_optimization_potential(priorities)
        
        self.assertIsInstance(potential, dict)
        self.assertIn('total_reduction_potential', potential)
        self.assertIn('feasibility_assessment', potential)
        self.assertIn('implementation_phases', potential)
        
        self.assertGreater(potential['total_reduction_potential'], 1.0)
        self.assertIn(potential['feasibility_assessment'], 
                     ["HIGHLY ACHIEVABLE", "ACHIEVABLE", "CHALLENGING"])

class TestEnergyLossEvaluator(unittest.TestCase):
    """Test suite for EnergyLossEvaluator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = EnergyLossEvaluator()
    
    def test_initialization(self):
        """Test evaluator initialization."""
        self.assertEqual(self.evaluator.corolla_energy, 520359)
        self.assertEqual(self.evaluator.warp_current_energy, 5.4e9)
        self.assertAlmostEqual(self.evaluator.energy_excess_ratio, 10373, delta=100)
    
    def test_fundamental_loss_evaluation(self):
        """Test fundamental energy loss evaluation."""
        losses = self.evaluator.evaluate_fundamental_losses()
        
        self.assertIsInstance(losses, list)
        self.assertGreater(len(losses), 0)
        
        for loss in losses:
            self.assertIsInstance(loss, EnergyLoss)
            self.assertGreater(loss.current_loss_J, 0)
            self.assertGreaterEqual(loss.loss_percentage, 0)
            self.assertLessEqual(loss.loss_percentage, 100)
            self.assertGreaterEqual(loss.recoverability, 0)
            self.assertLessEqual(loss.recoverability, 1)
            self.assertGreater(loss.estimated_recovery_J, 0)
    
    def test_systematic_inefficiency_evaluation(self):
        """Test systematic inefficiency evaluation."""
        inefficiencies = self.evaluator.evaluate_systematic_inefficiencies()
        
        self.assertIsInstance(inefficiencies, list)
        self.assertGreater(len(inefficiencies), 0)
        
        # Check that inefficiencies have proper structure
        for ineff in inefficiencies:
            self.assertIsInstance(ineff, EnergyLoss)
            self.assertIn(ineff.implementation_complexity, ["Low", "Medium", "High"])
    
    def test_thermodynamic_loss_evaluation(self):
        """Test thermodynamic loss evaluation."""
        thermo_losses = self.evaluator.evaluate_thermodynamic_losses()
        
        self.assertIsInstance(thermo_losses, list)
        self.assertGreater(len(thermo_losses), 0)
        
        # Check for expected thermodynamic losses
        loss_names = [loss.name for loss in thermo_losses]
        self.assertIn("Quantum Decoherence Energy Loss", loss_names)
        self.assertIn("Vacuum Fluctuation Coupling Loss", loss_names)
    
    def test_total_loss_analysis(self):
        """Test comprehensive loss analysis."""
        fundamental = self.evaluator.evaluate_fundamental_losses()
        systematic = self.evaluator.evaluate_systematic_inefficiencies()
        thermodynamic = self.evaluator.evaluate_thermodynamic_losses()
        
        analysis = self.evaluator.calculate_total_loss_analysis(
            fundamental, systematic, thermodynamic
        )
        
        self.assertIsInstance(analysis, LossAnalysis)
        self.assertGreater(analysis.total_energy_loss, 0)
        self.assertGreater(analysis.recoverable_energy, 0)
        self.assertGreaterEqual(analysis.recovery_potential, 0)
        self.assertLessEqual(analysis.recovery_potential, 1)
        self.assertGreater(analysis.theoretical_minimum, 0)
        
        # Check that theoretical minimum is less than current energy
        self.assertLess(analysis.theoretical_minimum, self.evaluator.warp_current_energy)
    
    def test_recovery_prioritization(self):
        """Test energy recovery prioritization."""
        fundamental = self.evaluator.evaluate_fundamental_losses()
        systematic = self.evaluator.evaluate_systematic_inefficiencies()
        thermodynamic = self.evaluator.evaluate_thermodynamic_losses()
        
        analysis = self.evaluator.calculate_total_loss_analysis(
            fundamental, systematic, thermodynamic
        )
        
        prioritized = self.evaluator.prioritize_recovery_opportunities(analysis)
        
        self.assertIsInstance(prioritized, list)
        self.assertEqual(len(prioritized), len(analysis.loss_mechanisms))
        
        # Check that prioritization is meaningful (high-impact items first)
        if len(prioritized) > 1:
            # First item should have high energy impact or high recoverability
            first_item = prioritized[0]
            self.assertTrue(
                first_item.estimated_recovery_J > self.evaluator.warp_current_energy * 0.05 or
                first_item.recoverability > 0.7
            )
    
    def test_recovery_roadmap_generation(self):
        """Test recovery roadmap generation."""
        fundamental = self.evaluator.evaluate_fundamental_losses()
        systematic = self.evaluator.evaluate_systematic_inefficiencies()
        thermodynamic = self.evaluator.evaluate_thermodynamic_losses()
        
        analysis = self.evaluator.calculate_total_loss_analysis(
            fundamental, systematic, thermodynamic
        )
        
        prioritized = self.evaluator.prioritize_recovery_opportunities(analysis)
        roadmap = self.evaluator.generate_recovery_roadmap(prioritized)
        
        self.assertIsInstance(roadmap, dict)
        self.assertIn('overview', roadmap)
        self.assertIn('phase_1', roadmap)
        self.assertIn('phase_2', roadmap)
        self.assertIn('phase_3', roadmap)
        self.assertIn('cumulative_impact', roadmap)
        
        # Check that phases have proper structure
        for phase_name in ['phase_1', 'phase_2', 'phase_3']:
            phase = roadmap[phase_name]
            self.assertIn('title', phase)
            self.assertIn('duration', phase)
            self.assertIn('targets', phase)
            self.assertIn('recovery_potential', phase)
            self.assertIn('reduction_factor', phase)
            self.assertGreater(phase['reduction_factor'], 1.0)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete energy analysis framework."""
    
    def test_full_analysis_pipeline(self):
        """Test complete analysis pipeline from start to finish."""
        # Initialize all analyzers
        analyzer = EnergyComponentAnalyzer()
        identifier = OptimizationTargetIdentifier()
        evaluator = EnergyLossEvaluator()
        
        # Run complete analysis
        energy_analysis = analyzer.analyze_corolla_sized_bubble()
        loss_analysis = evaluator.calculate_total_loss_analysis(
            evaluator.evaluate_fundamental_losses(),
            evaluator.evaluate_systematic_inefficiencies(),
            evaluator.evaluate_thermodynamic_losses()
        )
        
        # Check consistency between analyses
        self.assertAlmostEqual(
            energy_analysis.total_energy, 
            evaluator.warp_current_energy, 
            delta=evaluator.warp_current_energy * 0.1  # Within 10%
        )
        
        # Check that total recoverable energy is less than total energy
        self.assertLess(loss_analysis.recoverable_energy, evaluator.warp_current_energy)
        
        # Check 100× target feasibility
        target_energy = evaluator.corolla_energy * 100
        final_energy = evaluator.warp_current_energy - loss_analysis.recoverable_energy
        
        # Should be achievable or close to achievable
        self.assertLess(final_energy, target_energy * 2)  # Within 2× of target
    
    def test_energy_conservation(self):
        """Test energy conservation in calculations."""
        evaluator = EnergyLossEvaluator()
        
        fundamental = evaluator.evaluate_fundamental_losses()
        total_losses = sum(loss.current_loss_J for loss in fundamental)
        
        # Total losses should not exceed current energy
        self.assertLessEqual(total_losses, evaluator.warp_current_energy)
        
        # Recoverable energy should not exceed total losses
        total_recoverable = sum(loss.estimated_recovery_J for loss in fundamental)
        self.assertLessEqual(total_recoverable, total_losses)
    
    def test_validation_framework_completeness(self):
        """Test that validation framework covers all necessary aspects."""
        identifier = OptimizationTargetIdentifier()
        
        # Create mock data for testing
        targets = identifier.identify_geometric_optimization_targets()
        priorities = identifier.prioritize_optimization_targets(targets)
        potential = identifier.calculate_total_optimization_potential(priorities)
        plan = identifier.generate_implementation_plan(priorities, potential)
        
        # Check validation framework completeness
        validation = plan['validation_framework']
        
        required_validations = [
            'physics_validation',
            'performance_validation', 
            'safety_validation'
        ]
        
        for validation_type in required_validations:
            self.assertIn(validation_type, validation)
            self.assertIsInstance(validation[validation_type], list)
            self.assertGreater(len(validation[validation_type]), 0)

def run_comprehensive_tests():
    """Run all tests with detailed reporting."""
    print("🧪 RUNNING PHASE 1 ENERGY ANALYSIS TEST SUITE")
    print("=" * 70)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestEnergyComponentAnalyzer,
        TestOptimizationTargetIdentifier,
        TestEnergyLossEvaluator,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Summary
    print(f"\n📊 TEST RESULTS SUMMARY:")
    print(f"   • Tests run: {result.testsRun}")
    print(f"   • Failures: {len(result.failures)}")
    print(f"   • Errors: {len(result.errors)}")
    print(f"   • Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"   • {test}: {traceback.split(chr(10))[-2]}")
    
    if result.errors:
        print(f"\n🚨 ERRORS:")
        for test, traceback in result.errors:
            print(f"   • {test}: {traceback.split(chr(10))[-2]}")
    
    if result.wasSuccessful():
        print(f"\n✅ ALL TESTS PASSED - PHASE 1 ANALYSIS FRAMEWORK VALIDATED")
        print(f"   → Ready for implementation of energy optimization targets")
        print(f"   → Framework validated for 100× energy reduction methodology")
    else:
        print(f"\n⚠️ SOME TESTS FAILED - REVIEW REQUIRED")
        print(f"   → Address test failures before proceeding")
        print(f"   → Validate framework components individually")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)

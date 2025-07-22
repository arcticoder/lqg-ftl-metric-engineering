#!/usr/bin/env python3
import sys
import traceback

def test_radiation():
    try:
        from radiation_shielding_optimizer import AdvancedRadiationShieldingOptimizer
        optimizer = AdvancedRadiationShieldingOptimizer()
        results = optimizer.generate_shielding_optimization_report()
        print("✅ RADIATION TEST SUCCESSFUL")
        return True
    except Exception as e:
        print(f"❌ RADIATION ERROR: {e}")
        traceback.print_exc()
        return False

def test_magnetic():
    try:
        from magnetic_stability_enhancer import MagneticStabilityEnhancer
        enhancer = MagneticStabilityEnhancer()
        results = enhancer.generate_stability_enhancement_report()
        print("✅ MAGNETIC TEST SUCCESSFUL")
        return True
    except Exception as e:
        print(f"❌ MAGNETIC ERROR: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 DIRECT MODULE TESTING")
    print("=" * 30)
    
    print("\n🛡️ Testing Radiation Shielding...")
    rad_success = test_radiation()
    
    print("\n🧲 Testing Magnetic Stability...")
    mag_success = test_magnetic()
    
    print(f"\n📊 RESULTS:")
    print(f"Radiation: {'✅ PASS' if rad_success else '❌ FAIL'}")
    print(f"Magnetic: {'✅ PASS' if mag_success else '❌ FAIL'}")
    
    if rad_success and mag_success:
        print("🎯 ALL TESTS SUCCESSFUL!")
    else:
        print("⚠️ SOME TESTS FAILED")

#!/usr/bin/env python3
import sys
import traceback

def test_radiation_only():
    try:
        print("🛡️ Testing radiation shielding...")
        from radiation_shielding_optimizer import AdvancedRadiationShieldingOptimizer
        
        print("✅ Module imported successfully")
        optimizer = AdvancedRadiationShieldingOptimizer()
        print("✅ Optimizer initialized")
        
        # Test simple dose calculation first
        simple_stack = [('tungsten', 0.1), ('concrete', 1.0)]
        dose_result = optimizer.calculate_dose_rate(simple_stack)
        print(f"✅ Simple dose calculation: {dose_result['total_dose_Sv_year']*1000:.2f} mSv/year")
        
        # Test optimization
        print("🔧 Running optimization...")
        optimization_result = optimizer.optimize_shielding_design()
        print(f"✅ Optimization completed: {optimization_result['optimization_success']}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 ISOLATED RADIATION TEST")
    print("=" * 30)
    success = test_radiation_only()
    print(f"\n📊 Result: {'✅ SUCCESS' if success else '❌ FAILED'}")

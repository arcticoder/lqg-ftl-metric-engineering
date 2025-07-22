#!/usr/bin/env python3
import numpy as np

# Simple test to isolate the issue
try:
    print("Testing basic imports...")
    from scipy.optimize import minimize
    print("✅ scipy imported")
    
    print("Testing basic calculation...")
    # Simple exponential attenuation
    neutron_flux = 1e15
    attenuation = 850.0
    thickness = 0.1
    transmission = np.exp(-attenuation * thickness)
    print(f"✅ Transmission: {transmission:.2e}")
    
    # Simple optimization test
    def simple_objective(x):
        return x[0]**2 + x[1]**2
    
    print("Testing minimize...")
    result = minimize(simple_objective, [1.0, 1.0], options={'maxiter': 10})
    print(f"✅ Minimize result: {result.success}")
    
    print("🎯 All basic tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

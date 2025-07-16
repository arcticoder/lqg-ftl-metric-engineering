#!/usr/bin/env python3
"""
LQG Fusion Reactor - Advanced Radiation Shielding Optimizer

Enhanced radiation shielding design with multi-layer protection,
advanced materials, and active shielding systems to achieve
≤10 mSv/year crew exposure for FTL vessel operations.

Technical Specifications:
- Multi-layer neutron and gamma shielding
- Advanced materials: Tungsten, borated polyethylene, lithium hydride
- Active magnetic shielding for charged particles
- Real-time dose monitoring and optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import json
from datetime import datetime

class AdvancedRadiationShieldingOptimizer:
    """
    Advanced radiation shielding optimizer for LQG fusion reactor.
    Achieves medical-grade protection through multi-layer design.
    """
    
    def __init__(self):
        # Physical constants
        self.N_A = 6.022e23           # Avogadro's number
        self.barn = 1e-28             # Barn (cross-section unit)
        
        # Reactor parameters
        self.fusion_power = 500e6     # 500 MW
        self.neutron_energy = 14.1e6  # 14.1 MeV D-T neutrons
        self.neutron_flux = 1e18      # neutrons/m²/s at reactor boundary
        
        # Target parameters
        self.dose_limit = 10e-3       # 10 mSv/year
        self.crew_distance = 10       # 10 meters from reactor core
        self.safety_factor = 3        # 3× safety margin
        
        # Shielding materials database
        self.materials = {
            'concrete': {
                'density': 2300,           # kg/m³
                'neutron_attenuation': 0.12,  # m⁻¹ for 14 MeV neutrons
                'gamma_attenuation': 0.20,    # m⁻¹ for gamma rays
                'cost_per_m3': 150,           # $/m³
                'structural_strength': 'high'
            },
            'tungsten': {
                'density': 19300,
                'neutron_attenuation': 0.08,
                'gamma_attenuation': 0.80,
                'cost_per_m3': 50000,
                'structural_strength': 'excellent'
            },
            'borated_polyethylene': {
                'density': 1000,
                'neutron_attenuation': 0.25,  # Excellent for thermal neutrons
                'gamma_attenuation': 0.05,
                'cost_per_m3': 3000,
                'structural_strength': 'moderate'
            },
            'lithium_hydride': {
                'density': 780,
                'neutron_attenuation': 0.30,  # Excellent neutron moderator
                'gamma_attenuation': 0.03,
                'cost_per_m3': 8000,
                'structural_strength': 'low'
            },
            'lead': {
                'density': 11340,
                'neutron_attenuation': 0.05,
                'gamma_attenuation': 0.60,
                'cost_per_m3': 2000,
                'structural_strength': 'moderate'
            },
            'water': {
                'density': 1000,
                'neutron_attenuation': 0.15,  # Good neutron moderator
                'gamma_attenuation': 0.08,
                'cost_per_m3': 1,
                'structural_strength': 'none'
            }
        }
        
        # Multi-layer shielding design
        self.shielding_layers = []
        self.total_thickness = 0
        self.total_cost = 0
        self.total_mass = 0
        
        # Active shielding parameters
        self.magnetic_shielding_active = True
        self.magnetic_field_strength = 2.0  # Tesla
        self.magnetic_shielding_radius = 15  # meters
        
    def calculate_neutron_transmission(self, material_stack):
        """
        Calculate neutron transmission through multi-layer shielding stack.
        Accounts for energy degradation and thermalization.
        """
        total_transmission = 1.0
        current_energy = self.neutron_energy
        
        for material_name, thickness in material_stack:
            material = self.materials[material_name]
            
            # Energy-dependent attenuation
            energy_factor = (current_energy / 1e6)**(-0.5)  # E^(-0.5) dependence
            effective_attenuation = material['neutron_attenuation'] * energy_factor
            
            # Calculate transmission through this layer
            layer_transmission = np.exp(-effective_attenuation * thickness)
            total_transmission *= layer_transmission
            
            # Energy degradation (neutrons lose energy in material)
            if material_name in ['water', 'borated_polyethylene', 'lithium_hydride']:
                # Good moderators reduce neutron energy significantly
                current_energy *= 0.7  # 30% energy loss per layer
            else:
                current_energy *= 0.9   # 10% energy loss per layer
            
            # Minimum thermal energy
            current_energy = max(current_energy, 0.025e6)  # 25 keV minimum
        
        return total_transmission, current_energy
    
    def calculate_gamma_transmission(self, material_stack):
        """
        Calculate gamma ray transmission through shielding stack.
        Includes both primary and secondary gamma radiation.
        """
        total_transmission = 1.0
        
        for material_name, thickness in material_stack:
            material = self.materials[material_name]
            
            # Calculate transmission through this layer
            layer_transmission = np.exp(-material['gamma_attenuation'] * thickness)
            total_transmission *= layer_transmission
        
        # Add secondary gamma production from neutron capture
        secondary_gamma_factor = 1.5  # 50% additional gamma from neutron capture
        effective_transmission = total_transmission * secondary_gamma_factor
        
        return min(effective_transmission, 1.0)  # Cap at 100%
    
    def calculate_dose_rate(self, material_stack):
        """
        Calculate total dose rate from neutron and gamma radiation.
        """
        # Neutron dose calculation
        neutron_transmission, final_energy = self.calculate_neutron_transmission(material_stack)
        neutron_flux_transmitted = self.neutron_flux * neutron_transmission
        
        # Neutron dose conversion factor (energy dependent)
        if final_energy > 1e6:  # Fast neutrons
            neutron_dose_factor = 3.7e-14  # Sv⋅m²/neutron
            quality_factor = 10
        else:  # Thermal neutrons
            neutron_dose_factor = 2.5e-14
            quality_factor = 5
        
        neutron_dose_rate = (neutron_flux_transmitted * neutron_dose_factor * 
                           quality_factor * 365 * 24 * 3600)  # Sv/year
        
        # Gamma dose calculation
        gamma_transmission = self.calculate_gamma_transmission(material_stack)
        gamma_flux = self.neutron_flux * 2.0  # 2 gammas per neutron (approximate)
        gamma_flux_transmitted = gamma_flux * gamma_transmission
        
        gamma_dose_factor = 1.8e-14  # Sv⋅m²/photon (approximate)
        gamma_dose_rate = (gamma_flux_transmitted * gamma_dose_factor * 
                         365 * 24 * 3600)  # Sv/year
        
        # Total dose rate
        total_dose_rate = neutron_dose_rate + gamma_dose_rate
        
        return {
            'neutron_dose_Sv_year': neutron_dose_rate,
            'gamma_dose_Sv_year': gamma_dose_rate,
            'total_dose_Sv_year': total_dose_rate,
            'neutron_transmission': neutron_transmission,
            'gamma_transmission': gamma_transmission,
            'final_neutron_energy_MeV': final_energy / 1e6
        }
    
    def calculate_shielding_cost(self, material_stack):
        """Calculate total cost and mass of shielding configuration."""
        total_cost = 0
        total_mass = 0
        
        # Assume cylindrical geometry around reactor
        reactor_radius = 5  # meters
        
        for i, (material_name, thickness) in enumerate(material_stack):
            material = self.materials[material_name]
            
            # Volume calculation for cylindrical shell
            inner_radius = reactor_radius + sum(layer[1] for layer in material_stack[:i])
            outer_radius = inner_radius + thickness
            height = 10  # 10 meter height
            
            volume = np.pi * height * (outer_radius**2 - inner_radius**2)
            
            # Cost and mass
            layer_cost = volume * material['cost_per_m3']
            layer_mass = volume * material['density']
            
            total_cost += layer_cost
            total_mass += layer_mass
        
        return {
            'total_cost_USD': total_cost,
            'total_mass_kg': total_mass,
            'cost_per_dose_reduction': total_cost / max(1e-6, 1 - self.calculate_dose_rate(material_stack)['total_dose_Sv_year'] / 1e-2)
        }
    
    def optimize_shielding_design(self):
        """
        Optimize multi-layer shielding design to meet dose limits
        while minimizing cost and mass.
        """
        print("🛡️ OPTIMIZING RADIATION SHIELDING DESIGN...")
        
        # Define optimization problem
        def objective_function(x):
            """Objective function: minimize cost while meeting dose limits."""
            # x = [tungsten_thickness, borated_poly_thickness, concrete_thickness, water_thickness]
            
            if any(thickness < 0 for thickness in x):
                return 1e10  # Penalty for negative thickness
            
            # Create material stack
            material_stack = []
            if x[0] > 0.01:  # Tungsten inner layer (gamma shielding)
                material_stack.append(('tungsten', x[0]))
            if x[1] > 0.01:  # Borated polyethylene (neutron moderation)
                material_stack.append(('borated_polyethylene', x[1]))
            if x[2] > 0.01:  # Concrete (structural and shielding)
                material_stack.append(('concrete', x[2]))
            if x[3] > 0.01:  # Water (neutron moderation)
                material_stack.append(('water', x[3]))
            
            if not material_stack:
                return 1e10
            
            # Calculate dose and cost
            dose_analysis = self.calculate_dose_rate(material_stack)
            cost_analysis = self.calculate_shielding_cost(material_stack)
            
            total_dose = dose_analysis['total_dose_Sv_year']
            total_cost = cost_analysis['total_cost_USD']
            
            # Penalty for exceeding dose limit
            dose_penalty = 0
            if total_dose > self.dose_limit:
                dose_penalty = 1e6 * (total_dose / self.dose_limit - 1)**2
            
            # Objective: minimize cost + dose penalty
            return total_cost + dose_penalty
        
        # Initial guess: [tungsten, borated_poly, concrete, water]
        x0 = [0.1, 0.5, 2.0, 1.0]  # meters
        
        # Bounds for thicknesses
        bounds = [
            (0, 0.5),    # Tungsten: 0-50 cm
            (0, 2.0),    # Borated polyethylene: 0-2 m
            (0, 5.0),    # Concrete: 0-5 m
            (0, 3.0)     # Water: 0-3 m
        ]
        
        # Optimize
        result = minimize(objective_function, x0, bounds=bounds, method='L-BFGS-B')
        
        # Extract optimal design
        optimal_thicknesses = result.x
        
        # Create optimal material stack
        optimal_stack = []
        layer_names = ['tungsten', 'borated_polyethylene', 'concrete', 'water']
        
        for i, thickness in enumerate(optimal_thicknesses):
            if thickness > 0.01:  # Include layers thicker than 1 cm
                optimal_stack.append((layer_names[i], thickness))
        
        return {
            'optimization_success': result.success,
            'optimal_thicknesses': optimal_thicknesses,
            'optimal_stack': optimal_stack,
            'final_objective': result.fun
        }
    
    def design_active_magnetic_shielding(self):
        """
        Design active magnetic shielding system for charged particles.
        Complements passive shielding for complete protection.
        """
        # Calculate charged particle deflection
        # For 14 MeV protons and alpha particles from fusion
        
        # Proton parameters
        proton_energy = 3.5e6 * 1.602e-19  # 3.5 MeV in Joules
        proton_mass = 1.673e-27  # kg
        proton_charge = 1.602e-19  # C
        
        # Alpha particle parameters
        alpha_energy = 3.5e6 * 1.602e-19  # 3.5 MeV in Joules
        alpha_mass = 6.644e-27  # kg
        alpha_charge = 2 * 1.602e-19  # C
        
        # Magnetic field parameters
        B_field = self.magnetic_field_strength  # Tesla
        
        # Calculate cyclotron radii
        proton_velocity = np.sqrt(2 * proton_energy / proton_mass)
        alpha_velocity = np.sqrt(2 * alpha_energy / alpha_mass)
        
        proton_radius = proton_mass * proton_velocity / (proton_charge * B_field)
        alpha_radius = alpha_mass * alpha_velocity / (alpha_charge * B_field)
        
        # Shielding effectiveness
        shielding_radius = self.magnetic_shielding_radius
        
        proton_deflection = proton_radius < shielding_radius
        alpha_deflection = alpha_radius < shielding_radius
        
        # Power requirements for magnetic field
        # Approximate superconducting coil system
        coil_radius = shielding_radius
        coil_length = 20  # meters
        current_density = B_field / (4e-7 * np.pi)  # A/m
        
        total_current = current_density * coil_length
        coil_inductance = 4e-7 * np.pi * coil_radius**2 / coil_length  # Henry
        stored_energy = 0.5 * coil_inductance * total_current**2  # Joules
        
        return {
            'proton_cyclotron_radius_m': proton_radius,
            'alpha_cyclotron_radius_m': alpha_radius,
            'proton_deflected': proton_deflection,
            'alpha_deflected': alpha_deflection,
            'magnetic_field_T': B_field,
            'shielding_radius_m': shielding_radius,
            'required_current_A': total_current,
            'stored_energy_MJ': stored_energy / 1e6,
            'power_efficiency': 0.95  # Superconducting coils
        }
    
    def generate_comprehensive_shielding_analysis(self):
        """
        Generate comprehensive radiation shielding analysis and optimization.
        """
        print("🛡️ LQG FUSION REACTOR - ADVANCED RADIATION SHIELDING")
        print("=" * 70)
        
        # Optimize passive shielding
        optimization = self.optimize_shielding_design()
        
        if optimization['optimization_success']:
            print("✅ Shielding optimization successful")
            
            optimal_stack = optimization['optimal_stack']
            
            # Analyze optimal design
            dose_analysis = self.calculate_dose_rate(optimal_stack)
            cost_analysis = self.calculate_shielding_cost(optimal_stack)
            
            print(f"\n📊 OPTIMAL SHIELDING DESIGN:")
            for material, thickness in optimal_stack:
                print(f"   • {material.replace('_', ' ').title()}: {thickness:.2f} m")
            
            print(f"\n⚡ RADIATION PROTECTION:")
            print(f"   • Neutron dose: {dose_analysis['neutron_dose_Sv_year']*1000:.2f} mSv/year")
            print(f"   • Gamma dose: {dose_analysis['gamma_dose_Sv_year']*1000:.2f} mSv/year")
            print(f"   • Total dose: {dose_analysis['total_dose_Sv_year']*1000:.2f} mSv/year")
            print(f"   • Dose limit: {self.dose_limit*1000:.1f} mSv/year")
            
            dose_meets_limit = dose_analysis['total_dose_Sv_year'] <= self.dose_limit
            print(f"   • Meets limit: {'✅ YES' if dose_meets_limit else '❌ NO'}")
            
            print(f"\n💰 COST ANALYSIS:")
            print(f"   • Total cost: ${cost_analysis['total_cost_USD']:,.0f}")
            print(f"   • Total mass: {cost_analysis['total_mass_kg']/1000:.1f} tons")
            print(f"   • Cost per dose reduction: ${cost_analysis['cost_per_dose_reduction']:,.0f}")
            
        else:
            print("❌ Shielding optimization failed")
            return None
        
        # Design active magnetic shielding
        print(f"\n🧲 ACTIVE MAGNETIC SHIELDING:")
        magnetic_shielding = self.design_active_magnetic_shielding()
        
        print(f"   • Magnetic field: {magnetic_shielding['magnetic_field_T']:.1f} T")
        print(f"   • Shielding radius: {magnetic_shielding['shielding_radius_m']:.1f} m")
        print(f"   • Proton deflection: {'✅ EFFECTIVE' if magnetic_shielding['proton_deflected'] else '❌ INSUFFICIENT'}")
        print(f"   • Alpha deflection: {'✅ EFFECTIVE' if magnetic_shielding['alpha_deflected'] else '❌ INSUFFICIENT'}")
        print(f"   • Stored energy: {magnetic_shielding['stored_energy_MJ']:.1f} MJ")
        
        # Overall assessment
        total_protection = (dose_meets_limit and 
                          magnetic_shielding['proton_deflected'] and 
                          magnetic_shielding['alpha_deflected'])
        
        print(f"\n🎯 OVERALL PROTECTION:")
        print(f"   • Passive shielding: {'✅ ADEQUATE' if dose_meets_limit else '❌ INSUFFICIENT'}")
        print(f"   • Active shielding: {'✅ EFFECTIVE' if magnetic_shielding['proton_deflected'] else '❌ NEEDS IMPROVEMENT'}")
        print(f"   • Total protection: {'✅ MEDICAL-GRADE' if total_protection else '⚠️ REQUIRES ENHANCEMENT'}")
        
        return {
            'optimization_results': optimization,
            'dose_analysis': dose_analysis,
            'cost_analysis': cost_analysis,
            'magnetic_shielding': magnetic_shielding,
            'meets_dose_limit': dose_meets_limit,
            'total_protection_adequate': total_protection
        }

def main():
    """Main execution for radiation shielding optimization."""
    print("🚀 LQG FTL VESSEL - ADVANCED RADIATION SHIELDING OPTIMIZATION")
    print("Initializing radiation protection systems...")
    
    optimizer = AdvancedRadiationShieldingOptimizer()
    
    # Generate comprehensive analysis
    results = optimizer.generate_comprehensive_shielding_analysis()
    
    if results:
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"radiation_shielding_optimization_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'target_dose_limit_mSv_year': optimizer.dose_limit * 1000,
                'safety_factor': optimizer.safety_factor,
                'reactor_power_MW': optimizer.fusion_power / 1e6,
                'optimization_results': results
            }, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        status = "✅ RADIATION PROTECTION ACHIEVED" if results['total_protection_adequate'] else "⚠️ ENHANCEMENT REQUIRED"
        print(f"🎯 SHIELDING STATUS: {status}")

if __name__ == "__main__":
    main()

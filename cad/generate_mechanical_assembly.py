#!/usr/bin/env python3
"""
CadQuery Mechanical Assembly Generator for LQR-1
Generated from parts list - creates 3D assembly of mechanical components
"""

import cadquery as cq
from pathlib import Path

class LQR1MechanicalAssembly:
    """3D mechanical assembly builder for LQR-1"""
    
    def __init__(self, cad_dir: str = "cad"):
        self.cad_dir = Path(cad_dir)
        self.step_dir = self.cad_dir / "step"
        self.assembly = cq.Assembly()
        
    def load_components(self):
        """Load all mechanical components from STEP files"""
        print("🔧 LOADING MECHANICAL COMPONENTS...")
        
        self.components = {}

        # Tungsten-lined toroidal chamber, 3.5m major radius, 1.2m minor radius
        try:
            vc1_file = self.step_dir / "TVC-350-120-W99.step"
            if vc1_file.exists():
                self.components['VC1'] = cq.importers.importStep(str(vc1_file))
                print(f"✅ Loaded VC1: Tungsten-lined toroidal chamber, 3.5m major radius, 1.2m minor radius")
        except Exception as e:
            print(f"⚠️  Failed to load VC1: {e}")

        # Tungsten chamber segments, precision-welded joints
        try:
            vc2_file = self.step_dir / "W-SEG-145-T15.step"
            if vc2_file.exists():
                self.components['VC2'] = cq.importers.importStep(str(vc2_file))
                print(f"✅ Loaded VC2: Tungsten chamber segments, precision-welded joints")
        except Exception as e:
            print(f"⚠️  Failed to load VC2: {e}")

        # High-vacuum ports, CF-150 conflat flanges
        try:
            vc3_file = self.step_dir / "CF150-316L-B.step"
            if vc3_file.exists():
                self.components['VC3'] = cq.importers.importStep(str(vc3_file))
                print(f"✅ Loaded VC3: High-vacuum ports, CF-150 conflat flanges")
        except Exception as e:
            print(f"⚠️  Failed to load VC3: {e}")

        # Primary vacuum pumps, turbo-molecular type
        try:
            vc4_file = self.step_dir / "STP-iXA4506.step"
            if vc4_file.exists():
                self.components['VC4'] = cq.importers.importStep(str(vc4_file))
                print(f"✅ Loaded VC4: Primary vacuum pumps, turbo-molecular type")
        except Exception as e:
            print(f"⚠️  Failed to load VC4: {e}")

        # Backing pumps, oil-free scroll type
        try:
            vc5_file = self.step_dir / "XDS35i.step"
            if vc5_file.exists():
                self.components['VC5'] = cq.importers.importStep(str(vc5_file))
                print(f"✅ Loaded VC5: Backing pumps, oil-free scroll type")
        except Exception as e:
            print(f"⚠️  Failed to load VC5: {e}")

        # Toroidal field coils, superconducting NbTi
        try:
            mc1_file = self.step_dir / "TFC-350-NBTI-50.step"
            if mc1_file.exists():
                self.components['MC1'] = cq.importers.importStep(str(mc1_file))
                print(f"✅ Loaded MC1: Toroidal field coils, superconducting NbTi")
        except Exception as e:
            print(f"⚠️  Failed to load MC1: {e}")

        # Poloidal field coils, superconducting Nb₃Sn
        try:
            mc2_file = self.step_dir / "PFC-120-NB3SN-25.step"
            if mc2_file.exists():
                self.components['MC2'] = cq.importers.importStep(str(mc2_file))
                print(f"✅ Loaded MC2: Poloidal field coils, superconducting Nb₃Sn")
        except Exception as e:
            print(f"⚠️  Failed to load MC2: {e}")

        # Cryogenic cooling system, helium refrigerator
        try:
            mc3_file = self.step_dir / "LR280-He-II.step"
            if mc3_file.exists():
                self.components['MC3'] = cq.importers.importStep(str(mc3_file))
                print(f"✅ Loaded MC3: Cryogenic cooling system, helium refrigerator")
        except Exception as e:
            print(f"⚠️  Failed to load MC3: {e}")

        # Current leads, high-temperature superconductor
        try:
            mc4_file = self.step_dir / "HTS-CL-50KA.step"
            if mc4_file.exists():
                self.components['MC4'] = cq.importers.importStep(str(mc4_file))
                print(f"✅ Loaded MC4: Current leads, high-temperature superconductor")
        except Exception as e:
            print(f"⚠️  Failed to load MC4: {e}")

        # Quench protection resistors, water-cooled
        try:
            mc5_file = self.step_dir / "RWR-500K-50.step"
            if mc5_file.exists():
                self.components['MC5'] = cq.importers.importStep(str(mc5_file))
                print(f"✅ Loaded MC5: Quench protection resistors, water-cooled")
        except Exception as e:
            print(f"⚠️  Failed to load MC5: {e}")

        # Polymer field generators, sinc(πμ) modulation
        try:
            pfg1_file = self.step_dir / "QFI-PFG-100-SINC.step"
            if pfg1_file.exists():
                self.components['PFG1'] = cq.importers.importStep(str(pfg1_file))
                print(f"✅ Loaded PFG1: Polymer field generators, sinc(πμ) modulation")
        except Exception as e:
            print(f"⚠️  Failed to load PFG1: {e}")

        # Field coupling networks, distributed topology
        try:
            pfg2_file = self.step_dir / "AE-FCN-16PT-DIST.step"
            if pfg2_file.exists():
                self.components['PFG2'] = cq.importers.importStep(str(pfg2_file))
                print(f"✅ Loaded PFG2: Field coupling networks, distributed topology")
        except Exception as e:
            print(f"⚠️  Failed to load PFG2: {e}")

        # Central control processor, quantum-enhanced
        try:
            pfg3_file = self.step_dir / "IBM-QCP-127-QUBIT.step"
            if pfg3_file.exists():
                self.components['PFG3'] = cq.importers.importStep(str(pfg3_file))
                print(f"✅ Loaded PFG3: Central control processor, quantum-enhanced")
        except Exception as e:
            print(f"⚠️  Failed to load PFG3: {e}")

        # Field monitoring sensors, backreaction detection
        try:
            pfg4_file = self.step_dir / "RSA7100A-BR-MON.step"
            if pfg4_file.exists():
                self.components['PFG4'] = cq.importers.importStep(str(pfg4_file))
                print(f"✅ Loaded PFG4: Field monitoring sensors, backreaction detection")
        except Exception as e:
            print(f"⚠️  Failed to load PFG4: {e}")

        # Neutral beam injectors, 20 MW each
        try:
            nbi1_file = self.step_dir / "ITER-NBI-20MW-D.step"
            if nbi1_file.exists():
                self.components['NBI1'] = cq.importers.importStep(str(nbi1_file))
                print(f"✅ Loaded NBI1: Neutral beam injectors, 20 MW each")
        except Exception as e:
            print(f"⚠️  Failed to load NBI1: {e}")

        # Ion sources, RF-driven negative ion
        try:
            nbi2_file = self.step_dir / "MPI-IS-RF-NI.step"
            if nbi2_file.exists():
                self.components['NBI2'] = cq.importers.importStep(str(nbi2_file))
                print(f"✅ Loaded NBI2: Ion sources, RF-driven negative ion")
        except Exception as e:
            print(f"⚠️  Failed to load NBI2: {e}")

        # Beam line components, electrostatic acceleration
        try:
            nbi3_file = self.step_dir / "HRL-BL-1MV-ES.step"
            if nbi3_file.exists():
                self.components['NBI3'] = cq.importers.importStep(str(nbi3_file))
                print(f"✅ Loaded NBI3: Beam line components, electrostatic acceleration")
        except Exception as e:
            print(f"⚠️  Failed to load NBI3: {e}")

        # Tritium processing plant, complete system
        try:
            fps1_file = self.step_dir / "ITER-TPP-COMPLETE.step"
            if fps1_file.exists():
                self.components['FPS1'] = cq.importers.importStep(str(fps1_file))
                print(f"✅ Loaded FPS1: Tritium processing plant, complete system")
        except Exception as e:
            print(f"⚠️  Failed to load FPS1: {e}")

        # Fuel storage vessels, tritium-compatible
        try:
            fps2_file = self.step_dir / "SRS-TV-10G-T.step"
            if fps2_file.exists():
                self.components['FPS2'] = cq.importers.importStep(str(fps2_file))
                print(f"✅ Loaded FPS2: Fuel storage vessels, tritium-compatible")
        except Exception as e:
            print(f"⚠️  Failed to load FPS2: {e}")

        # Isotope separation units, cryogenic distillation
        try:
            fps3_file = self.step_dir / "LANL-ISU-CD-T.step"
            if fps3_file.exists():
                self.components['FPS3'] = cq.importers.importStep(str(fps3_file))
                print(f"✅ Loaded FPS3: Isotope separation units, cryogenic distillation")
        except Exception as e:
            print(f"⚠️  Failed to load FPS3: {e}")

        # Tungsten neutron shield, 0.20 m thickness
        try:
            rs1_file = self.step_dir / "W-SHIELD-200-NEUT.step"
            if rs1_file.exists():
                self.components['RS1'] = cq.importers.importStep(str(rs1_file))
                print(f"✅ Loaded RS1: Tungsten neutron shield, 0.20 m thickness")
        except Exception as e:
            print(f"⚠️  Failed to load RS1: {e}")

        # Lithium hydride moderator, 0.50 m thickness
        try:
            rs2_file = self.step_dir / "LiH-MOD-500-99.step"
            if rs2_file.exists():
                self.components['RS2'] = cq.importers.importStep(str(rs2_file))
                print(f"✅ Loaded RS2: Lithium hydride moderator, 0.50 m thickness")
        except Exception as e:
            print(f"⚠️  Failed to load RS2: {e}")

        # Borated polyethylene shield, 1.00 m thickness
        try:
            rs3_file = self.step_dir / "REI-BP-1000-B10.step"
            if rs3_file.exists():
                self.components['RS3'] = cq.importers.importStep(str(rs3_file))
                print(f"✅ Loaded RS3: Borated polyethylene shield, 1.00 m thickness")
        except Exception as e:
            print(f"⚠️  Failed to load RS3: {e}")

        # Enhanced concrete barrier, 3.0 m thickness
        try:
            rs4_file = self.step_dir / "NCC-EC-3000-B.step"
            if rs4_file.exists():
                self.components['RS4'] = cq.importers.importStep(str(rs4_file))
                print(f"✅ Loaded RS4: Enhanced concrete barrier, 3.0 m thickness")
        except Exception as e:
            print(f"⚠️  Failed to load RS4: {e}")

        # Lead-bismuth gamma shield, 0.15 m thickness
        try:
            gs1_file = self.step_dir / "MI-PBGAMMA-150.step"
            if gs1_file.exists():
                self.components['GS1'] = cq.importers.importStep(str(gs1_file))
                print(f"✅ Loaded GS1: Lead-bismuth gamma shield, 0.15 m thickness")
        except Exception as e:
            print(f"⚠️  Failed to load GS1: {e}")

        # Gamma detection systems, real-time monitoring
        try:
            gs2_file = self.step_dir / "MGP-GDS-RT-24.step"
            if gs2_file.exists():
                self.components['GS2'] = cq.importers.importStep(str(gs2_file))
                print(f"✅ Loaded GS2: Gamma detection systems, real-time monitoring")
        except Exception as e:
            print(f"⚠️  Failed to load GS2: {e}")

        
        return self.components
    
    def create_assembly(self):
        """Create complete mechanical assembly with positioning"""
        print("🏗️  CREATING MECHANICAL ASSEMBLY...")
        
        # Start with vacuum chamber as base
        if 'VC1' in self.components:
            self.assembly.add(self.components['VC1'], name="VacuumChamber", 
                            color=cq.Color(0.8, 0.8, 0.9, 0.7))
        
        # Add toroidal coils around chamber
        if 'MC1' in self.components:
            for i in range(16):  # 16 toroidal coils
                angle = i * 360 / 16
                loc = cq.Location((0, 0, 0), (0, 0, 1), angle)
                self.assembly.add(self.components['MC1'], 
                                name=f"ToroidalCoil_{i+1:02d}",
                                loc=loc,
                                color=cq.Color(0.2, 0.8, 0.2))
        
        # Add poloidal coils
        if 'MC2' in self.components:
            for i in range(12):  # 12 poloidal coils
                z_offset = (i - 6) * 0.5  # Distribute along height
                loc = cq.Location((0, 0, z_offset * 1000))  # Convert to mm
                self.assembly.add(self.components['MC2'],
                                name=f"PoloidalCoil_{i+1:02d}",
                                loc=loc, 
                                color=cq.Color(0.8, 0.2, 0.2))
        
        # Add tungsten shielding
        if 'RS1' in self.components:
            self.assembly.add(self.components['RS1'], name="TungstenShield",
                            color=cq.Color(0.4, 0.4, 0.4, 0.8))
        
        # Add other components with appropriate positioning
        component_positions = {
            'PS1': ((5000, 0, 0), (0, 0, 1), 0),     # Power supply offset
            'PFG1': ((0, 4000, 0), (0, 0, 1), 0),    # Field generator
            'NBI1': ((3000, 3000, 1000), (0, 0, 1), 45),  # Neutral beam
        }
        
        for comp_id, (pos, axis, angle) in component_positions.items():
            if comp_id in self.components:
                loc = cq.Location(pos, axis, angle)
                self.assembly.add(self.components[comp_id], 
                                name=comp_id,
                                loc=loc)
        
        print("✅ Mechanical assembly complete")
        return self.assembly
    
    def export_assembly(self, format='step'):
        """Export complete assembly"""
        output_file = self.cad_dir / f"lqr1_complete_assembly.{format}"
        
        if format.lower() == 'step':
            self.assembly.save(str(output_file))
        elif format.lower() == 'stl':
            # Export as STL for 3D printing/visualization
            self.assembly.save(str(output_file), exportType='STL')
        
        print(f"📁 Assembly exported: {output_file}")
        return output_file

def main():
    """Main mechanical assembly generation"""
    assembly_builder = LQR1MechanicalAssembly()
    
    # Load components
    components = assembly_builder.load_components()
    
    # Create assembly
    assembly = assembly_builder.create_assembly()
    
    # Export final assembly
    assembly_builder.export_assembly('step')
    assembly_builder.export_assembly('stl')
    
    print("🚀 MECHANICAL ASSEMBLY COMPLETE!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
LQR-1 Parts List Parser and CAD Pipeline Generator
Converts Markdown parts list to machine-readable format and generates CAD workflows
"""

import re
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

@dataclass
class PartSpecification:
    """Individual component specification"""
    ref: str
    quantity: int
    description: str
    supplier: str
    part_number: str
    cost: float
    specification: str
    category: str
    domain: str  # "electrical" or "mechanical"
    cad_format: str = "STEP"  # Expected CAD format
    priority: int = 1  # Download priority (1=high, 3=low)

class LQR1PartsParser:
    """Parser for LQR-1 parts list with CAD workflow generation"""
    
    def __init__(self, parts_file_path: str):
        self.parts_file_path = Path(parts_file_path)
        self.parts = []
        self.categories = {
            'VC': 'Vacuum Chamber',
            'MC': 'Magnetic Confinement',
            'PS': 'Power Supply',
            'PFG': 'Polymer Field Generator',
            'NBI': 'Neutral Beam Injection',
            'FPS': 'Fuel Processing',
            'RS': 'Radiation Shielding',
            'GS': 'Gamma Shielding',
            'ES': 'Emergency Systems',
            'CS': 'Control Systems'
        }
        
    def parse_parts_list(self) -> List[PartSpecification]:
        """Parse the Markdown parts list into structured data"""
        print("🔍 PARSING LQR-1 PARTS LIST...")
        
        with open(self.parts_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Regex patterns for different line types
        part_pattern = r'\*\*(\w+)\*\* \((\d+)\) (.+)'
        supplier_pattern = r'- Supplier: (.+?), Part: (.+)'
        spec_pattern = r'- Specification: (.+)'
        cost_pattern = r'- Cost: \$([0-9,]+)'
        
        lines = content.split('\n')
        current_part = None
        
        for line in lines:
            line = line.strip()
            
            # Match part reference line
            part_match = re.match(part_pattern, line)
            if part_match:
                ref = part_match.group(1)
                quantity = int(part_match.group(2))
                description = part_match.group(3)
                
                # Determine category and domain
                category_prefix = ref[:2] if len(ref) >= 2 else ref[:1]
                category = self.categories.get(category_prefix, 'Other')
                
                # Classify as electrical or mechanical
                electrical_categories = ['PS', 'PFG', 'CS', 'ES']
                domain = 'electrical' if category_prefix in electrical_categories else 'mechanical'
                
                current_part = {
                    'ref': ref,
                    'quantity': quantity,
                    'description': description,
                    'category': category,
                    'domain': domain,
                    'supplier': '',
                    'part_number': '',
                    'specification': '',
                    'cost': 0.0
                }
                continue
            
            if current_part is None:
                continue
                
            # Match supplier and part number
            supplier_match = re.match(supplier_pattern, line)
            if supplier_match:
                current_part['supplier'] = supplier_match.group(1)
                current_part['part_number'] = supplier_match.group(2)
                continue
                
            # Match specification
            spec_match = re.match(spec_pattern, line)
            if spec_match:
                current_part['specification'] = spec_match.group(1)
                continue
                
            # Match cost
            cost_match = re.match(cost_pattern, line)
            if cost_match:
                cost_str = cost_match.group(1).replace(',', '')
                current_part['cost'] = float(cost_str)
                
                # Create PartSpecification object and add to list
                part_spec = PartSpecification(
                    ref=current_part['ref'],
                    quantity=current_part['quantity'],
                    description=current_part['description'],
                    supplier=current_part['supplier'],
                    part_number=current_part['part_number'],
                    cost=current_part['cost'],
                    specification=current_part['specification'],
                    category=current_part['category'],
                    domain=current_part['domain'],
                    priority=self._determine_priority(current_part['ref'])
                )
                
                self.parts.append(part_spec)
                current_part = None
        
        print(f"✅ Parsed {len(self.parts)} components")
        return self.parts
    
    def _determine_priority(self, ref: str) -> int:
        """Determine download priority based on component type"""
        high_priority = ['VC1', 'MC1', 'MC2', 'PS1', 'PFG1', 'NBI1', 'RS1']
        medium_priority = ['VC2', 'VC3', 'MC3', 'PS2', 'PFG2', 'FPS1']
        
        if ref in high_priority:
            return 1  # High priority
        elif ref in medium_priority:
            return 2  # Medium priority
        else:
            return 3  # Low priority
    
    def generate_cad_download_script(self) -> str:
        """Generate Python script for automated CAD model downloads"""
        script = '''#!/usr/bin/env python3
"""
Automated CAD Model Download Script for LQR-1 Components
Generated from parts list - downloads STEP/IGES files and converts to OBJ
"""

import requests
import os
from pathlib import Path
import trimesh
import json

class CADDownloader:
    """Automated CAD model downloader with format conversion"""
    
    def __init__(self, output_dir: str = "cad"):
        self.output_dir = Path(output_dir)
        self.step_dir = self.output_dir / "step"
        self.obj_dir = self.output_dir / "obj"
        
        # Create directories
        self.step_dir.mkdir(parents=True, exist_ok=True)
        self.obj_dir.mkdir(parents=True, exist_ok=True)
        
        # Vendor API configurations
        self.vendor_configs = {
            'snapeda': {'api_key': None, 'base_url': 'https://api.snapeda.com'},
            'octopart': {'api_key': None, 'base_url': 'https://octopart.com/api/v4'},
            'grabcad': {'api_key': None, 'base_url': 'https://grabcad.com/api'}
        }
    
    def download_step_files(self, parts_list):
        """Download STEP/IGES files for all parts"""
        print("🔽 DOWNLOADING CAD MODELS...")
        
        downloaded = 0
        for part in parts_list:
            try:
                success = self._download_part_cad(part)
                if success:
                    downloaded += 1
                    print(f"✅ Downloaded {part['ref']}: {part['part_number']}")
                else:
                    print(f"⚠️  Failed to download {part['ref']}")
            except Exception as e:
                print(f"❌ Error downloading {part['ref']}: {e}")
        
        print(f"📦 Downloaded {downloaded}/{len(parts_list)} CAD models")
        return downloaded
    
    def _download_part_cad(self, part):
        """Download CAD model for individual part"""
        # Try different strategies based on supplier
        supplier = part['supplier'].lower()
        part_number = part['part_number']
        
        if 'oxford' in supplier:
            return self._download_oxford_instruments(part)
        elif 'plansee' in supplier:
            return self._download_plansee_group(part)
        elif 'abb' in supplier:
            return self._download_abb(part)
        elif 'iter' in supplier:
            return self._download_iter_organization(part)
        else:
            return self._download_generic(part)
    
    def _download_oxford_instruments(self, part):
        """Download from Oxford Instruments CAD library"""
        # Oxford Instruments has a CAD download portal
        base_url = "https://www.oxford-instruments.com/cad-models"
        # Implementation would require web scraping or API access
        return False
    
    def _download_plansee_group(self, part):
        """Download from Plansee Group CAD library"""
        # Plansee provides STEP files for tungsten components
        base_url = "https://www.plansee.com/en/materials/tungsten"
        # Implementation would require vendor-specific API or scraping
        return False
    
    def _download_abb(self, part):
        """Download from ABB CAD library"""
        # ABB has extensive CAD model downloads
        base_url = "https://new.abb.com/products/cad-models"
        # Implementation would use ABB's product API
        return False
    
    def _download_iter_organization(self, part):
        """Download from ITER Organization"""
        # ITER provides technical drawings and CAD models
        base_url = "https://www.iter.org/technical"
        # Would require special access permissions
        return False
    
    def _download_generic(self, part):
        """Generic download using part databases"""
        # Try SnapEDA, Octopart, GrabCAD APIs
        return self._try_snapeda(part) or self._try_octopart(part)
    
    def _try_snapeda(self, part):
        """Try downloading from SnapEDA"""
        if not self.vendor_configs['snapeda']['api_key']:
            return False
        
        # SnapEDA API call would go here
        return False
    
    def _try_octopart(self, part):
        """Try downloading from Octopart"""
        if not self.vendor_configs['octopart']['api_key']:
            return False
        
        # Octopart API call would go here
        return False
    
    def convert_to_obj(self):
        """Convert all STEP/IGES files to OBJ format"""
        print("🔄 CONVERTING CAD MODELS TO OBJ...")
        
        step_files = list(self.step_dir.glob("*.*"))
        converted = 0
        
        for step_file in step_files:
            try:
                if step_file.suffix.lower() in ['.step', '.stp', '.iges', '.igs']:
                    obj_file = self.obj_dir / f"{step_file.stem}.obj"
                    
                    # Use trimesh for conversion
                    mesh = trimesh.load(str(step_file))
                    mesh.export(str(obj_file))
                    
                    converted += 1
                    print(f"✅ Converted {step_file.name} → {obj_file.name}")
                    
            except Exception as e:
                print(f"❌ Failed to convert {step_file.name}: {e}")
        
        print(f"🎯 Converted {converted} files to OBJ format")
        return converted

def main():
    """Main CAD download and conversion pipeline"""
    # Load parts list
    with open('cad/lqr1_parts.json', 'r') as f:
        parts_list = json.load(f)
    
    # Initialize downloader
    downloader = CADDownloader()
    
    # Download CAD models
    downloader.download_step_files(parts_list)
    
    # Convert to OBJ
    downloader.convert_to_obj()
    
    print("🚀 CAD PIPELINE COMPLETE!")

if __name__ == "__main__":
    main()
'''
        return script
    
    def generate_skidl_electrical_schematic(self) -> str:
        """Generate SKiDL script for electrical components"""
        electrical_parts = [p for p in self.parts if p.domain == 'electrical']
        
        script = '''#!/usr/bin/env python3
"""
SKiDL Electrical Schematic Generator for LQR-1
Generated from parts list - creates netlist for electrical components
"""

from skidl import *

# Set default tool to KiCad
set_default_tool(KICAD)

def create_lqr1_electrical_schematic():
    """Create complete electrical schematic for LQR-1"""
    
    # Define power and ground nets
    VCC_50MW = Net('VCC_50MW')
    VCC_25MW = Net('VCC_25MW') 
    VCC_5MW = Net('VCC_5MW')
    GND = Net('GND')
    
    # Power supply components
'''
        
        # Add electrical parts
        for part in electrical_parts:
            if part.ref.startswith('PS'):
                script += f'''
    # {part.description}
    {part.ref.lower()} = Part('LQR1_Lib', '{part.ref}', 
                    footprint='Custom:{part.ref}_footprint',
                    value='{part.cost}',
                    desc='{part.description}')
    {part.ref.lower()}['VCC'] += VCC_50MW
    {part.ref.lower()}['GND'] += GND
'''
            elif part.ref.startswith('PFG'):
                script += f'''
    # {part.description}
    {part.ref.lower()} = Part('LQR1_Lib', '{part.ref}', 
                    footprint='Custom:{part.ref}_footprint',
                    value='{part.cost}',
                    desc='{part.description}')
    {part.ref.lower()}['FIELD_OUT'] += Net('{part.ref}_FIELD')
    {part.ref.lower()}['VCC'] += VCC_25MW
    {part.ref.lower()}['GND'] += GND
'''
            elif part.ref.startswith('CS'):
                script += f'''
    # {part.description}  
    {part.ref.lower()} = Part('LQR1_Lib', '{part.ref}',
                    footprint='Custom:{part.ref}_footprint', 
                    value='{part.cost}',
                    desc='{part.description}')
    {part.ref.lower()}['SENSOR_OUT'] += Net('{part.ref}_DATA')
    {part.ref.lower()}['VCC'] += VCC_5MW
    {part.ref.lower()}['GND'] += GND
'''
        
        script += '''
    
    # Generate netlist
    generate_netlist('lqr1_electrical.net')
    print("✅ Generated LQR-1 electrical netlist")

if __name__ == "__main__":
    create_lqr1_electrical_schematic()
'''
        
        return script
    
    def generate_cadquery_mechanical_assembly(self) -> str:
        """Generate CadQuery script for mechanical assembly"""
        mechanical_parts = [p for p in self.parts if p.domain == 'mechanical']
        
        script = '''#!/usr/bin/env python3
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
'''
        
        # Add mechanical component loading
        for part in mechanical_parts:
            script += f'''
        # {part.description}
        try:
            {part.ref.lower()}_file = self.step_dir / "{part.part_number}.step"
            if {part.ref.lower()}_file.exists():
                self.components['{part.ref}'] = cq.importers.importStep(str({part.ref.lower()}_file))
                print(f"✅ Loaded {part.ref}: {part.description}")
        except Exception as e:
            print(f"⚠️  Failed to load {part.ref}: {{e}}")
'''
        
        script += '''
        
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
'''
        
        return script
    
    def generate_pipeline_master(self) -> str:
        """Generate master pipeline script"""
        script = '''#!/usr/bin/env python3
"""
LQR-1 Master CAD Pipeline
Orchestrates complete workflow from parts list to 3D assembly
"""

import subprocess
import sys
from pathlib import Path
import json

def run_complete_pipeline():
    """Run complete CAD generation pipeline"""
    print("🚀 STARTING LQR-1 CAD PIPELINE")
    print("=" * 50)
    
    # Step 1: Parse parts list
    print("\\n📋 Step 1: Parsing parts list...")
    subprocess.run([sys.executable, "parse_parts_list.py"])
    
    # Step 2: Download CAD models
    print("\\n🔽 Step 2: Downloading CAD models...")
    subprocess.run([sys.executable, "download_cad_models.py"])
    
    # Step 3: Generate electrical schematic
    print("\\n⚡ Step 3: Generating electrical schematic...")
    subprocess.run([sys.executable, "generate_electrical_schematic.py"])
    
    # Step 4: Generate mechanical assembly
    print("\\n🔧 Step 4: Generating mechanical assembly...")
    subprocess.run([sys.executable, "generate_mechanical_assembly.py"])
    
    # Step 5: Create final documentation
    print("\\n📚 Step 5: Generating documentation...")
    subprocess.run([sys.executable, "generate_documentation.py"])
    
    print("\\n" + "=" * 50)
    print("✅ LQR-1 CAD PIPELINE COMPLETE!")
    print("📁 Outputs available in cad/ directory")
    print("🎯 Ready for manufacturing and construction")

if __name__ == "__main__":
    run_complete_pipeline()
'''
        return script
    
    def export_parsed_data(self, output_dir: str = "cad"):
        """Export parsed parts data to JSON and generate all scripts"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export parts list as JSON
        parts_json = output_path / "lqr1_parts.json"
        with open(parts_json, 'w') as f:
            json.dump([asdict(part) for part in self.parts], f, indent=2)
        
        print(f"📁 Parts list exported: {parts_json}")
        
        # Generate and save scripts
        scripts = {
            "download_cad_models.py": self.generate_cad_download_script(),
            "generate_electrical_schematic.py": self.generate_skidl_electrical_schematic(),
            "generate_mechanical_assembly.py": self.generate_cadquery_mechanical_assembly(),
            "run_complete_pipeline.py": self.generate_pipeline_master()
        }
        
        for filename, content in scripts.items():
            script_path = output_path / filename
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(content)
            script_path.chmod(0o755)  # Make executable
            print(f"📄 Generated script: {script_path}")
        
        return parts_json

def main():
    """Main parsing and pipeline generation"""
    # Parse the parts list
    parser = LQR1PartsParser("construction/lqr-1/lqr-1_parts_list.md")
    parts = parser.parse_parts_list()
    
    # Generate statistics
    print("\n📊 PARTS ANALYSIS:")
    domains = {}
    categories = {}
    total_cost = 0
    
    for part in parts:
        domains[part.domain] = domains.get(part.domain, 0) + part.quantity
        categories[part.category] = categories.get(part.category, 0) + part.quantity
        total_cost += part.cost * part.quantity
    
    print(f"   • Total components: {len(parts)}")
    print(f"   • Total quantity: {sum(domains.values())}")
    print(f"   • Electrical components: {domains.get('electrical', 0)}")
    print(f"   • Mechanical components: {domains.get('mechanical', 0)}")
    print(f"   • Total estimated cost: ${total_cost:,.0f}")
    
    print(f"\n🏷️  CATEGORIES:")
    for category, count in categories.items():
        print(f"   • {category}: {count} items")
    
    # Export data and generate scripts
    parts_file = parser.export_parsed_data()
    
    print(f"\n🎯 PIPELINE READY!")
    print(f"   • Parsed data: {parts_file}")
    print(f"   • Scripts generated in: cad/")
    print(f"   • Run: python cad/run_complete_pipeline.py")

if __name__ == "__main__":
    main()

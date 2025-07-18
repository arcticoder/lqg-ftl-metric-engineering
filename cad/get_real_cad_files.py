#!/usr/bin/env python3
"""
Real CAD Model Download Script
Practical guide to obtaining actual STEP/IGES files from suppliers
"""

import webbrowser
import json
from pathlib import Path

def open_supplier_websites():
    """Open supplier websites for CAD model downloads"""
    print("🌐 OPENING SUPPLIER CAD DOWNLOAD WEBSITES...")
    
    # Load parts to get supplier list
    with open('cad/lqr1_parts.json', 'r') as f:
        parts = json.load(f)
    
    # Major suppliers with direct CAD access
    supplier_urls = {
        'Oxford Instruments': 'https://www.oxford-instruments.com/cad-models',
        'Plansee Group': 'https://www.plansee.com/en/downloads/cad-data',
        'ABB': 'https://new.abb.com/products/cad-models',
        'Leybold': 'https://www.leybold.com/configurator',
        'Edwards Vacuum': 'https://www.edwardsvacuum.com/cad-models',
        'Siemens': 'https://www.siemens.com/cad-models',
        'Schneider Electric': 'https://www.se.com/cad-models'
    }
    
    # Industrial CAD databases
    industrial_urls = {
        'SnapEDA': 'https://www.snapeda.com/',
        'TraceParts': 'https://www.traceparts.com/',
        'GrabCAD': 'https://grabcad.com/library',
        '3D ContentCentral': 'https://www.3dcontentcentral.com/',
        'Octopart': 'https://octopart.com/'
    }
    
    print("\n🏭 DIRECT SUPPLIER WEBSITES:")
    for supplier, url in supplier_urls.items():
        print(f"   • {supplier}: {url}")
    
    print("\n🏪 INDUSTRIAL CAD DATABASES:")
    for service, url in industrial_urls.items():
        print(f"   • {service}: {url}")
    
    # Open key websites
    print("\n🚀 Opening key websites in browser...")
    try:
        webbrowser.open('https://www.snapeda.com/')
        webbrowser.open('https://www.traceparts.com/')
        webbrowser.open('https://grabcad.com/library')
        print("   ✅ Opened SnapEDA, TraceParts, and GrabCAD")
    except Exception as e:
        print(f"   ⚠️ Browser opening failed: {e}")

def generate_download_checklist():
    """Generate a checklist for downloading real CAD files"""
    checklist = """
# 📋 CAD Model Download Checklist for LQR-1 Components

## 🎯 Priority Components (Download First)
- [ ] **VC1** (TVC-350-120-W99) - Vacuum Chamber Assembly - $2.85M
  - ⚠️ CUSTOM COMPONENT - No standard CAD available
  - Alternative Search: "stainless steel vacuum chamber" OR "torus vacuum vessel"
  - Real Options: KF/CF flanged chambers, spherical vessels, custom welded assemblies
  - Suppliers: Kurt J. Lesker, Nor-Cal Products, MDC Vacuum
  
- [ ] **MC1** (TFC-350-NBTI-50) - Toroidal Field Coils - $485K each
  - Supplier: Oxford Instruments
  - Method: Register at oxford-instruments.com/cad-models
  - Expected: STEP file with superconducting coil assembly
  
- [ ] **PS1** (SACE-THYRO-50MW) - Main Power Converter - $3.2M
  - Supplier: ABB
  - Method: Search ABB CAD portal by part number
  - Expected: STEP file with enclosure and connections
  
- [ ] **RS1** (W-SHIELD-200-NEUT) - Tungsten Shield - $8.5M
  - Supplier: Plansee Group
  - Method: Contact for custom tungsten shielding
  - Expected: STEP file with shield geometry

## 🔍 Search Strategies by Component Type

### Vacuum Components (VC1-VC5)
- **Realistic Search Terms**: 
  - "CF150 vacuum chamber" (ConFlat flanges)
  - "KF40 tee chamber" (Klein flanges)  
  - "spherical vacuum vessel"
  - "cylindrical vacuum chamber"
- **Real Suppliers**: 
  - Kurt J. Lesker: https://www.lesker.com/vacuum-chambers
  - Nor-Cal Products: https://www.n-c.com/vacuum-chambers
  - MDC Vacuum: https://www.mdcvacuum.com/chambers
- **CAD Availability**: Standard chambers have STEP files, custom requires quotes

### Magnetic Components (MC1-MC5)
- **Primary**: Oxford Instruments CAD library
- **Secondary**: Bruker EAS technical support
- **Tertiary**: ITER Organization technical database

### Power Components (PS1-PS4)
- **Primary**: ABB CAD portal
- **Secondary**: Schneider Electric CAD library
- **Tertiary**: Danfysik direct contact

### Control Components (CS1-CS3)
- **Primary**: Siemens CAD models
- **Secondary**: Tektronix product pages
- **Tertiary**: SnapEDA for electronics

## 📞 Contact Information
- **Materials Research Corporation**: Contact via website for custom quotes
- **Oxford Instruments**: CAD library requires registration
- **ABB**: Part search available on product pages
- **Plansee Group**: Technical support for tungsten components
- **ITER Organization**: Research collaboration required

## 🔄 Conversion Pipeline
1. Download STEP/IGES files → `cad/step/`
2. Install: `pip install trimesh[easy] FreeCAD`
3. Run: `python convert_real_cad_models.py`
4. Output OBJ files → `cad/obj/`
5. Integrate with LQG Circuit DSL framework

## ✅ Quality Checklist
- [ ] File opens in FreeCAD without errors
- [ ] Dimensions match datasheet specifications
- [ ] All mounting points and interfaces included
- [ ] Material properties specified in metadata
- [ ] Coordinate system matches expected orientation

## 🎯 Success Metrics
- **Target**: 36 components with CAD models
- **Priority**: 15 high-value components first
- **Timeline**: 2-4 weeks for complete collection
- **Quality**: All files verified against datasheets
"""
    
    checklist_file = Path('cad/CAD_Download_Checklist.md')
    with open(checklist_file, 'w', encoding='utf-8') as f:
        f.write(checklist)
    
    print(f"📋 Generated download checklist: {checklist_file}")
    return checklist_file

def create_real_conversion_script():
    """Create script for converting real STEP files to OBJ"""
    script = '''#!/usr/bin/env python3
"""
Real CAD Model Conversion Script
Converts downloaded STEP/IGES files to OBJ format for Python integration
"""

import trimesh
import glob
import os
from pathlib import Path
import json

def convert_step_to_obj():
    """Convert all STEP files to OBJ format"""
    print("🔄 CONVERTING REAL CAD MODELS...")
    
    step_dir = Path('cad/step')
    obj_dir = Path('cad/obj')
    
    # Find all STEP/IGES files
    step_files = list(step_dir.glob('*.step')) + list(step_dir.glob('*.stp'))
    iges_files = list(step_dir.glob('*.iges')) + list(step_dir.glob('*.igs'))
    all_files = step_files + iges_files
    
    print(f"📦 Found {len(all_files)} CAD files to convert")
    
    converted = 0
    failed = 0
    
    for cad_file in all_files:
        try:
            print(f"   🔄 Converting {cad_file.name}...")
            
            # Load CAD file
            mesh = trimesh.load(str(cad_file))
            
            # Create OBJ filename
            obj_file = obj_dir / f"{cad_file.stem}.obj"
            
            # Export to OBJ
            mesh.export(str(obj_file))
            
            print(f"   ✅ Converted to {obj_file.name}")
            converted += 1
            
        except Exception as e:
            print(f"   ❌ Failed to convert {cad_file.name}: {e}")
            failed += 1
    
    print(f"\\n📊 CONVERSION COMPLETE:")
    print(f"   • Converted: {converted} files")
    print(f"   • Failed: {failed} files")
    print(f"   • Success rate: {converted/(converted+failed)*100:.1f}%")
    
    return converted, failed

def validate_obj_files():
    """Validate converted OBJ files"""
    print("\\n🔍 VALIDATING OBJ FILES...")
    
    obj_dir = Path('cad/obj')
    obj_files = list(obj_dir.glob('*.obj'))
    
    valid = 0
    invalid = 0
    
    for obj_file in obj_files:
        try:
            mesh = trimesh.load(str(obj_file))
            if mesh.is_valid:
                print(f"   ✅ {obj_file.name}: Valid mesh ({mesh.vertices.shape[0]} vertices)")
                valid += 1
            else:
                print(f"   ⚠️ {obj_file.name}: Invalid mesh geometry")
                invalid += 1
        except Exception as e:
            print(f"   ❌ {obj_file.name}: Load error - {e}")
            invalid += 1
    
    print(f"\\n📊 VALIDATION RESULTS:")
    print(f"   • Valid: {valid} files")
    print(f"   • Invalid: {invalid} files")
    print(f"   • Quality: {valid/(valid+invalid)*100:.1f}%")

def integrate_with_circuit_dsl():
    """Integration with LQG Circuit DSL framework"""
    print("\\n🔗 INTEGRATING WITH LQG CIRCUIT DSL...")
    
    # Load parts database
    with open('cad/lqr1_parts.json', 'r') as f:
        parts = json.load(f)
    
    # Check which parts have CAD models
    obj_dir = Path('cad/obj')
    available_models = {f.stem for f in obj_dir.glob('*.obj')}
    
    parts_with_models = []
    parts_without_models = []
    
    for part in parts:
        if part['part_number'] in available_models:
            parts_with_models.append(part)
        else:
            parts_without_models.append(part)
    
    print(f"   📊 CAD Model Availability:")
    print(f"   • Parts with models: {len(parts_with_models)}")
    print(f"   • Parts without models: {len(parts_without_models)}")
    print(f"   • Coverage: {len(parts_with_models)/len(parts)*100:.1f}%")
    
    # Generate integration summary
    summary = {
        'total_parts': len(parts),
        'parts_with_models': len(parts_with_models),
        'parts_without_models': len(parts_without_models),
        'coverage_percentage': len(parts_with_models)/len(parts)*100,
        'available_models': list(available_models)
    }
    
    with open('cad/integration_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"   📁 Integration summary saved: cad/integration_summary.json")

if __name__ == "__main__":
    print("🚀 REAL CAD MODEL CONVERSION PIPELINE")
    print("=" * 50)
    
    # Convert STEP to OBJ
    converted, failed = convert_step_to_obj()
    
    if converted > 0:
        # Validate OBJ files
        validate_obj_files()
        
        # Integrate with Circuit DSL
        integrate_with_circuit_dsl()
        
        print("\\n✅ CONVERSION PIPELINE COMPLETE!")
        print("   Ready for LQG Circuit DSL integration")
    else:
        print("\\n⚠️ No files converted - check STEP/IGES files")
'''
    
    script_file = Path('cad/convert_real_cad_models.py')
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write(script)
    
    print(f"🔄 Created conversion script: {script_file}")
    return script_file

def realistic_component_search():
    """Generate realistic search terms for actual CAD components"""
    print("🔍 REALISTIC CAD SEARCH STRATEGIES")
    print("=" * 50)
    
    # Map LQR-1 specs to real commercial equivalents
    realistic_searches = {
        "Vacuum Chamber (3.5m torus)": {
            "search_terms": ["CF150 vacuum chamber", "spherical vacuum vessel", "cylindrical chamber"],
            "real_parts": ["MDC 948005", "Lesker VPZL-275", "NorCal VCC-150"],
            "suppliers": ["mdcvacuum.com", "lesker.com", "n-c.com"]
        },
        "Toroidal Field Coils (NbTi)": {
            "search_terms": ["superconducting magnet coil", "NbTi wire coil", "cryogenic magnet"],
            "real_parts": ["Oxford 15T magnet", "Bruker WB140", "Cryomagnetics 4K"],
            "suppliers": ["oxford-instruments.com", "bruker.com", "cryomagnetics.com"]
        },
        "ConFlat Flanges": {
            "search_terms": ["CF150 flange", "CF40 blank", "ultra-high vacuum flange"],
            "real_parts": ["MDC 150000", "Lesker CFFT150", "NorCal CF150-BlkFlg"],
            "suppliers": ["mdcvacuum.com", "lesker.com", "n-c.com"]
        },
        "Turbo Pumps": {
            "search_terms": ["turbo molecular pump", "HiPace 700", "STP-iXA"],
            "real_parts": ["Pfeiffer HiPace 700", "Edwards STP-iXA4506", "Leybold TURBOVAC"],
            "suppliers": ["pfeiffer-vacuum.com", "edwardsvacuum.com", "leybold.com"]
        },
        "Power Converters": {
            "search_terms": ["50kW thyristor converter", "IGBT power module", "high power rectifier"],
            "real_parts": ["ABB ACS880", "Danfysik System 8000", "Schneider ATV71"],
            "suppliers": ["abb.com", "danfysik.com", "schneider-electric.com"]
        }
    }
    
    for category, info in realistic_searches.items():
        print(f"\n🔧 {category}:")
        print(f"   Search Terms: {', '.join(info['search_terms'])}")
        print(f"   Real Parts: {', '.join(info['real_parts'])}")
        print(f"   Suppliers: {', '.join(info['suppliers'])}")
    
    return realistic_searches

def open_realistic_searches():
    """Open searches for actual available components"""
    print("\n🌐 OPENING REALISTIC COMPONENT SEARCHES...")
    
    # Real component searches that will return results
    search_urls = [
        # Vacuum chambers
        "https://www.lesker.com/vacuum-chambers/",
        "https://www.mdcvacuum.com/chambers/",
        "https://www.traceparts.com/en/search?Keywords=CF150+vacuum+chamber",
        
        # Turbo pumps (these definitely exist)
        "https://www.pfeiffer-vacuum.com/en/products/vacuum-pumps/turbopumps/",
        "https://www.edwardsvacuum.com/products/turbomolecular-pumps/",
        "https://www.grabcad.com/library?query=turbo+molecular+pump",
        
        # ConFlat flanges (standard vacuum components)
        "https://www.lesker.com/flanges/conflat/",
        "https://www.traceparts.com/en/search?Keywords=CF150+flange",
        
        # Power electronics
        "https://www.abb.com/products/power-converters-inverters/",
        "https://www.snapeda.com/search/?q=IGBT+module"
    ]
    
    print("   Opening real component catalogs...")
    try:
        for i, url in enumerate(search_urls[:3]):  # Open first 3 to avoid spam
            webbrowser.open(url)
        print(f"   ✅ Opened {len(search_urls[:3])} realistic component searches")
    except Exception as e:
        print(f"   ⚠️ Browser opening failed: {e}")

if __name__ == "__main__":
    print("🎯 REALISTIC CAD MODEL ACQUISITION GUIDE")
    print("=" * 50)
    
    # Show realistic search strategies first
    realistic_component_search()
    
    # Open realistic searches instead of generic ones
    open_realistic_searches()
    
    # Open supplier websites
    open_supplier_websites()
    
    # Generate download checklist
    checklist = generate_download_checklist()
    
    # Create conversion script
    converter = create_real_conversion_script()
    
    print("\n" + "=" * 50)
    print("📋 SUMMARY:")
    print("=" * 50)
    print("⚠️  IMPORTANT: LQR-1 uses CUSTOM fusion reactor components!")
    print("   Real CAD files available for standard equivalents only.")
    print(f"✅ Generated checklist: {checklist}")
    print(f"✅ Created converter: {converter}")
    print("✅ Opened realistic component searches")
    
    print("\n🚀 REALISTIC NEXT STEPS:")
    print("1. Search for REAL component equivalents (not exact LQR-1 parts)")
    print("2. Download standard vacuum/power/magnet component STEP files")
    print("3. Adapt geometry to match LQR-1 specifications")
    print("4. Use real CAD as basis for custom designs")
    
    print("\n💡 Example: Search 'CF150 vacuum chamber' instead of 'TVC-350-120-W99'")
    print("   The real parts will have STEP files you can actually download!")

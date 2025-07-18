#!/usr/bin/env python3
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
    
    print(f"\n📊 CONVERSION COMPLETE:")
    print(f"   • Converted: {converted} files")
    print(f"   • Failed: {failed} files")
    print(f"   • Success rate: {converted/(converted+failed)*100:.1f}%")
    
    return converted, failed

def validate_obj_files():
    """Validate converted OBJ files"""
    print("\n🔍 VALIDATING OBJ FILES...")
    
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
    
    print(f"\n📊 VALIDATION RESULTS:")
    print(f"   • Valid: {valid} files")
    print(f"   • Invalid: {invalid} files")
    print(f"   • Quality: {valid/(valid+invalid)*100:.1f}%")

def integrate_with_circuit_dsl():
    """Integration with LQG Circuit DSL framework"""
    print("\n🔗 INTEGRATING WITH LQG CIRCUIT DSL...")
    
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
        
        print("\n✅ CONVERSION PIPELINE COMPLETE!")
        print("   Ready for LQG Circuit DSL integration")
    else:
        print("\n⚠️ No files converted - check STEP/IGES files")

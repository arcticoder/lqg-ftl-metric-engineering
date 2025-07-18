#!/usr/bin/env python3
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
    print("\n📋 Step 1: Parsing parts list...")
    subprocess.run([sys.executable, "parse_parts_list.py"])
    
    # Step 2: Download CAD models
    print("\n🔽 Step 2: Downloading CAD models...")
    subprocess.run([sys.executable, "download_cad_models.py"])
    
    # Step 3: Generate electrical schematic
    print("\n⚡ Step 3: Generating electrical schematic...")
    subprocess.run([sys.executable, "generate_electrical_schematic.py"])
    
    # Step 4: Generate mechanical assembly
    print("\n🔧 Step 4: Generating mechanical assembly...")
    subprocess.run([sys.executable, "generate_mechanical_assembly.py"])
    
    # Step 5: Create final documentation
    print("\n📚 Step 5: Generating documentation...")
    subprocess.run([sys.executable, "generate_documentation.py"])
    
    print("\n" + "=" * 50)
    print("✅ LQR-1 CAD PIPELINE COMPLETE!")
    print("📁 Outputs available in cad/ directory")
    print("🎯 Ready for manufacturing and construction")

if __name__ == "__main__":
    run_complete_pipeline()

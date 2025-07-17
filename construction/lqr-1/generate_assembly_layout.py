#!/usr/bin/env python3
"""
LQG Assembly Layout Schematic Generator
Generates detailed assembly layout diagrams for LQR-1 construction
"""

import sys
import os

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import schemdraw
    import schemdraw.elements as elm
    SCHEMDRAW_AVAILABLE = True
except ImportError:
    SCHEMDRAW_AVAILABLE = False
    print("⚠️ Schemdraw not available - install with: pip install schemdraw")

def generate_assembly_layout_schematic():
    """Generate detailed assembly layout schematic for construction"""
    if not SCHEMDRAW_AVAILABLE:
        print("❌ Cannot generate assembly layout - Schemdraw required")
        return None
        
    # Create assembly layout drawing
    drawing = schemdraw.Drawing()
    drawing.config(lw=2, fontsize=10)
    
    # Title and header
    drawing.add(elm.Label().label('LQR-1 FUSION REACTOR - ASSEMBLY LAYOUT').scale(1.8))
    drawing.add(elm.Line().down(0.5))
    
    # Central reactor vessel (large rectangle representing toroidal chamber)
    drawing.push()
    drawing.move(0, -2)
    reactor_vessel = drawing.add(elm.Rect((-3, -2), (3, 2)).fill('lightblue').label('TOROIDAL\\nREACTOR VESSEL\\n3.5m Major Radius'))
    drawing.pop()
    
    # Superconducting magnet coils (around the reactor)
    drawing.push()
    drawing.move(-4, 0)
    drawing.add(elm.Rect((-1, -3), (1, 3)).fill('yellow').label('TF COILS\\nNbTi/Nb₃Sn\\n5.3T Field'))
    drawing.pop()
    
    drawing.push()
    drawing.move(4, 0)
    drawing.add(elm.Rect((-1, -3), (1, 3)).fill('yellow').label('TF COILS\\nNbTi/Nb₃Sn\\n5.3T Field'))
    drawing.pop()
    
    # Power conversion system
    drawing.push()
    drawing.move(7, 0)
    drawing.add(elm.Rect((-1.5, -1.5), (1.5, 1.5)).fill('orange').label('POWER\\nCONVERSION\\n200 MW'))
    # Connection line from reactor to power system
    drawing.add(elm.Line().left(4))
    drawing.add(elm.Arrow().left(1).label('Electrical'))
    drawing.pop()
    
    # Coolant system
    drawing.push()
    drawing.move(-7, 1.5)
    drawing.add(elm.Rect((-1.5, -1), (1.5, 1)).fill('lightgreen').label('COOLANT\\nSYSTEM\\nLi-6 Primary'))
    # Coolant loop connections
    drawing.add(elm.Line().right(2))
    drawing.add(elm.Arrow().right(2).label('Coolant In'))
    drawing.pop()
    
    drawing.push()
    drawing.move(-7, -1.5)
    drawing.add(elm.Rect((-1.5, -1), (1.5, 1)).fill('lightgreen').label('HEAT\\nEXCHANGER\\n550°C'))
    # Return coolant line
    drawing.add(elm.Line().right(2))
    drawing.add(elm.Arrow().right(2).label('Coolant Out'))
    drawing.pop()
    
    # Fuel injection system
    drawing.push()
    drawing.move(0, 4)
    drawing.add(elm.Rect((-1.5, -1), (1.5, 1)).fill('pink').label('FUEL\\nINJECTION\\nD-T System'))
    # Fuel line to reactor
    drawing.add(elm.Line().down(2))
    drawing.add(elm.Arrow().down(1).label('D-T Fuel'))
    drawing.pop()
    
    # LQG Polymer Field Generators (distributed around reactor)
    for i, (x, y, label) in enumerate([(-2, 3, 'LQG-1'), (2, 3, 'LQG-2'), (-2, -3, 'LQG-3'), (2, -3, 'LQG-4')]):
        drawing.push()
        drawing.move(x, y)
        drawing.add(elm.Rect((-0.8, -0.5), (0.8, 0.5)).fill('purple').label(f'{label}\\nPolymer\\nField'))
        # Connection to reactor
        drawing.add(elm.Line().to((0, 0 if y > 0 else 0)))
        drawing.pop()
    
    # Control system
    drawing.push()
    drawing.move(0, -5)
    drawing.add(elm.Rect((-2, -1), (2, 1)).fill('gray').label('CONTROL SYSTEM\\nDigital Twin Integration\\nReal-time Monitoring'))
    # Control connections to major systems
    drawing.add(elm.Line().up(1))
    drawing.add(elm.Arrow().up(1).label('Control'))
    drawing.pop()
    
    # Radiation shielding (outer boundary)
    drawing.push()
    drawing.move(0, 0)
    shielding = drawing.add(elm.Rect((-9, -6), (9, 6)).linestyle('--'))
    shielding.label('RADIATION SHIELDING\\n2m Tungsten + Lead-Boron\\n0.00 mSv/year')
    drawing.pop()
    
    # Assembly specifications
    drawing.push()
    drawing.move(-8, -8)
    drawing.add(elm.Label().label('ASSEMBLY SPECIFICATIONS:\\n• Foundation: Reinforced concrete, 15m × 15m\\n• Crane Access: 500-ton overhead crane\\n• Utilities: 50 MW power supply, cooling water\\n• Timeline: 60 months construction\\n• Cost: $485.75M total\\n• Safety: BLACK/RED LABEL classification'))
    drawing.pop()
    
    # Save the assembly layout
    os.makedirs('construction/lqr-1', exist_ok=True)
    drawing.save('construction/lqr-1/lqr-1_assembly_layout.svg')
    
    print("✅ Assembly layout schematic generated: construction/lqr-1/lqr-1_assembly_layout.svg")
    return drawing

if __name__ == "__main__":
    print("🏗️ LQG ASSEMBLY LAYOUT GENERATOR")
    print("=" * 50)
    
    schematic = generate_assembly_layout_schematic()
    
    if schematic:
        print("🎉 Assembly layout schematic generation complete!")
        print("   Files ready for construction planning and implementation.")
    else:
        print("❌ Assembly layout generation failed")
        print("   Check dependencies: pip install schemdraw")
    
    print("=" * 50)

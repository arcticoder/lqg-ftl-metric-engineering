#!/usr/bin/env python3
"""
Test Schemdraw functionality to verify proper SVG generation
"""

import schemdraw
import schemdraw.elements as elm

def test_schemdraw():
    """Test basic Schemdraw functionality"""
    print("Testing Schemdraw...")
    
    # Create a simple test drawing
    drawing = schemdraw.Drawing()
    drawing.config(lw=2, fontsize=12)
    
    # Add a simple circuit
    drawing.add(elm.Label().label('Test Schematic'))
    drawing.add(elm.Line().down(0.5))
    drawing.add(elm.Rect(w=3, h=2).fill('lightblue').label('Test Component'))
    drawing.add(elm.Line().right(2))
    drawing.add(elm.Dot().label('Output'))
    
    # Save test file
    drawing.save('construction/lqr-1/test_schemdraw.svg')
    print("✅ Test schematic saved to construction/lqr-1/test_schemdraw.svg")
    
    return drawing

if __name__ == "__main__":
    test_schemdraw()

#!/usr/bin/env python3
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

    # Main power converter, thyristor-based
    ps1 = Part('LQR1_Lib', 'PS1', 
                    footprint='Custom:PS1_footprint',
                    value='3200000.0',
                    desc='Main power converter, thyristor-based')
    ps1['VCC'] += VCC_50MW
    ps1['GND'] += GND

    # Coil power supplies, individual control
    ps2 = Part('LQR1_Lib', 'PS2', 
                    footprint='Custom:PS2_footprint',
                    value='125000.0',
                    desc='Coil power supplies, individual control')
    ps2['VCC'] += VCC_50MW
    ps2['GND'] += GND

    # Energy storage systems, supercapacitor banks
    ps3 = Part('LQR1_Lib', 'PS3', 
                    footprint='Custom:PS3_footprint',
                    value='25000.0',
                    desc='Energy storage systems, supercapacitor banks')
    ps3['VCC'] += VCC_50MW
    ps3['GND'] += GND

    # Uninterruptible power supplies, 500 kVA
    ps4 = Part('LQR1_Lib', 'PS4', 
                    footprint='Custom:PS4_footprint',
                    value='85000.0',
                    desc='Uninterruptible power supplies, 500 kVA')
    ps4['VCC'] += VCC_50MW
    ps4['GND'] += GND

    # Distributed control system, real-time operation  
    cs1 = Part('LQR1_Lib', 'CS1',
                    footprint='Custom:CS1_footprint', 
                    value='1250000.0',
                    desc='Distributed control system, real-time operation')
    cs1['SENSOR_OUT'] += Net('CS1_DATA')
    cs1['VCC'] += VCC_5MW
    cs1['GND'] += GND

    # Plasma diagnostics, multi-channel systems  
    cs2 = Part('LQR1_Lib', 'CS2',
                    footprint='Custom:CS2_footprint', 
                    value='85000.0',
                    desc='Plasma diagnostics, multi-channel systems')
    cs2['SENSOR_OUT'] += Net('CS2_DATA')
    cs2['VCC'] += VCC_5MW
    cs2['GND'] += GND

    # Magnetic field sensors, Hall effect type  
    cs3 = Part('LQR1_Lib', 'CS3',
                    footprint='Custom:CS3_footprint', 
                    value='15000.0',
                    desc='Magnetic field sensors, Hall effect type')
    cs3['SENSOR_OUT'] += Net('CS3_DATA')
    cs3['VCC'] += VCC_5MW
    cs3['GND'] += GND

    
    # Generate netlist
    generate_netlist('lqr1_electrical.net')
    print("✅ Generated LQR-1 electrical netlist")

if __name__ == "__main__":
    create_lqr1_electrical_schematic()

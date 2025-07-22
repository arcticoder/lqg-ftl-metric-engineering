"""
Improved Tokamak Vacuum Chamber Geometry
Implements proper elongated D-shaped cross-section with triangularity
"""

import numpy as np
try:
    import cadquery as cq
    CADQUERY_AVAILABLE = True
except ImportError:
    CADQUERY_AVAILABLE = False

from tokamak_vacuum_chamber_designer import TokamakParameters

class ImprovedVacuumChamberGeometry:
    """Enhanced vacuum chamber geometry with proper tokamak shape"""
    
    def __init__(self):
        self.wall_thickness = 0.15  # More realistic wall thickness
        
    def create_tokamak_cross_section(self, params: TokamakParameters, points=100):
        """
        Create proper tokamak D-shaped cross-section with elongation and triangularity
        
        Args:
            params: TokamakParameters with R, a, kappa, delta
            points: Number of points to define the curve
            
        Returns:
            List of (r, z) coordinates defining the plasma boundary
        """
        # Theta parameter from 0 to 2π
        theta = np.linspace(0, 2*np.pi, points)
        
        # Standard tokamak parameterization with elongation and triangularity
        # r(θ) = R + a*cos(θ + δ*sin(θ))  
        # z(θ) = a*κ*sin(θ)
        
        # Apply triangularity shift to theta
        theta_shifted = theta + params.delta * np.sin(theta)
        
        # Calculate r and z coordinates
        r_coords = params.R + params.a * np.cos(theta_shifted)
        z_coords = params.a * params.kappa * np.sin(theta)
        
        return list(zip(r_coords, z_coords))
    
    def create_vacuum_chamber_profile(self, params: TokamakParameters, points=100):
        """
        Create vacuum chamber wall profile (plasma boundary + wall thickness)
        """
        # Get plasma boundary
        plasma_coords = self.create_tokamak_cross_section(params, points)
        
        # Create outer wall by expanding normal to plasma boundary
        wall_coords = []
        
        for i in range(len(plasma_coords)):
            r, z = plasma_coords[i]
            
            # Calculate normal vector at this point
            # Use neighboring points to estimate tangent
            r_prev, z_prev = plasma_coords[i-1]
            r_next, z_next = plasma_coords[(i+1) % len(plasma_coords)]
            
            # Tangent vector
            dr_dt = (r_next - r_prev) / 2
            dz_dt = (z_next - z_prev) / 2
            
            # Normal vector (perpendicular to tangent, pointing outward)
            normal_r = dz_dt
            normal_z = -dr_dt
            
            # Normalize
            norm_length = np.sqrt(normal_r**2 + normal_z**2)
            if norm_length > 0:
                normal_r /= norm_length
                normal_z /= norm_length
            
            # Create outer wall point
            wall_r = r + self.wall_thickness * normal_r
            wall_z = z + self.wall_thickness * normal_z
            
            wall_coords.append((wall_r, wall_z))
        
        return plasma_coords, wall_coords
    
    def generate_improved_tokamak_cad(self, params: TokamakParameters):
        """Generate improved tokamak CAD with proper D-shaped geometry"""
        
        if not CADQUERY_AVAILABLE:
            print("CadQuery not available - returning geometric data")
            plasma_coords, wall_coords = self.create_vacuum_chamber_profile(params)
            return {
                'plasma_boundary': plasma_coords,
                'wall_boundary': wall_coords,
                'major_radius': params.R,
                'minor_radius': params.a,
                'elongation': params.kappa,
                'triangularity': params.delta,
                'wall_thickness': self.wall_thickness
            }
        
        try:
            print(f"Generating improved CAD: R={params.R:.2f}m, a={params.a:.2f}m, κ={params.kappa:.2f}, δ={params.delta:.2f}")
            
            # Get plasma and wall boundary coordinates
            plasma_coords, wall_coords = self.create_vacuum_chamber_profile(params, points=50)
            
            # Create plasma cavity profile
            plasma_profile = cq.Workplane("XZ").moveTo(*plasma_coords[0])
            for r, z in plasma_coords[1:]:
                plasma_profile = plasma_profile.lineTo(r, z)
            plasma_profile = plasma_profile.close()
            
            # Create wall profile  
            wall_profile = cq.Workplane("XZ").moveTo(*wall_coords[0])
            for r, z in wall_coords[1:]:
                wall_profile = wall_profile.lineTo(r, z)
            wall_profile = wall_profile.close()
            
            # Revolve to create 3D torus
            plasma_cavity = plasma_profile.revolve(360, (0, 0, 1))
            wall_solid = wall_profile.revolve(360, (0, 0, 1))
            
            # Create hollow chamber
            chamber = wall_solid.cut(plasma_cavity)
            
            # Add specialized ports
            chamber = self._add_tokamak_ports(chamber, params)
            
            # Add support structure
            chamber = self._add_support_structure(chamber, params)
            
            return chamber
            
        except Exception as e:
            print(f"CAD generation failed: {e}")
            # Return geometric data instead
            plasma_coords, wall_coords = self.create_vacuum_chamber_profile(params)
            return {
                'plasma_boundary': plasma_coords,
                'wall_boundary': wall_coords,
                'major_radius': params.R,
                'minor_radius': params.a,
                'elongation': params.kappa,
                'triangularity': params.delta,
                'wall_thickness': self.wall_thickness,
                'cad_error': str(e)
            }
    
    def _add_tokamak_ports(self, chamber, params: TokamakParameters):
        """Add realistic tokamak ports"""
        
        # Neutral beam injection ports (2 large tangential ports)
        nbi_diameter = 0.8
        nbi_angles = [30, 150]  # degrees
        
        for angle in nbi_angles:
            x = params.R * np.cos(np.radians(angle))
            y = params.R * np.sin(np.radians(angle))
            
            nbi_port = (cq.Workplane("XY")
                       .center(x, y)
                       .circle(nbi_diameter/2)
                       .extrude(params.a * 1.5))
            chamber = chamber.cut(nbi_port)
        
        # Diagnostic ports (multiple small ports)
        diag_diameter = 0.3
        diag_angles = [45, 135, 225, 315]  # degrees
        
        for angle in diag_angles:
            x = params.R * np.cos(np.radians(angle))
            y = params.R * np.sin(np.radians(angle))
            
            # Add ports at different Z levels
            for z_offset in [-params.a*0.5, 0, params.a*0.5]:
                diag_port = (cq.Workplane("XY", origin=(0, 0, z_offset))
                            .center(x, y)  
                            .circle(diag_diameter/2)
                            .extrude(params.a))
                chamber = chamber.cut(diag_port)
        
        # Pumping ports (4 large ports at bottom)
        pump_diameter = 0.6
        pump_angles = [60, 120, 240, 300]  # degrees
        
        for angle in pump_angles:
            x = params.R * np.cos(np.radians(angle))
            y = params.R * np.sin(np.radians(angle))
            
            pump_port = (cq.Workplane("XY", origin=(0, 0, -params.a*params.kappa*0.7))
                        .center(x, y)
                        .circle(pump_diameter/2) 
                        .extrude(0.5))
            chamber = chamber.cut(pump_port)
        
        return chamber
    
    def _add_support_structure(self, chamber, params: TokamakParameters):
        """Add realistic tokamak support structure"""
        
        # Toroidal field coil supports (simplified)
        n_supports = 16
        support_width = 0.3
        support_height = params.a * params.kappa * 2.5
        
        for i in range(n_supports):
            angle = 2 * np.pi * i / n_supports
            x = (params.R + params.a + 0.5) * np.cos(angle)
            y = (params.R + params.a + 0.5) * np.sin(angle)
            
            support = (cq.Workplane("XY", origin=(x, y, -support_height/2))
                      .rect(support_width, support_width)
                      .extrude(support_height))
            
            chamber = chamber.union(support)
        
        # Base platform
        platform_radius = params.R + params.a + 1.0
        platform_thickness = 0.2
        
        platform = (cq.Workplane("XY", origin=(0, 0, -params.a*params.kappa - platform_thickness))
                   .circle(platform_radius)
                   .extrude(platform_thickness))
        
        chamber = chamber.union(platform)
        
        return chamber

def test_improved_geometry():
    """Test the improved geometry with sample parameters"""
    
    # Create test parameters - TokamakParameters requires all arguments
    class TestParams:
        def __init__(self):
            self.R = 4.5      # Major radius (m)
            self.a = 1.2      # Minor radius (m) 
            self.kappa = 1.8  # Elongation
            self.delta = 0.4  # Triangularity
            self.mu = 0.3     # LQG parameter
            self.B0 = 5.5     # Magnetic field (T)
            self.Ip = 8.0     # Plasma current (MA)
    
    params = TestParams()
    
    # Test geometry creation
    geometry = ImprovedVacuumChamberGeometry()
    
    # Test cross-section generation
    plasma_coords, wall_coords = geometry.create_vacuum_chamber_profile(params)
    
    print(f"Generated {len(plasma_coords)} plasma boundary points")
    print(f"Generated {len(wall_coords)} wall boundary points")
    
    # Test a few coordinate values
    print(f"Sample plasma coords: {plasma_coords[0]}, {plasma_coords[len(plasma_coords)//4]}")
    print(f"Sample wall coords: {wall_coords[0]}, {wall_coords[len(wall_coords)//4]}")
    
    # Verify elongation is applied
    z_coords = [coord[1] for coord in plasma_coords]
    max_z = max(z_coords)
    expected_max_z = params.a * params.kappa
    print(f"Max Z coordinate: {max_z:.3f}m, Expected: {expected_max_z:.3f}m")
    
    # Test CAD generation
    cad_result = geometry.generate_improved_tokamak_cad(params)
    
    if isinstance(cad_result, dict):
        print("CAD generation returned geometric data:")
        for key, value in cad_result.items():
            if key in ['plasma_boundary', 'wall_boundary']:
                print(f"  {key}: {len(value) if isinstance(value, list) else 'N/A'} points")
            else:
                print(f"  {key}: {value}")
    else:
        print("CAD generation successful - 3D model created")
    
    return geometry, cad_result

if __name__ == "__main__":
    test_improved_geometry()

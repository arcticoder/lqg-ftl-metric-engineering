"""
Ship Hull Geometry OBJ Framework - Phase 2: OBJ Mesh Generation
==============================================================

Advanced OBJ file generation from physics-informed hull geometry with 
material properties, UV mapping, and WebGL optimization.
"""

import numpy as np
import os
from typing import Dict, List, Tuple, Optional, TextIO
from dataclasses import dataclass
import logging

from hull_geometry_generator import HullGeometry, HullPhysicsEngine, AlcubierreMetricConstraints

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OBJMaterial:
    """Material definition for OBJ export."""
    name: str
    ambient_color: Tuple[float, float, float]  # Ka
    diffuse_color: Tuple[float, float, float]  # Kd  
    specular_color: Tuple[float, float, float]  # Ks
    specular_exponent: float  # Ns
    transparency: float  # d (1.0 = opaque)
    illumination_model: int  # illum
    texture_map: Optional[str] = None  # map_Kd

@dataclass 
class UVMapping:
    """UV texture coordinate mapping."""
    vertices: np.ndarray  # Shape (N, 2) - UV coordinates [0,1] x [0,1]
    faces: np.ndarray  # Shape (M, 3) - UV face indices
    
class OBJMeshGenerator:
    """
    Phase 2: OBJ Mesh Generation
    
    Converts physics-informed hull geometry to industry-standard OBJ format
    with materials, UV mapping, and WebGL optimization.
    """
    
    def __init__(self):
        """Initialize OBJ mesh generator."""
        self.logger = logging.getLogger(f"{__name__}.OBJMeshGenerator")
        
        # Standard FTL hull materials
        self.materials = {
            'hull_primary': OBJMaterial(
                name='hull_primary',
                ambient_color=(0.1, 0.1, 0.12),
                diffuse_color=(0.3, 0.35, 0.4),
                specular_color=(0.8, 0.85, 0.9),
                specular_exponent=128.0,
                transparency=1.0,
                illumination_model=2
            ),
            'hull_reinforced': OBJMaterial(
                name='hull_reinforced',
                ambient_color=(0.15, 0.1, 0.05),
                diffuse_color=(0.4, 0.3, 0.2),
                specular_color=(0.9, 0.8, 0.7),
                specular_exponent=256.0,
                transparency=1.0,
                illumination_model=2
            ),
            'hull_transparent': OBJMaterial(
                name='hull_transparent',
                ambient_color=(0.05, 0.05, 0.1),
                diffuse_color=(0.1, 0.1, 0.3),
                specular_color=(0.9, 0.9, 1.0),
                specular_exponent=512.0,
                transparency=0.3,
                illumination_model=4  # Transparent
            )
        }
        
    def generate_uv_mapping(self, hull_geometry: HullGeometry) -> UVMapping:
        """
        Generate UV texture coordinates for hull geometry.
        
        Args:
            hull_geometry: Physics-informed hull geometry
            
        Returns:
            uv_mapping: UV coordinate mapping for texturing
        """
        vertices = hull_geometry.vertices
        n_vertices = len(vertices)
        
        # Calculate cylindrical UV mapping for ship hull
        uv_coords = np.zeros((n_vertices, 2))
        
        # Get hull bounds
        x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
        y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
        z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
        
        for i, vertex in enumerate(vertices):
            x, y, z = vertex
            
            # U coordinate: longitudinal position (bow to stern)
            u = (x - x_min) / (x_max - x_min)
            
            # V coordinate: cylindrical projection around hull
            # Calculate angle around the hull centerline
            r = np.sqrt(y**2 + z**2)
            if r > 1e-12:
                theta = np.arctan2(z, y)
                # Normalize to [0, 1]
                v = (theta + np.pi) / (2 * np.pi)
            else:
                v = 0.0
                
            uv_coords[i] = [u, v]
            
        # UV faces match geometry faces
        uv_faces = hull_geometry.faces.copy()
        
        return UVMapping(vertices=uv_coords, faces=uv_faces)
        
    def assign_materials_by_thickness(self, hull_geometry: HullGeometry) -> np.ndarray:
        """
        Assign materials to faces based on hull thickness analysis.
        
        Args:
            hull_geometry: Physics-informed hull geometry
            
        Returns:
            material_assignments: Material name for each face
        """
        n_faces = len(hull_geometry.faces)
        material_assignments = np.empty(n_faces, dtype=object)
        
        # Calculate average thickness per face
        face_thickness = np.zeros(n_faces)
        for i, face in enumerate(hull_geometry.faces):
            face_vertices = face
            face_thickness[i] = hull_geometry.thickness_map[face_vertices].mean()
            
        # Thickness thresholds for material assignment
        thick_threshold = np.percentile(face_thickness, 80)  # Top 20% thickest
        thin_threshold = np.percentile(face_thickness, 20)   # Bottom 20% thinnest
        
        for i in range(n_faces):
            if face_thickness[i] >= thick_threshold:
                material_assignments[i] = 'hull_reinforced'
            elif face_thickness[i] <= thin_threshold:
                material_assignments[i] = 'hull_transparent'
            else:
                material_assignments[i] = 'hull_primary'
                
        self.logger.info(
            f"Material assignment: {np.sum(material_assignments == 'hull_reinforced')} reinforced, "
            f"{np.sum(material_assignments == 'hull_primary')} primary, "
            f"{np.sum(material_assignments == 'hull_transparent')} transparent"
        )
        
        return material_assignments
        
    def write_mtl_file(self, filepath: str) -> None:
        """
        Write MTL (Material Template Library) file for OBJ materials.
        
        Args:
            filepath: Path to write MTL file
        """
        with open(filepath, 'w') as f:
            f.write("# FTL Ship Hull Materials\n")
            f.write("# Generated by Ship Hull Geometry OBJ Framework\n\n")
            
            for material in self.materials.values():
                f.write(f"newmtl {material.name}\n")
                f.write(f"Ka {material.ambient_color[0]:.6f} {material.ambient_color[1]:.6f} {material.ambient_color[2]:.6f}\n")
                f.write(f"Kd {material.diffuse_color[0]:.6f} {material.diffuse_color[1]:.6f} {material.diffuse_color[2]:.6f}\n")
                f.write(f"Ks {material.specular_color[0]:.6f} {material.specular_color[1]:.6f} {material.specular_color[2]:.6f}\n")
                f.write(f"Ns {material.specular_exponent:.1f}\n")
                f.write(f"d {material.transparency:.6f}\n")
                f.write(f"illum {material.illumination_model}\n")
                
                if material.texture_map:
                    f.write(f"map_Kd {material.texture_map}\n")
                    
                f.write("\n")
                
        self.logger.info(f"MTL file written: {filepath}")
        
    def write_obj_file(self, 
                      hull_geometry: HullGeometry,
                      filepath: str,
                      include_materials: bool = True,
                      include_normals: bool = True,
                      include_uvs: bool = True) -> Dict:
        """
        Write complete OBJ file with hull geometry, materials, and UV mapping.
        
        Args:
            hull_geometry: Physics-informed hull geometry
            filepath: Path to write OBJ file
            include_materials: Whether to include material assignments
            include_normals: Whether to include vertex normals
            include_uvs: Whether to include UV coordinates
            
        Returns:
            export_info: Information about the exported mesh
        """
        # Generate UV mapping
        uv_mapping = self.generate_uv_mapping(hull_geometry) if include_uvs else None
        
        # Assign materials
        material_assignments = self.assign_materials_by_thickness(hull_geometry) if include_materials else None
        
        # Write MTL file
        mtl_filepath = filepath.replace('.obj', '.mtl')
        if include_materials:
            self.write_mtl_file(mtl_filepath)
            
        with open(filepath, 'w') as f:
            # Header
            f.write("# FTL Ship Hull Geometry\n")
            f.write("# Generated by Ship Hull Geometry OBJ Framework Phase 2\n")
            f.write("# Physics-informed hull with Alcubierre metric constraints\n\n")
            
            # Material library reference
            if include_materials:
                mtl_filename = os.path.basename(mtl_filepath)
                f.write(f"mtllib {mtl_filename}\n\n")
                
            # Vertices
            f.write("# Vertices\n")
            for i, vertex in enumerate(hull_geometry.vertices):
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
                
            f.write("\n")
            
            # UV coordinates
            if include_uvs and uv_mapping:
                f.write("# UV Coordinates\n")
                for uv in uv_mapping.vertices:
                    f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
                f.write("\n")
                
            # Vertex normals
            if include_normals:
                # Calculate vertex normals from face normals
                vertex_normals = np.zeros_like(hull_geometry.vertices)
                vertex_counts = np.zeros(len(hull_geometry.vertices))
                
                for i, face in enumerate(hull_geometry.faces):
                    face_normal = hull_geometry.normals[i]
                    for vertex_idx in face:
                        vertex_normals[vertex_idx] += face_normal
                        vertex_counts[vertex_idx] += 1
                        
                # Normalize
                for i in range(len(vertex_normals)):
                    if vertex_counts[i] > 0:
                        vertex_normals[i] /= vertex_counts[i]
                        norm = np.linalg.norm(vertex_normals[i])
                        if norm > 1e-12:
                            vertex_normals[i] /= norm
                            
                f.write("# Vertex Normals\n")
                for normal in vertex_normals:
                    f.write(f"vn {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
                f.write("\n")
                
            # Group faces by material
            if include_materials:
                material_groups = {}
                for i, material in enumerate(material_assignments):
                    if material not in material_groups:
                        material_groups[material] = []
                    material_groups[material].append(i)
                    
                # Write faces grouped by material
                for material_name, face_indices in material_groups.items():
                    f.write(f"# Material: {material_name}\n")
                    f.write(f"usemtl {material_name}\n")
                    
                    for face_idx in face_indices:
                        face = hull_geometry.faces[face_idx]
                        if include_uvs and include_normals:
                            # f vertex/uv/normal vertex/uv/normal vertex/uv/normal
                            f.write(f"f {face[0]+1}/{face[0]+1}/{face[0]+1} "
                                   f"{face[1]+1}/{face[1]+1}/{face[1]+1} "
                                   f"{face[2]+1}/{face[2]+1}/{face[2]+1}\n")
                        elif include_uvs:
                            # f vertex/uv vertex/uv vertex/uv
                            f.write(f"f {face[0]+1}/{face[0]+1} "
                                   f"{face[1]+1}/{face[1]+1} "
                                   f"{face[2]+1}/{face[2]+1}\n")
                        elif include_normals:
                            # f vertex//normal vertex//normal vertex//normal  
                            f.write(f"f {face[0]+1}//{face[0]+1} "
                                   f"{face[1]+1}//{face[1]+1} "
                                   f"{face[2]+1}//{face[2]+1}\n")
                        else:
                            # f vertex vertex vertex
                            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
                    f.write("\n")
            else:
                # Write all faces without materials
                f.write("# Faces\n")
                for face in hull_geometry.faces:
                    if include_uvs and include_normals:
                        f.write(f"f {face[0]+1}/{face[0]+1}/{face[0]+1} "
                               f"{face[1]+1}/{face[1]+1}/{face[1]+1} "
                               f"{face[2]+1}/{face[2]+1}/{face[2]+1}\n")
                    elif include_uvs:
                        f.write(f"f {face[0]+1}/{face[0]+1} "
                               f"{face[1]+1}/{face[1]+1} "
                               f"{face[2]+1}/{face[2]+1}\n")
                    elif include_normals:
                        f.write(f"f {face[0]+1}//{face[0]+1} "
                               f"{face[1]+1}//{face[1]+1} "
                               f"{face[2]+1}//{face[2]+1}\n")
                    else:
                        f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
                        
        export_info = {
            'obj_file': filepath,
            'mtl_file': mtl_filepath if include_materials else None,
            'vertices': len(hull_geometry.vertices),
            'faces': len(hull_geometry.faces),
            'materials': len(self.materials) if include_materials else 0,
            'has_uvs': include_uvs,
            'has_normals': include_normals,
            'file_size_bytes': os.path.getsize(filepath)
        }
        
        self.logger.info(
            f"OBJ export complete: {export_info['vertices']} vertices, "
            f"{export_info['faces']} faces, {export_info['file_size_bytes']} bytes"
        )
        
        return export_info
        
    def optimize_for_webgl(self, hull_geometry: HullGeometry) -> HullGeometry:
        """
        Optimize hull geometry for WebGL rendering performance.
        
        Args:
            hull_geometry: Original hull geometry
            
        Returns:
            optimized_geometry: WebGL-optimized geometry
        """
        # Target vertex count for WebGL (65k limit for 16-bit indices)
        target_vertices = 32000
        current_vertices = len(hull_geometry.vertices)
        
        if current_vertices <= target_vertices:
            self.logger.info(f"Hull geometry already WebGL optimized: {current_vertices} vertices")
            return hull_geometry
            
        # Simplification ratio
        simplification_ratio = target_vertices / current_vertices
        
        self.logger.info(
            f"Simplifying hull geometry: {current_vertices} → {target_vertices} vertices "
            f"({simplification_ratio:.2f} ratio)"
        )
        
        # Simplified vertex decimation (preserving critical geometry)
        # Keep vertices with high thickness variation or stress concentrations
        thickness_variation = np.abs(
            hull_geometry.thickness_map - hull_geometry.thickness_map.mean()
        )
        stress_importance = thickness_variation / thickness_variation.max()
        
        # Sample vertices based on importance
        importance_threshold = np.percentile(stress_importance, (1 - simplification_ratio) * 100)
        keep_indices = stress_importance >= importance_threshold
        
        # Ensure minimum vertex count
        if np.sum(keep_indices) < target_vertices:
            # If too few vertices, keep random additional ones
            remaining_indices = ~keep_indices
            additional_count = target_vertices - np.sum(keep_indices)
            additional_indices = np.random.choice(
                np.where(remaining_indices)[0], 
                size=min(additional_count, np.sum(remaining_indices)),
                replace=False
            )
            keep_indices[additional_indices] = True
            
        # Extract simplified geometry
        old_to_new_mapping = np.full(current_vertices, -1, dtype=int)
        old_to_new_mapping[keep_indices] = np.arange(np.sum(keep_indices))
        
        new_vertices = hull_geometry.vertices[keep_indices]
        new_thickness_map = hull_geometry.thickness_map[keep_indices]
        
        # Rebuild faces with valid vertex references
        valid_faces = []
        valid_normals = []
        
        for i, face in enumerate(hull_geometry.faces):
            # Check if all face vertices are kept
            new_face = [old_to_new_mapping[v] for v in face]
            if all(v >= 0 for v in new_face):
                valid_faces.append(new_face)
                valid_normals.append(hull_geometry.normals[i])
                
        optimized_geometry = HullGeometry(
            vertices=new_vertices,
            faces=np.array(valid_faces),
            normals=np.array(valid_normals),
            thickness_map=new_thickness_map,
            material_properties=hull_geometry.material_properties,
            deck_levels=hull_geometry.deck_levels
        )
        
        self.logger.info(
            f"WebGL optimization complete: {len(new_vertices)} vertices, "
            f"{len(valid_faces)} faces"
        )
        
        return optimized_geometry


def create_obj_export_demo() -> Dict:
    """
    Demonstration of OBJ mesh generation from physics-informed hull geometry.
    
    Returns:
        demo_results: Complete OBJ export demonstration results
    """
    logger.info("Starting OBJ Mesh Generation Demo")
    
    # Generate physics-informed hull (Phase 1)
    constraints = AlcubierreMetricConstraints(
        warp_velocity=48.0,
        bubble_radius=500.0,
        exotic_energy_density=0.0,
        metric_signature="(-,+,+,+)",
        coordinate_system="cartesian"
    )
    
    hull_engine = HullPhysicsEngine(constraints)
    hull_geometry = hull_engine.generate_physics_informed_hull(
        length=200.0,  # Smaller for demo
        beam=40.0,
        height=30.0,
        n_sections=15
    )
    
    # Initialize OBJ generator (Phase 2)
    obj_generator = OBJMeshGenerator()
    
    # Optimize for WebGL
    optimized_geometry = obj_generator.optimize_for_webgl(hull_geometry)
    
    # Create output directory
    output_dir = "hull_obj_exports"
    os.makedirs(output_dir, exist_ok=True)
    
    # Export different variants
    exports = {}
    
    # Full featured export
    exports['full'] = obj_generator.write_obj_file(
        optimized_geometry,
        os.path.join(output_dir, "ftl_hull_full.obj"),
        include_materials=True,
        include_normals=True,
        include_uvs=True
    )
    
    # WebGL optimized export
    exports['webgl'] = obj_generator.write_obj_file(
        optimized_geometry,
        os.path.join(output_dir, "ftl_hull_webgl.obj"),
        include_materials=True,
        include_normals=True,
        include_uvs=True
    )
    
    # Simple export (geometry only)
    exports['simple'] = obj_generator.write_obj_file(
        optimized_geometry,
        os.path.join(output_dir, "ftl_hull_simple.obj"),
        include_materials=False,
        include_normals=False,
        include_uvs=False
    )
    
    # Analysis results
    demo_results = {
        'original_geometry': {
            'vertices': len(hull_geometry.vertices),
            'faces': len(hull_geometry.faces),
            'materials': len(obj_generator.materials)
        },
        'optimized_geometry': {
            'vertices': len(optimized_geometry.vertices),
            'faces': len(optimized_geometry.faces),
            'optimization_ratio': len(optimized_geometry.vertices) / len(hull_geometry.vertices)
        },
        'exports': exports,
        'output_directory': output_dir,
        'total_files': len(exports) * 2,  # OBJ + MTL files
        'webgl_ready': len(optimized_geometry.vertices) <= 65536
    }
    
    logger.info(
        f"OBJ export demo complete: {len(exports)} variants, "
        f"{demo_results['optimized_geometry']['vertices']} vertices"
    )
    
    return demo_results


if __name__ == "__main__":
    # Run OBJ generation demonstration
    results = create_obj_export_demo()
    
    print("\n" + "="*60)
    print("SHIP HULL GEOMETRY PHASE 2: OBJ MESH GENERATION")
    print("="*60)
    print(f"Original Geometry: {results['original_geometry']['vertices']} vertices, {results['original_geometry']['faces']} faces")
    print(f"Optimized Geometry: {results['optimized_geometry']['vertices']} vertices")
    print(f"Optimization Ratio: {results['optimized_geometry']['optimization_ratio']:.2f}")
    print(f"Materials: {results['original_geometry']['materials']}")
    print(f"Export Variants: {len(results['exports'])}")
    print(f"Output Directory: {results['output_directory']}")
    print(f"WebGL Ready: {results['webgl_ready']}")
    print("\nExported Files:")
    for variant, info in results['exports'].items():
        print(f"  {variant}: {info['obj_file']} ({info['file_size_bytes']} bytes)")
    print("="*60)

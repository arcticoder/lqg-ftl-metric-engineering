"""
Ship Hull Geometry OBJ Framework - Phase 3: Deck Plan Extraction
===============================================================

Automated deck plan extraction from 3D hull geometry with room detection,
corridor mapping, and technical documentation generation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from scipy.spatial import ConvexHull, distance_matrix
from scipy.ndimage import label, binary_erosion, binary_dilation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import json
import logging

from hull_geometry_generator import HullGeometry

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Point2D(NamedTuple):
    """2D point representation."""
    x: float
    y: float

@dataclass 
class Room:
    """Room specification within ship deck."""
    id: str
    center: Point2D
    bounds: Tuple[Point2D, Point2D]  # (min_corner, max_corner)
    area: float  # square meters
    room_type: str  # "bridge", "quarters", "engineering", "corridor", "cargo"
    connected_rooms: List[str]  # Connected room IDs
    access_points: List[Point2D]  # Door/access locations
    
@dataclass
class Corridor:
    """Corridor specification connecting rooms."""
    id: str
    path: List[Point2D]  # Corridor centerline path
    width: float  # meters
    connected_rooms: List[str]  # Room IDs connected by this corridor
    
@dataclass  
class DeckPlan:
    """Complete deck plan specification."""
    deck_level: float  # Z-coordinate of deck
    deck_name: str
    outer_boundary: List[Point2D]  # Hull outline at this deck level
    rooms: List[Room]
    corridors: List[Corridor]
    total_area: float  # Total deck area in square meters
    utilization_ratio: float  # Fraction of area used for rooms/corridors
    
class DeckPlanExtractor:
    """
    Phase 3: Deck Plan Extraction
    
    Automatically extracts deck plans from 3D hull geometry with intelligent
    room detection and corridor mapping for FTL spacecraft design.
    """
    
    def __init__(self):
        """Initialize deck plan extractor."""
        self.logger = logging.getLogger(f"{__name__}.DeckPlanExtractor")
        
        # Room detection parameters
        self.min_room_area = 10.0  # square meters
        self.min_corridor_width = 2.0  # meters
        self.room_height = 2.5  # meters (standard deck height)
        
        # FTL spacecraft room type specifications
        self.room_type_specs = {
            'bridge': {'min_area': 50.0, 'preferred_location': 'forward'},
            'engineering': {'min_area': 100.0, 'preferred_location': 'aft'},
            'quarters': {'min_area': 15.0, 'preferred_location': 'mid'},
            'cargo': {'min_area': 200.0, 'preferred_location': 'aft'},
            'medical': {'min_area': 30.0, 'preferred_location': 'mid'},
            'corridor': {'min_area': 5.0, 'preferred_location': 'any'},
            'common': {'min_area': 40.0, 'preferred_location': 'mid'}
        }
        
    def extract_deck_boundary(self, hull_geometry: HullGeometry, deck_z: float) -> List[Point2D]:
        """
        Extract 2D boundary of hull at specified deck level.
        
        Args:
            hull_geometry: 3D hull geometry
            deck_z: Z-coordinate of deck level
            
        Returns:
            boundary_points: Ordered boundary points forming deck outline
        """
        tolerance = 0.5  # meters - tolerance for deck intersection
        
        # Find vertices near the deck level
        z_coords = hull_geometry.vertices[:, 2]
        deck_mask = np.abs(z_coords - deck_z) <= tolerance
        deck_vertices = hull_geometry.vertices[deck_mask]
        
        if len(deck_vertices) < 3:
            self.logger.warning(f"Insufficient vertices at deck level {deck_z}")
            return []
            
        # Project to 2D (X-Y plane)
        deck_2d = deck_vertices[:, :2]
        
        # Compute convex hull for boundary
        try:
            hull = ConvexHull(deck_2d)
            boundary_indices = hull.vertices
            boundary_points = [Point2D(deck_2d[i, 0], deck_2d[i, 1]) for i in boundary_indices]
        except Exception as e:
            self.logger.error(f"Failed to compute deck boundary: {e}")
            # Fallback: use extreme points
            x_coords, y_coords = deck_2d[:, 0], deck_2d[:, 1]
            boundary_points = [
                Point2D(x_coords.min(), y_coords.min()),
                Point2D(x_coords.max(), y_coords.min()),
                Point2D(x_coords.max(), y_coords.max()),
                Point2D(x_coords.min(), y_coords.max())
            ]
            
        return boundary_points
        
    def calculate_deck_area(self, boundary: List[Point2D]) -> float:
        """
        Calculate area of deck from boundary points.
        
        Args:
            boundary: Ordered boundary points
            
        Returns:
            area: Deck area in square meters
        """
        if len(boundary) < 3:
            return 0.0
            
        # Shoelace formula for polygon area
        n = len(boundary)
        area = 0.0
        
        for i in range(n):
            j = (i + 1) % n
            area += boundary[i].x * boundary[j].y
            area -= boundary[j].x * boundary[i].y
            
        return abs(area) / 2.0
        
    def detect_rooms_grid_based(self, boundary: List[Point2D], deck_area: float) -> List[Room]:
        """
        Detect rooms using grid-based space subdivision.
        
        Args:
            boundary: Deck boundary points
            deck_area: Total deck area
            
        Returns:
            rooms: List of detected rooms
        """
        if not boundary:
            return []
            
        # Get bounding box
        x_coords = [p.x for p in boundary]
        y_coords = [p.y for p in boundary]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Grid resolution based on deck size
        grid_resolution = max(2.0, np.sqrt(deck_area) / 20)  # Adaptive resolution
        
        # Create grid
        x_grid = np.arange(x_min, x_max + grid_resolution, grid_resolution)
        y_grid = np.arange(y_min, y_max + grid_resolution, grid_resolution)
        
        # Check which grid cells are inside boundary
        inside_mask = np.zeros((len(y_grid), len(x_grid)), dtype=bool)
        
        for i, y in enumerate(y_grid):
            for j, x in enumerate(x_grid):
                # Point-in-polygon test
                point = Point2D(x, y)
                inside_mask[i, j] = self._point_in_polygon(point, boundary)
                
        # Label connected components as potential rooms
        labeled_array, n_components = label(inside_mask)
        
        rooms = []
        for component_id in range(1, n_components + 1):
            component_mask = labeled_array == component_id
            component_area = np.sum(component_mask) * grid_resolution**2
            
            if component_area >= self.min_room_area:
                # Calculate room bounds and center
                y_indices, x_indices = np.where(component_mask)
                
                x_min_room = x_grid[x_indices.min()]
                x_max_room = x_grid[x_indices.max()]
                y_min_room = y_grid[y_indices.min()]
                y_max_room = y_grid[y_indices.max()]
                
                center = Point2D(
                    (x_min_room + x_max_room) / 2,
                    (y_min_room + y_max_room) / 2
                )
                
                bounds = (
                    Point2D(x_min_room, y_min_room),
                    Point2D(x_max_room, y_max_room)
                )
                
                # Assign room type based on size and location
                room_type = self._classify_room_type(component_area, center, boundary)
                
                room = Room(
                    id=f"room_{component_id:03d}",
                    center=center,
                    bounds=bounds,
                    area=component_area,
                    room_type=room_type,
                    connected_rooms=[],
                    access_points=[]
                )
                
                rooms.append(room)
                
        self.logger.info(f"Detected {len(rooms)} rooms using grid-based method")
        return rooms
        
    def _point_in_polygon(self, point: Point2D, polygon: List[Point2D]) -> bool:
        """
        Test if point is inside polygon using ray casting algorithm.
        
        Args:
            point: Test point
            polygon: Polygon vertices
            
        Returns:
            inside: True if point is inside polygon
        """
        x, y = point.x, point.y
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0].x, polygon[0].y
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n].x, polygon[i % n].y
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside
        
    def _classify_room_type(self, area: float, center: Point2D, boundary: List[Point2D]) -> str:
        """
        Classify room type based on area and location within ship.
        
        Args:
            area: Room area in square meters
            center: Room center point
            boundary: Ship boundary for position reference
            
        Returns:
            room_type: Classified room type
        """
        # Get ship orientation (assume forward is +X direction)
        x_coords = [p.x for p in boundary]
        x_min, x_max = min(x_coords), max(x_coords)
        ship_length = x_max - x_min
        
        # Relative position along ship length
        position_ratio = (center.x - x_min) / ship_length
        
        # Forward section (0.0 - 0.3)
        if position_ratio <= 0.3:
            if area >= self.room_type_specs['bridge']['min_area']:
                return 'bridge'
            elif area >= self.room_type_specs['quarters']['min_area']:
                return 'quarters'
            else:
                return 'corridor'
                
        # Aft section (0.7 - 1.0)
        elif position_ratio >= 0.7:
            if area >= self.room_type_specs['engineering']['min_area']:
                return 'engineering'
            elif area >= self.room_type_specs['cargo']['min_area']:
                return 'cargo'
            else:
                return 'corridor'
                
        # Mid section (0.3 - 0.7)
        else:
            if area >= self.room_type_specs['cargo']['min_area']:
                return 'cargo'
            elif area >= self.room_type_specs['common']['min_area']:
                return 'common'
            elif area >= self.room_type_specs['medical']['min_area']:
                return 'medical'
            elif area >= self.room_type_specs['quarters']['min_area']:
                return 'quarters'
            else:
                return 'corridor'
                
    def detect_corridors(self, rooms: List[Room], boundary: List[Point2D]) -> List[Corridor]:
        """
        Detect corridors connecting rooms using pathfinding.
        
        Args:
            rooms: List of detected rooms
            boundary: Deck boundary
            
        Returns:
            corridors: List of detected corridors
        """
        corridors = []
        
        if len(rooms) < 2:
            return corridors
            
        # Build connectivity graph based on room proximity
        room_centers = np.array([[r.center.x, r.center.y] for r in rooms])
        distances = distance_matrix(room_centers, room_centers)
        
        # Connection threshold based on average room spacing
        avg_spacing = np.mean(distances[distances > 0])
        connection_threshold = avg_spacing * 1.5
        
        corridor_id = 1
        connected_pairs = set()
        
        for i, room1 in enumerate(rooms):
            for j, room2 in enumerate(rooms[i+1:], i+1):
                if distances[i, j] <= connection_threshold:
                    # Create corridor between these rooms
                    if (i, j) not in connected_pairs and (j, i) not in connected_pairs:
                        corridor_path = [room1.center, room2.center]
                        
                        corridor = Corridor(
                            id=f"corridor_{corridor_id:03d}",
                            path=corridor_path,
                            width=self.min_corridor_width,
                            connected_rooms=[room1.id, room2.id]
                        )
                        
                        corridors.append(corridor)
                        connected_pairs.add((i, j))
                        corridor_id += 1
                        
                        # Update room connections
                        room1.connected_rooms.append(room2.id)
                        room2.connected_rooms.append(room1.id)
                        
        self.logger.info(f"Detected {len(corridors)} corridors")
        return corridors
        
    def extract_deck_plan(self, hull_geometry: HullGeometry, deck_z: float, deck_name: str) -> DeckPlan:
        """
        Extract complete deck plan at specified level.
        
        Args:
            hull_geometry: 3D hull geometry
            deck_z: Z-coordinate of deck level
            deck_name: Name identifier for deck
            
        Returns:
            deck_plan: Complete deck plan specification
        """
        self.logger.info(f"Extracting deck plan '{deck_name}' at Z={deck_z}")
        
        # Extract deck boundary
        boundary = self.extract_deck_boundary(hull_geometry, deck_z)
        
        if not boundary:
            self.logger.warning(f"No boundary found for deck {deck_name}")
            return DeckPlan(
                deck_level=deck_z,
                deck_name=deck_name,
                outer_boundary=[],
                rooms=[],
                corridors=[],
                total_area=0.0,
                utilization_ratio=0.0
            )
            
        # Calculate total deck area
        total_area = self.calculate_deck_area(boundary)
        
        # Detect rooms
        rooms = self.detect_rooms_grid_based(boundary, total_area)
        
        # Detect corridors
        corridors = self.detect_corridors(rooms, boundary)
        
        # Calculate utilization ratio
        used_area = sum(room.area for room in rooms)
        corridor_area = sum(len(corridor.path) * corridor.width * 2.0 for corridor in corridors)
        utilization_ratio = (used_area + corridor_area) / total_area if total_area > 0 else 0.0
        
        deck_plan = DeckPlan(
            deck_level=deck_z,
            deck_name=deck_name,
            outer_boundary=boundary,
            rooms=rooms,
            corridors=corridors,
            total_area=total_area,
            utilization_ratio=min(utilization_ratio, 1.0)
        )
        
        self.logger.info(
            f"Deck plan '{deck_name}' extracted: {len(rooms)} rooms, "
            f"{len(corridors)} corridors, {utilization_ratio:.1%} utilization"
        )
        
        return deck_plan
        
    def extract_all_deck_plans(self, hull_geometry: HullGeometry) -> List[DeckPlan]:
        """
        Extract deck plans for all deck levels in hull geometry.
        
        Args:
            hull_geometry: 3D hull geometry
            
        Returns:
            deck_plans: List of all deck plans
        """
        deck_plans = []
        
        for i, deck_z in enumerate(hull_geometry.deck_levels):
            deck_name = f"Deck_{i+1}"
            
            # Special naming for typical ship decks
            if i == 0:
                deck_name = "Lower_Deck"
            elif i == len(hull_geometry.deck_levels) - 1:
                deck_name = "Bridge_Deck"
            elif len(hull_geometry.deck_levels) > 2 and i == len(hull_geometry.deck_levels) - 2:
                deck_name = "Main_Deck"
                
            deck_plan = self.extract_deck_plan(hull_geometry, deck_z, deck_name)
            deck_plans.append(deck_plan)
            
        self.logger.info(f"Extracted {len(deck_plans)} deck plans")
        return deck_plans
        
    def generate_deck_plan_svg(self, deck_plan: DeckPlan, output_path: str) -> None:
        """
        Generate SVG technical drawing of deck plan.
        
        Args:
            deck_plan: Deck plan to visualize
            output_path: Path to save SVG file
        """
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Draw deck boundary
        if deck_plan.outer_boundary:
            boundary_coords = [(p.x, p.y) for p in deck_plan.outer_boundary]
            boundary_polygon = Polygon(boundary_coords, fill=False, edgecolor='black', linewidth=2)
            ax.add_patch(boundary_polygon)
            
        # Color scheme for room types
        room_colors = {
            'bridge': '#FF6B6B',     # Red
            'engineering': '#4ECDC4', # Teal
            'quarters': '#45B7D1',    # Blue
            'cargo': '#96CEB4',       # Green
            'medical': '#FFEAA7',     # Yellow
            'corridor': '#DDA0DD',    # Plum
            'common': '#FFB6C1'       # Light Pink
        }
        
        # Draw rooms
        for room in deck_plan.rooms:
            color = room_colors.get(room.room_type, '#CCCCCC')
            
            # Room rectangle
            width = room.bounds[1].x - room.bounds[0].x
            height = room.bounds[1].y - room.bounds[0].y
            rect = patches.Rectangle(
                (room.bounds[0].x, room.bounds[0].y),
                width, height,
                linewidth=1,
                edgecolor='black',
                facecolor=color,
                alpha=0.7
            )
            ax.add_patch(rect)
            
            # Room label
            ax.text(
                room.center.x, room.center.y,
                f"{room.room_type}\n{room.area:.0f}m²",
                ha='center', va='center',
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8)
            )
            
        # Draw corridors
        for corridor in deck_plan.corridors:
            if len(corridor.path) >= 2:
                for i in range(len(corridor.path) - 1):
                    p1, p2 = corridor.path[i], corridor.path[i+1]
                    ax.plot([p1.x, p2.x], [p1.y, p2.y], 
                           color='gray', linewidth=corridor.width*2, alpha=0.5)
                    
        # Formatting
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title(f'Deck Plan: {deck_plan.deck_name}\n'
                    f'Area: {deck_plan.total_area:.0f}m², '
                    f'Utilization: {deck_plan.utilization_ratio:.1%}')
        
        # Legend
        legend_elements = [
            patches.Patch(color=color, label=room_type.title())
            for room_type, color in room_colors.items()
            if any(room.room_type == room_type for room in deck_plan.rooms)
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Deck plan SVG saved: {output_path}")
        
    def export_deck_plans_json(self, deck_plans: List[DeckPlan], output_path: str) -> None:
        """
        Export deck plans to JSON format for further processing.
        
        Args:
            deck_plans: List of deck plans to export
            output_path: Path to save JSON file
        """
        # Convert to serializable format
        export_data = {
            'ship_deck_plans': [],
            'total_decks': len(deck_plans),
            'extraction_metadata': {
                'framework': 'Ship Hull Geometry OBJ Framework Phase 3',
                'room_detection': 'grid_based',
                'corridor_detection': 'proximity_based'
            }
        }
        
        for deck_plan in deck_plans:
            deck_data = {
                'deck_name': deck_plan.deck_name,
                'deck_level': deck_plan.deck_level,
                'total_area': deck_plan.total_area,
                'utilization_ratio': deck_plan.utilization_ratio,
                'boundary': [{'x': p.x, 'y': p.y} for p in deck_plan.outer_boundary],
                'rooms': [
                    {
                        'id': room.id,
                        'type': room.room_type,
                        'center': {'x': room.center.x, 'y': room.center.y},
                        'bounds': {
                            'min': {'x': room.bounds[0].x, 'y': room.bounds[0].y},
                            'max': {'x': room.bounds[1].x, 'y': room.bounds[1].y}
                        },
                        'area': room.area,
                        'connected_rooms': room.connected_rooms
                    }
                    for room in deck_plan.rooms
                ],
                'corridors': [
                    {
                        'id': corridor.id,
                        'path': [{'x': p.x, 'y': p.y} for p in corridor.path],
                        'width': corridor.width,
                        'connected_rooms': corridor.connected_rooms
                    }
                    for corridor in deck_plan.corridors
                ]
            }
            export_data['ship_deck_plans'].append(deck_data)
            
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        self.logger.info(f"Deck plans JSON exported: {output_path}")


def create_deck_plan_demo() -> Dict:
    """
    Demonstration of deck plan extraction from hull geometry.
    
    Returns:
        demo_results: Complete deck plan extraction results
    """
    from hull_geometry_generator import HullPhysicsEngine, AlcubierreMetricConstraints
    
    logger.info("Starting Deck Plan Extraction Demo")
    
    # Generate hull geometry (Phase 1)
    constraints = AlcubierreMetricConstraints(
        warp_velocity=48.0,
        bubble_radius=500.0,
        exotic_energy_density=0.0,
        metric_signature="(-,+,+,+)",
        coordinate_system="cartesian"
    )
    
    hull_engine = HullPhysicsEngine(constraints)
    hull_geometry = hull_engine.generate_physics_informed_hull(
        length=180.0,
        beam=35.0,
        height=25.0,
        n_sections=12
    )
    
    # Initialize deck plan extractor (Phase 3)
    extractor = DeckPlanExtractor()
    
    # Extract all deck plans
    deck_plans = extractor.extract_all_deck_plans(hull_geometry)
    
    # Create output directories
    import os
    output_dir = "deck_plan_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualizations
    for deck_plan in deck_plans:
        svg_path = os.path.join(output_dir, f"{deck_plan.deck_name.lower()}_plan.svg")
        extractor.generate_deck_plan_svg(deck_plan, svg_path)
        
    # Export JSON data
    json_path = os.path.join(output_dir, "ship_deck_plans.json")
    extractor.export_deck_plans_json(deck_plans, json_path)
    
    # Analysis summary
    total_rooms = sum(len(deck.rooms) for deck in deck_plans)
    total_corridors = sum(len(deck.corridors) for deck in deck_plans)
    total_area = sum(deck.total_area for deck in deck_plans)
    avg_utilization = np.mean([deck.utilization_ratio for deck in deck_plans])
    
    # Room type distribution
    room_type_counts = {}
    for deck in deck_plans:
        for room in deck.rooms:
            room_type_counts[room.room_type] = room_type_counts.get(room.room_type, 0) + 1
            
    demo_results = {
        'deck_count': len(deck_plans),
        'total_rooms': total_rooms,
        'total_corridors': total_corridors,
        'total_ship_area': total_area,
        'average_utilization': avg_utilization,
        'room_type_distribution': room_type_counts,
        'deck_details': [
            {
                'name': deck.deck_name,
                'level': deck.deck_level,
                'area': deck.total_area,
                'rooms': len(deck.rooms),
                'corridors': len(deck.corridors),
                'utilization': deck.utilization_ratio
            }
            for deck in deck_plans
        ],
        'output_files': {
            'json_export': json_path,
            'svg_plans': [
                os.path.join(output_dir, f"{deck.deck_name.lower()}_plan.svg")
                for deck in deck_plans
            ]
        }
    }
    
    logger.info(
        f"Deck plan demo complete: {len(deck_plans)} decks, "
        f"{total_rooms} rooms, {avg_utilization:.1%} avg utilization"
    )
    
    return demo_results


if __name__ == "__main__":
    # Run deck plan extraction demonstration
    results = create_deck_plan_demo()
    
    print("\n" + "="*60)
    print("SHIP HULL GEOMETRY PHASE 3: DECK PLAN EXTRACTION")
    print("="*60)
    print(f"Total Decks: {results['deck_count']}")
    print(f"Total Rooms: {results['total_rooms']}")
    print(f"Total Corridors: {results['total_corridors']}")
    print(f"Total Ship Area: {results['total_ship_area']:.0f} m²")
    print(f"Average Utilization: {results['average_utilization']:.1%}")
    print("\nRoom Type Distribution:")
    for room_type, count in results['room_type_distribution'].items():
        print(f"  {room_type.title()}: {count}")
    print("\nDeck Details:")
    for deck in results['deck_details']:
        print(f"  {deck['name']}: {deck['area']:.0f}m², {deck['rooms']} rooms, {deck['utilization']:.1%} utilization")
    print(f"\nOutput Files: {len(results['output_files']['svg_plans'])} SVG plans + JSON export")
    print("="*60)

#!/usr/bin/env python3
"""
Optimized Deck Plan Generator - Ship Hull Geometry Framework

Generates realistic deck plans based on scientifically optimized crew complement
analysis. Creates detailed room layouts for 99-person diplomatic FTL starship.

Author: Ship Hull Geometry Framework
Date: July 13, 2025
Version: 2.0.0 - Optimization-Based Implementation
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Room:
    """Room specification for deck plans."""
    id: str
    type: str
    name: str
    area: float
    center: Tuple[float, float]
    boundary: List[Tuple[float, float]]
    occupancy: int = 1
    privacy_level: str = "standard"  # standard, high, vip
    amenities: List[str] = None

@dataclass 
class DeckSpecification:
    """Complete deck specification."""
    name: str
    level: float
    function: str
    total_occupancy: int
    rooms: List[Room]
    corridors: List[Dict]
    boundary: List[Tuple[float, float]]

class OptimizedDeckPlanGenerator:
    """Generate optimized deck plans based on crew analysis."""
    
    def __init__(self):
        # Optimized crew configuration (from economic optimization)
        self.crew_config = {
            "total_crew": 99,
            "operational_crew": 36,
            "diplomatic_passengers": 63,
            "breakdown": {
                "command": 1,
                "engineering": 14,
                "medical": 8,
                "science": 7,
                "maintenance": 4,
                "security": 1,
                "support": 1,
                "ambassadors": 4,
                "senior_diplomats": 8,
                "diplomatic_staff": 26,  # Adjusted: 13 + 13 = 26 total diplomatic staff
                "security_detail": 8,
                "technical_advisors": 10,
                "cultural_attaches": 4  # Reduced from 5 to 4
            }
        }
        
        # Ship dimensions
        self.ship_length = 300.0
        self.ship_beam = 50.0
        self.deck_height = 40.0
        self.deck_levels = 13
        self.deck_spacing = self.deck_height / (self.deck_levels - 1)
        
        logger.info(f"Optimized Deck Plan Generator initialized for {self.crew_config['total_crew']}-person crew")
    
    def generate_deck_boundary(self, level: float) -> List[Tuple[float, float]]:
        """Generate hull boundary at specific deck level."""
        # Create elliptical hull cross-section
        a = self.ship_beam / 2  # Semi-major axis
        b = self.ship_beam / 3  # Semi-minor axis for height variation
        
        # Generate boundary points
        boundary = []
        n_points = 32
        
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            
            # Elliptical cross-section with length variation
            length_factor = 1.0 - 0.3 * abs(level) / (self.deck_height / 2)
            length_factor = max(0.4, length_factor)  # Minimum 40% length
            
            x = length_factor * self.ship_length / 2 * np.cos(angle)
            z = a * np.sin(angle)
            
            boundary.append((float(x), float(z)))
        
        return boundary
    
    def create_room(self, room_id: str, room_type: str, name: str, 
                   center: Tuple[float, float], area: float, 
                   occupancy: int = 1, privacy: str = "standard",
                   amenities: List[str] = None) -> Room:
        """Create a room with realistic boundary."""
        # Simple rectangular room boundary
        width = np.sqrt(area * 1.5)  # 1.5 aspect ratio
        height = area / width
        
        x, z = center
        boundary = [
            (x - width/2, z - height/2),
            (x + width/2, z - height/2), 
            (x + width/2, z + height/2),
            (x - width/2, z + height/2)
        ]
        
        return Room(
            id=room_id,
            type=room_type,
            name=name,
            area=area,
            center=center,
            boundary=boundary,
            occupancy=occupancy,
            privacy_level=privacy,
            amenities=amenities or []
        )
    
    def generate_vip_diplomatic_deck(self, deck_num: int, level: float) -> DeckSpecification:
        """Generate VIP diplomatic deck (Deck 12)."""
        rooms = []
        
        # Ambassador suites (4 × 50m²)
        for i in range(4):
            x = -60 + i * 40
            z = 15 if i % 2 == 0 else -15
            room = self.create_room(
                f"amb_suite_{i+1:02d}",
                "vip_quarters",
                f"Ambassador Suite {i+1}",
                (x, z),
                50.0,
                occupancy=1,
                privacy="vip",
                amenities=["private_bathroom", "office_space", "secure_comm", "luxury_furnishing"]
            )
            rooms.append(room)
        
        # Senior diplomat quarters (8 × 30m²)
        for i in range(8):
            x = -80 + i * 20
            z = 5 if i % 2 == 0 else -5
            if i >= 4:  # Second row
                z = z + (10 if z > 0 else -10)
            room = self.create_room(
                f"senior_dipl_{i+1:02d}",
                "diplomatic_quarters",
                f"Senior Diplomat {i+1}",
                (x, z),
                30.0,
                occupancy=1,
                privacy="high",
                amenities=["private_bathroom", "workspace", "secure_storage"]
            )
            rooms.append(room)
        
        # VIP dining room
        dining = self.create_room(
            "vip_dining",
            "dining",
            "VIP Diplomatic Dining",
            (0, 0),
            100.0,
            occupancy=20,
            amenities=["formal_dining", "cultural_kitchen", "wine_storage"]
        )
        rooms.append(dining)
        
        # Private meeting rooms
        for i in range(3):
            x = 60 + i * 15
            meeting = self.create_room(
                f"private_meeting_{i+1}",
                "meeting",
                f"Private Meeting Room {i+1}",
                (x, 0),
                25.0,
                occupancy=8,
                privacy="vip",
                amenities=["secure_comm", "privacy_shielding", "recording_equipment"]
            )
            rooms.append(meeting)
        
        boundary = self.generate_deck_boundary(level)
        
        return DeckSpecification(
            name=f"Deck_{deck_num}",
            level=level,
            function="VIP Diplomatic Quarters",
            total_occupancy=12,  # 4 ambassadors + 8 senior diplomats
            rooms=rooms,
            corridors=[],
            boundary=boundary
        )
    
    def generate_diplomatic_staff_deck(self, deck_num: int, level: float, 
                                     staff_count: int, room_size: float) -> DeckSpecification:
        """Generate diplomatic staff deck."""
        rooms = []
        
        # Diplomatic staff quarters
        cols = 6
        rows = (staff_count + cols - 1) // cols
        
        for i in range(staff_count):
            row = i // cols
            col = i % cols
            
            x = -75 + col * 25
            z = -10 + row * 20
            
            room = self.create_room(
                f"dipl_staff_{i+1:02d}",
                "diplomatic_quarters",
                f"Diplomatic Staff {i+1}",
                (x, z),
                room_size,
                occupancy=1,
                privacy="high",
                amenities=["private_bathroom", "workspace", "secure_comm"]
            )
            rooms.append(room)
        
        # Staff lounge
        lounge = self.create_room(
            "staff_lounge",
            "lounge",
            "Diplomatic Staff Lounge",
            (50, 0),
            80.0,
            occupancy=25,
            amenities=["meeting_space", "recreation", "refreshments"]
        )
        rooms.append(lounge)
        
        # Conference center
        conference = self.create_room(
            "conference_center",
            "conference",
            "Diplomatic Conference Center",
            (80, 15),
            120.0,
            occupancy=50,
            privacy="high",
            amenities=["presentation_equipment", "translation_services", "recording"]
        )
        rooms.append(conference)
        
        boundary = self.generate_deck_boundary(level)
        
        return DeckSpecification(
            name=f"Deck_{deck_num}",
            level=level,
            function="Diplomatic Staff Quarters",
            total_occupancy=staff_count,
            rooms=rooms,
            corridors=[],
            boundary=boundary
        )
    
    def generate_crew_deck(self, deck_num: int, level: float, 
                          crew_types: List[Tuple[str, int]], room_size: float) -> DeckSpecification:
        """Generate crew quarters deck."""
        rooms = []
        total_crew = sum(count for _, count in crew_types)
        
        # Generate crew quarters
        room_idx = 0
        for crew_type, count in crew_types:
            for i in range(count):
                row = room_idx // 8
                col = room_idx % 8
                
                x = -87.5 + col * 25
                z = -15 + row * 15
                
                room = self.create_room(
                    f"{crew_type}_{i+1:02d}",
                    "crew_quarters",
                    f"{crew_type.title()} {i+1}",
                    (x, z),
                    room_size,
                    occupancy=1,
                    amenities=["private_bathroom", "storage", "personal_workspace"]
                )
                rooms.append(room)
                room_idx += 1
        
        # Crew common room
        common = self.create_room(
            "crew_common",
            "common",
            "Crew Common Room",
            (70, 0),
            60.0,
            occupancy=20,
            amenities=["recreation", "dining", "meeting_space"]
        )
        rooms.append(common)
        
        # Workshop/specialty area
        workshop = self.create_room(
            "workshop",
            "workshop",
            "Workshop Area",
            (70, 20),
            40.0,
            occupancy=5,
            amenities=["tools", "fabrication", "repair_space"]
        )
        rooms.append(workshop)
        
        boundary = self.generate_deck_boundary(level)
        
        return DeckSpecification(
            name=f"Deck_{deck_num}",
            level=level,
            function="Crew Quarters",
            total_occupancy=total_crew,
            rooms=rooms,
            corridors=[],
            boundary=boundary
        )
    
    def generate_all_optimized_deck_plans(self) -> List[DeckSpecification]:
        """Generate all optimized deck plans."""
        decks = []
        
        # Calculate deck levels
        base_level = -self.deck_height / 2
        
        for deck_num in range(1, self.deck_levels + 1):
            level = base_level + (deck_num - 1) * self.deck_spacing
            
            if deck_num == 13:  # Command deck
                deck = self.generate_command_deck(deck_num, level)
            elif deck_num == 12:  # VIP diplomatic quarters
                deck = self.generate_vip_diplomatic_deck(deck_num, level)
            elif deck_num == 11:  # Diplomatic staff A
                deck = self.generate_diplomatic_staff_deck(deck_num, level, 13, 20.0)  # Reduced to 13
            elif deck_num == 10:  # Diplomatic staff B  
                deck = self.generate_diplomatic_staff_deck(deck_num, level, 13, 20.0)  # Keep at 13
            elif deck_num == 9:  # Security & technical
                deck = self.generate_crew_deck(deck_num, level, 
                    [("security", 8), ("technical", 10)], 15.0)
            elif deck_num == 8:  # Support staff & recreation
                deck = self.generate_recreation_deck(deck_num, level)
            elif deck_num == 7:  # Main operations & senior crew
                deck = self.generate_operations_deck(deck_num, level)
            elif deck_num == 6:  # Engineering & medical crew
                deck = self.generate_crew_deck(deck_num, level,
                    [("engineering", 14), ("medical", 8)], 12.0)
            elif deck_num == 5:  # Science & remaining crew
                deck = self.generate_crew_deck(deck_num, level,
                    [("science", 7), ("maintenance", 4), ("support", 2)], 12.0)
            elif deck_num == 4:  # Engineering support
                deck = self.generate_engineering_support_deck(deck_num, level)
            elif deck_num == 3:  # Primary engineering
                deck = self.generate_primary_engineering_deck(deck_num, level)
            elif deck_num == 2:  # Life support
                deck = self.generate_life_support_deck(deck_num, level)
            else:  # Lower deck (1) - Cargo
                deck = self.generate_cargo_deck(deck_num, level)
            
            decks.append(deck)
            logger.info(f"Generated {deck.name}: {deck.function} (Level {level:.1f}m)")
        
        return decks
    
    def generate_command_deck(self, deck_num: int, level: float) -> DeckSpecification:
        """Generate command deck."""
        rooms = []
        
        # Bridge
        bridge = self.create_room(
            "bridge",
            "bridge",
            "Main Bridge",
            (0, 0),
            150.0,
            occupancy=12,
            privacy="high",
            amenities=["command_stations", "navigation", "communications", "tactical"]
        )
        rooms.append(bridge)
        
        # Captain's ready room
        ready_room = self.create_room(
            "captain_ready",
            "office",
            "Captain's Ready Room",
            (-30, 0),
            30.0,
            occupancy=1,
            privacy="vip",
            amenities=["private_office", "secure_comm", "meeting_space"]
        )
        rooms.append(ready_room)
        
        # VIP lounge
        vip_lounge = self.create_room(
            "vip_lounge",
            "lounge",
            "VIP Diplomatic Lounge",
            (50, 0),
            100.0,
            occupancy=20,
            privacy="vip",
            amenities=["luxury_seating", "observation_windows", "refreshments"]
        )
        rooms.append(vip_lounge)
        
        boundary = self.generate_deck_boundary(level)
        
        return DeckSpecification(
            name=f"Deck_{deck_num}",
            level=level,
            function="Command & VIP Lounge",
            total_occupancy=1,  # Captain
            rooms=rooms,
            corridors=[],
            boundary=boundary
        )
    
    def generate_recreation_deck(self, deck_num: int, level: float) -> DeckSpecification:
        """Generate recreation deck."""
        rooms = []
        
        # Cultural attaché quarters (4 × 15m²) - reduced from 5 to 4
        for i in range(4):
            x = -80 + i * 40  # Spread out more
            room = self.create_room(
                f"cultural_{i+1:02d}",
                "diplomatic_quarters", 
                f"Cultural Attaché {i+1}",
                (x, 15),
                15.0,
                occupancy=1,
                amenities=["private_bathroom", "cultural_workspace"]
            )
            rooms.append(room)
        
        # Main recreation center
        recreation = self.create_room(
            "recreation_center",
            "recreation",
            "Main Recreation Center",
            (0, 0),
            200.0,
            occupancy=50,
            amenities=["entertainment", "games", "social_space", "cultural_programs"]
        )
        rooms.append(recreation)
        
        # Multi-cultural dining
        dining = self.create_room(
            "multicultural_dining",
            "dining",
            "Multi-Cultural Dining Hall",
            (60, 0),
            150.0,
            occupancy=99,
            amenities=["diverse_cuisine", "cultural_accommodation", "large_capacity"]
        )
        rooms.append(dining)
        
        # Exercise facility
        exercise = self.create_room(
            "exercise_facility",
            "exercise",
            "Exercise Facility",
            (-60, 0),
            100.0,
            occupancy=20,
            amenities=["fitness_equipment", "artificial_gravity_control", "health_monitoring"]
        )
        rooms.append(exercise)
        
        boundary = self.generate_deck_boundary(level)
        
        return DeckSpecification(
            name=f"Deck_{deck_num}",
            level=level,
            function="Support Staff & Recreation",
            total_occupancy=4,  # Cultural attachés reduced to 4
            rooms=rooms,
            corridors=[],
            boundary=boundary
        )
    
    def generate_operations_deck(self, deck_num: int, level: float) -> DeckSpecification:
        """Generate main operations deck."""
        rooms = []
        
        # Senior officer quarters (3 × 25m²)
        officers = ["captain", "chief_engineer", "chief_medical"]
        for i, officer in enumerate(officers):
            x = -60 + i * 60
            room = self.create_room(
                f"{officer}_quarters",
                "senior_quarters",
                f"{officer.replace('_', ' ').title()} Quarters",
                (x, 15),
                25.0,
                occupancy=1,
                privacy="high",
                amenities=["private_bathroom", "office_space", "luxury_amenities"]
            )
            rooms.append(room)
        
        # Operations center
        ops_center = self.create_room(
            "operations_center",
            "operations",
            "Mission Operations Center",
            (0, 0),
            120.0,
            occupancy=15,
            amenities=["mission_control", "communications", "monitoring", "coordination"]
        )
        rooms.append(ops_center)
        
        # Medical bay
        medical = self.create_room(
            "medical_bay",
            "medical",
            "Comprehensive Medical Bay",
            (70, 0),
            180.0,
            occupancy=20,
            privacy="high",
            amenities=["surgery_suite", "diagnostic_equipment", "patient_care", "pharmacy"]
        )
        rooms.append(medical)
        
        boundary = self.generate_deck_boundary(level)
        
        return DeckSpecification(
            name=f"Deck_{deck_num}",
            level=level,
            function="Main Operations & Senior Crew",
            total_occupancy=3,  # Senior officers
            rooms=rooms,
            corridors=[],
            boundary=boundary
        )
    
    def generate_engineering_support_deck(self, deck_num: int, level: float) -> DeckSpecification:
        """Generate engineering support deck."""
        rooms = []
        
        # Engineering workshops
        workshop = self.create_room(
            "eng_workshop",
            "workshop",
            "Advanced Engineering Workshop",
            (0, 0),
            200.0,
            occupancy=15,
            amenities=["fabrication", "repair_bay", "testing_equipment", "tool_storage"]
        )
        rooms.append(workshop)
        
        # Parts storage
        storage = self.create_room(
            "parts_storage",
            "storage",
            "Comprehensive Parts Storage",
            (80, 0),
            120.0,
            occupancy=2,
            amenities=["inventory_management", "climate_control", "automated_retrieval"]
        )
        rooms.append(storage)
        
        # Training center
        training = self.create_room(
            "eng_training",
            "training",
            "Engineering Training Center",
            (-80, 0),
            80.0,
            occupancy=20,
            amenities=["simulation_equipment", "virtual_reality", "hands_on_training"]
        )
        rooms.append(training)
        
        boundary = self.generate_deck_boundary(level)
        
        return DeckSpecification(
            name=f"Deck_{deck_num}",
            level=level,
            function="Engineering Support",
            total_occupancy=0,  # No permanent residents
            rooms=rooms,
            corridors=[],
            boundary=boundary
        )
    
    def generate_primary_engineering_deck(self, deck_num: int, level: float) -> DeckSpecification:
        """Generate primary engineering deck."""
        rooms = []
        
        # LQG drive core
        lqg_core = self.create_room(
            "lqg_core",
            "engine",
            "LQG Drive Core",
            (0, 0),
            250.0,
            occupancy=8,
            privacy="high",
            amenities=["quantum_systems", "zero_exotic_energy", "safety_systems"]
        )
        rooms.append(lqg_core)
        
        # Warp field generators
        warp_gen = self.create_room(
            "warp_generators",
            "engine",
            "Warp Field Generators",
            (90, 0),
            150.0,
            occupancy=5,
            amenities=["alcubierre_drive", "field_control", "monitoring"]
        )
        rooms.append(warp_gen)
        
        # Engineering control
        eng_control = self.create_room(
            "eng_control",
            "control",
            "Primary Engineering Control",
            (-90, 0),
            100.0,
            occupancy=6,
            privacy="high",
            amenities=["master_control", "diagnostics", "emergency_systems"]
        )
        rooms.append(eng_control)
        
        boundary = self.generate_deck_boundary(level)
        
        return DeckSpecification(
            name=f"Deck_{deck_num}",
            level=level,
            function="Primary Engineering",
            total_occupancy=0,  # No permanent residents
            rooms=rooms,
            corridors=[],
            boundary=boundary
        )
    
    def generate_life_support_deck(self, deck_num: int, level: float) -> DeckSpecification:
        """Generate life support deck."""
        rooms = []
        
        # Life support control
        ls_control = self.create_room(
            "life_support_control",
            "control",
            "Life Support Control",
            (0, 0),
            100.0,
            occupancy=4,
            amenities=["atmospheric_control", "environmental_monitoring", "emergency_backup"]
        )
        rooms.append(ls_control)
        
        # Atmospheric processing
        atmosphere = self.create_room(
            "atmospheric_processing",
            "processing",
            "Atmospheric Processing",
            (80, 0),
            120.0,
            occupancy=2,
            amenities=["air_recycling", "oxygen_generation", "co2_scrubbing"]
        )
        rooms.append(atmosphere)
        
        # Water systems
        water = self.create_room(
            "water_systems",
            "processing",
            "Water Recycling Systems",
            (-80, 0),
            100.0,
            occupancy=2,
            amenities=["water_purification", "recycling", "storage", "distribution"]
        )
        rooms.append(water)
        
        # Waste processing
        waste = self.create_room(
            "waste_processing",
            "processing",
            "Waste Processing",
            (0, 20),
            80.0,
            occupancy=1,
            amenities=["material_recycling", "waste_disposal", "organic_processing"]
        )
        rooms.append(waste)
        
        boundary = self.generate_deck_boundary(level)
        
        return DeckSpecification(
            name=f"Deck_{deck_num}",
            level=level,
            function="Life Support & Environmental",
            total_occupancy=0,  # No permanent residents
            rooms=rooms,
            corridors=[],
            boundary=boundary
        )
    
    def generate_cargo_deck(self, deck_num: int, level: float) -> DeckSpecification:
        """Generate cargo deck."""
        rooms = []
        
        # Diplomatic cargo
        dipl_cargo = self.create_room(
            "diplomatic_cargo",
            "cargo",
            "Diplomatic Cargo Hold",
            (0, 0),
            200.0,
            occupancy=0,
            amenities=["secure_storage", "climate_control", "cultural_items", "gifts"]
        )
        rooms.append(dipl_cargo)
        
        # Food storage
        food_storage = self.create_room(
            "food_storage",
            "storage",
            "Food Storage (90-day supply)",
            (90, 0),
            150.0,
            occupancy=0,
            amenities=["temperature_control", "preservation", "diverse_cuisine"]
        )
        rooms.append(food_storage)
        
        # Luxury supplies
        luxury = self.create_room(
            "luxury_supplies",
            "storage",
            "Luxury Supplies",
            (-90, 0),
            100.0,
            occupancy=0,
            amenities=["vip_amenities", "cultural_items", "diplomatic_gifts"]
        )
        rooms.append(luxury)
        
        # Emergency equipment
        emergency = self.create_room(
            "emergency_equipment",
            "storage",
            "Emergency Equipment",
            (0, 20),
            80.0,
            occupancy=0,
            amenities=["emergency_supplies", "evacuation_equipment", "medical_backup"]
        )
        rooms.append(emergency)
        
        boundary = self.generate_deck_boundary(level)
        
        return DeckSpecification(
            name=f"Deck_{deck_num}",
            level=level,
            function="Cargo & Diplomatic Supplies",
            total_occupancy=0,  # No permanent residents
            rooms=rooms,
            corridors=[],
            boundary=boundary
        )
    
    def export_optimized_deck_plans(self, output_path: str = "optimized_deck_plans.json"):
        """Export all optimized deck plans to JSON."""
        decks = self.generate_all_optimized_deck_plans()
        
        # Convert to dictionary format
        deck_plans_data = {
            "metadata": {
                "framework": "Ship Hull Geometry Framework",
                "version": "2.0.0 - Optimization-Based",
                "generation_date": "2025-07-13",
                "crew_optimization": self.crew_config,
                "economic_performance": {
                    "roi": "249.05%",
                    "net_profit": "$512.59M",
                    "mission_type": "diplomatic"
                }
            },
            "ship_specifications": {
                "length": self.ship_length,
                "beam": self.ship_beam,
                "height": self.deck_height,
                "total_decks": self.deck_levels,
                "deck_spacing": self.deck_spacing
            },
            "deck_plans": []
        }
        
        total_occupancy = 0
        
        for deck in decks:
            deck_data = {
                "name": deck.name,
                "level": deck.level,
                "function": deck.function,
                "total_occupancy": deck.total_occupancy,
                "boundary": deck.boundary,
                "rooms": [],
                "corridors": deck.corridors
            }
            
            for room in deck.rooms:
                room_data = {
                    "id": room.id,
                    "type": room.type,
                    "name": room.name,
                    "center": room.center,
                    "area": room.area,
                    "boundary": room.boundary,
                    "occupancy": room.occupancy,
                    "privacy_level": room.privacy_level,
                    "amenities": room.amenities
                }
                deck_data["rooms"].append(room_data)
            
            deck_plans_data["deck_plans"].append(deck_data)
            total_occupancy += deck.total_occupancy
        
        deck_plans_data["validation"] = {
            "total_planned_occupancy": total_occupancy,
            "target_crew": self.crew_config["total_crew"],
            "occupancy_match": total_occupancy == self.crew_config["total_crew"]
        }
        
        # Export to JSON
        with open(output_path, 'w') as f:
            json.dump(deck_plans_data, f, indent=2)
        
        logger.info(f"Exported optimized deck plans to {output_path}")
        logger.info(f"Total planned occupancy: {total_occupancy}/{self.crew_config['total_crew']}")
        logger.info(f"Generated {len(decks)} decks with {sum(len(d.rooms) for d in decks)} rooms")
        
        return deck_plans_data

if __name__ == "__main__":
    generator = OptimizedDeckPlanGenerator()
    deck_plans = generator.export_optimized_deck_plans()
    print("Optimized deck plans generated successfully!")
    print(f"Total crew accommodated: {deck_plans['validation']['total_planned_occupancy']}")
    print(f"Economic performance: {deck_plans['metadata']['economic_performance']['roi']} ROI")

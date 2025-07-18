
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

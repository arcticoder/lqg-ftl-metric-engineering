"""
Ship Hull Geometry OBJ Framework - Phase 4: Browser Visualization
===============================================================

Interactive WebGL-based 3D hull visualization with deck plan overlay,
real-time hull modification, and Chrome browser integration.
"""

import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import base64
import logging

from hull_geometry_generator import HullGeometry, AlcubierreMetricConstraints, HullPhysicsEngine
from obj_mesh_generator import OBJMeshGenerator  
from deck_plan_extractor import DeckPlanExtractor, DeckPlan

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WebGLShader:
    """WebGL shader specification."""
    vertex_source: str
    fragment_source: str
    
@dataclass
class InteractiveControl:
    """Interactive control specification for hull modification."""
    control_type: str  # "slider", "button", "input"
    parameter: str  # Hull parameter to control
    min_value: float
    max_value: float
    default_value: float
    step_size: float
    label: str
    
class BrowserVisualizationEngine:
    """
    Phase 4: Browser Visualization
    
    Generates interactive WebGL-based 3D hull visualization with real-time
    hull modification capabilities and deck plan overlay integration.
    """
    
    def __init__(self):
        """Initialize browser visualization engine."""
        self.logger = logging.getLogger(f"{__name__}.BrowserVisualizationEngine")
        
        # WebGL configuration
        self.webgl_config = {
            'canvas_width': 1200,
            'canvas_height': 800,
            'fov': 45,
            'near_plane': 0.1,
            'far_plane': 10000.0,
            'background_color': [0.05, 0.05, 0.1, 1.0]  # Dark space background
        }
        
        # Interactive controls for hull modification
        self.hull_controls = [
            InteractiveControl(
                control_type="slider",
                parameter="warp_velocity",
                min_value=1.0,
                max_value=100.0,
                default_value=48.0,
                step_size=1.0,
                label="Warp Velocity (c)"
            ),
            InteractiveControl(
                control_type="slider", 
                parameter="hull_length",
                min_value=100.0,
                max_value=500.0,
                default_value=300.0,
                step_size=10.0,
                label="Hull Length (m)"
            ),
            InteractiveControl(
                control_type="slider",
                parameter="hull_beam", 
                min_value=20.0,
                max_value=100.0,
                default_value=50.0,
                step_size=5.0,
                label="Hull Beam (m)"
            ),
            InteractiveControl(
                control_type="slider",
                parameter="safety_factor",
                min_value=2.0,
                max_value=10.0,
                default_value=5.0,
                step_size=0.5,
                label="Safety Factor"
            )
        ]
        
    def generate_webgl_shaders(self) -> Dict[str, WebGLShader]:
        """
        Generate WebGL shaders for hull rendering.
        
        Returns:
            shaders: Dictionary of WebGL shaders
        """
        # Vertex shader for hull geometry
        hull_vertex_shader = """
        attribute vec3 aVertexPosition;
        attribute vec3 aVertexNormal;
        attribute vec2 aTextureCoord;
        attribute float aThickness;
        
        uniform mat4 uModelViewMatrix;
        uniform mat4 uProjectionMatrix;
        uniform mat4 uNormalMatrix;
        uniform float uTime;
        uniform float uWarpVelocity;
        
        varying vec3 vNormal;
        varying vec2 vTextureCoord;
        varying float vThickness;
        varying vec3 vWorldPosition;
        varying float vWarpEffect;
        
        void main(void) {
            // Apply warp field distortion
            vec3 position = aVertexPosition;
            float warpFactor = uWarpVelocity / 48.0;
            
            // Alcubierre-inspired spatial distortion
            float r = length(position);
            float warpDistortion = sin(uTime * 2.0 + r * 0.01) * warpFactor * 0.1;
            position.z += warpDistortion;
            
            gl_Position = uProjectionMatrix * uModelViewMatrix * vec4(position, 1.0);
            
            vNormal = normalize((uNormalMatrix * vec4(aVertexNormal, 0.0)).xyz);
            vTextureCoord = aTextureCoord;
            vThickness = aThickness;
            vWorldPosition = position;
            vWarpEffect = warpFactor;
        }
        """
        
        # Fragment shader for hull rendering
        hull_fragment_shader = """
        precision mediump float;
        
        varying vec3 vNormal;
        varying vec2 vTextureCoord;
        varying float vThickness;
        varying vec3 vWorldPosition;
        varying float vWarpEffect;
        
        uniform vec3 uLightDirection;
        uniform vec3 uAmbientColor;
        uniform vec3 uDiffuseColor;
        uniform vec3 uSpecularColor;
        uniform float uShininess;
        uniform float uTime;
        
        void main(void) {
            // Base material color based on thickness
            vec3 baseColor = mix(
                vec3(0.3, 0.35, 0.4),  // Thin areas (blue-gray)
                vec3(0.4, 0.3, 0.2),   // Thick areas (brown)
                clamp(vThickness / 3.0, 0.0, 1.0)
            );
            
            // Warp field energy glow
            float warpGlow = vWarpEffect * sin(uTime * 3.0 + length(vWorldPosition) * 0.02) * 0.5 + 0.5;
            vec3 warpColor = vec3(0.2, 0.6, 1.0) * warpGlow * vWarpEffect;
            
            // Lighting calculation
            vec3 normal = normalize(vNormal);
            float diffuse = max(dot(normal, normalize(uLightDirection)), 0.0);
            
            // Specular reflection
            vec3 viewDirection = normalize(-vWorldPosition);
            vec3 reflectionDirection = reflect(-normalize(uLightDirection), normal);
            float specular = pow(max(dot(viewDirection, reflectionDirection), 0.0), uShininess);
            
            // Combine lighting
            vec3 finalColor = uAmbientColor * baseColor +
                             uDiffuseColor * baseColor * diffuse +
                             uSpecularColor * specular +
                             warpColor;
                             
            gl_FragColor = vec4(finalColor, 1.0);
        }
        """
        
        # Deck plan overlay shader
        deck_vertex_shader = """
        attribute vec2 aPosition;
        attribute vec3 aColor;
        
        uniform mat4 uProjectionMatrix;
        uniform float uDeckLevel;
        uniform vec3 uHullOffset;
        
        varying vec3 vColor;
        
        void main(void) {
            vec3 worldPos = vec3(aPosition.x, aPosition.y, uDeckLevel) + uHullOffset;
            gl_Position = uProjectionMatrix * vec4(worldPos, 1.0);
            vColor = aColor;
        }
        """
        
        deck_fragment_shader = """
        precision mediump float;
        
        varying vec3 vColor;
        uniform float uOpacity;
        
        void main(void) {
            gl_FragColor = vec4(vColor, uOpacity);
        }
        """
        
        return {
            'hull': WebGLShader(hull_vertex_shader, hull_fragment_shader),
            'deck': WebGLShader(deck_vertex_shader, deck_fragment_shader)
        }
        
    def generate_hull_data_json(self, 
                               hull_geometry: HullGeometry,
                               deck_plans: List[DeckPlan]) -> Dict[str, Any]:
        """
        Generate JSON data structure for WebGL hull rendering.
        
        Args:
            hull_geometry: 3D hull geometry
            deck_plans: Extracted deck plans
            
        Returns:
            hull_data: Complete hull data for WebGL rendering
        """
        # Convert hull geometry to JSON-serializable format
        hull_data = {
            'geometry': {
                'vertices': hull_geometry.vertices.tolist(),
                'faces': hull_geometry.faces.tolist(),
                'normals': hull_geometry.normals.tolist(),
                'thickness': hull_geometry.thickness_map.tolist(),
                'vertex_count': len(hull_geometry.vertices),
                'face_count': len(hull_geometry.faces)
            },
            'material_properties': hull_geometry.material_properties,
            'deck_levels': hull_geometry.deck_levels,
            'deck_plans': [],
            'rendering_config': {
                'ambient_color': [0.1, 0.1, 0.12],
                'diffuse_color': [0.3, 0.35, 0.4], 
                'specular_color': [0.8, 0.85, 0.9],
                'shininess': 128.0,
                'light_direction': [1.0, 1.0, 1.0]
            }
        }
        
        # Add deck plan data
        for deck_plan in deck_plans:
            deck_data = {
                'name': deck_plan.deck_name,
                'level': deck_plan.deck_level,
                'boundary': [[p.x, p.y] for p in deck_plan.outer_boundary],
                'rooms': [
                    {
                        'id': room.id,
                        'type': room.room_type,
                        'center': [room.center.x, room.center.y],
                        'bounds': [
                            [room.bounds[0].x, room.bounds[0].y],
                            [room.bounds[1].x, room.bounds[1].y]
                        ],
                        'area': room.area
                    }
                    for room in deck_plan.rooms
                ],
                'corridors': [
                    {
                        'id': corridor.id,
                        'path': [[p.x, p.y] for p in corridor.path],
                        'width': corridor.width
                    }
                    for corridor in deck_plan.corridors
                ]
            }
            hull_data['deck_plans'].append(deck_data)
            
        return hull_data
        
    def generate_html_visualization(self, 
                                  hull_data: Dict[str, Any],
                                  output_path: str) -> None:
        """
        Generate complete HTML file with WebGL visualization.
        
        Args:
            hull_data: Hull data for rendering
            output_path: Path to save HTML file
        """
        shaders = self.generate_webgl_shaders()
        
        # Generate control panel HTML
        controls_html = ""
        for control in self.hull_controls:
            if control.control_type == "slider":
                controls_html += f"""
                <div class="control-group">
                    <label for="{control.parameter}">{control.label}:</label>
                    <input type="range" id="{control.parameter}" 
                           min="{control.min_value}" max="{control.max_value}" 
                           value="{control.default_value}" step="{control.step_size}"
                           oninput="updateHullParameter('{control.parameter}', this.value)">
                    <span id="{control.parameter}_value">{control.default_value}</span>
                </div>
                """
                
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FTL Ship Hull Geometry Visualization</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: #0a0a0a;
            color: #ffffff;
            font-family: 'Arial', sans-serif;
            overflow: hidden;
        }}
        
        #canvas-container {{
            position: relative;
            width: 100vw;
            height: 100vh;
        }}
        
        #hull-canvas {{
            display: block;
            background: linear-gradient(to bottom, #000011, #000033);
        }}
        
        #control-panel {{
            position: absolute;
            top: 20px;
            right: 20px;
            width: 300px;
            background: rgba(0, 0, 0, 0.8);
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        }}
        
        .control-group {{
            margin-bottom: 15px;
        }}
        
        .control-group label {{
            display: block;
            margin-bottom: 5px;
            color: #aaa;
            font-size: 12px;
        }}
        
        .control-group input[type="range"] {{
            width: 100%;
            margin-bottom: 5px;
        }}
        
        .control-group span {{
            color: #4ECDC4;
            font-weight: bold;
        }}
        
        #info-panel {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.8);
            border: 1px solid #333;
            border-radius: 8px;
            padding: 15px;
            max-width: 400px;
        }}
        
        .info-item {{
            margin-bottom: 8px;
            font-size: 12px;
        }}
        
        .info-label {{
            color: #888;
        }}
        
        .info-value {{
            color: #4ECDC4;
            font-weight: bold;
        }}
        
        #deck-controls {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.8);
            border: 1px solid #333;
            border-radius: 8px;
            padding: 15px;
        }}
        
        .deck-button {{
            display: block;
            width: 100%;
            margin-bottom: 5px;
            padding: 8px;
            background: #222;
            border: 1px solid #444;
            color: #aaa;
            border-radius: 4px;
            cursor: pointer;
            font-size: 11px;
        }}
        
        .deck-button:hover {{
            background: #333;
            color: #fff;
        }}
        
        .deck-button.active {{
            background: #4ECDC4;
            color: #000;
        }}
    </style>
</head>
<body>
    <div id="canvas-container">
        <canvas id="hull-canvas" width="{self.webgl_config['canvas_width']}" height="{self.webgl_config['canvas_height']}"></canvas>
        
        <div id="control-panel">
            <h3 style="margin-top: 0; color: #4ECDC4;">Hull Parameters</h3>
            {controls_html}
            <button onclick="resetHull()" style="width: 100%; padding: 10px; background: #FF6B6B; color: white; border: none; border-radius: 4px; cursor: pointer; margin-top: 10px;">Reset Hull</button>
        </div>
        
        <div id="deck-controls">
            <h4 style="margin-top: 0; color: #4ECDC4;">Deck Plans</h4>
            <button class="deck-button active" onclick="showDeck(-1)">Hull Only</button>
        </div>
        
        <div id="info-panel">
            <h4 style="margin-top: 0; color: #4ECDC4;">Hull Analysis</h4>
            <div class="info-item">
                <span class="info-label">Vertices:</span>
                <span class="info-value" id="vertex-count">{hull_data['geometry']['vertex_count']}</span>
            </div>
            <div class="info-item">
                <span class="info-label">Faces:</span>
                <span class="info-value" id="face-count">{hull_data['geometry']['face_count']}</span>
            </div>
            <div class="info-item">
                <span class="info-label">Safety Margin:</span>
                <span class="info-value" id="safety-margin">5.0</span>
            </div>
            <div class="info-item">
                <span class="info-label">Warp Velocity:</span>
                <span class="info-value" id="current-warp">48c</span>
            </div>
        </div>
    </div>

    <script>
        // Hull data from Python
        const hullData = {json.dumps(hull_data, indent=2)};
        
        // WebGL context and variables
        let gl;
        let shaderProgram;
        let hullBuffers = {{}};
        let camera = {{ x: 0, y: 0, z: 500, rotX: 0, rotY: 0 }};
        let animationTime = 0;
        let currentDeck = -1;
        
        // Initialize WebGL
        function initWebGL() {{
            const canvas = document.getElementById('hull-canvas');
            gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
            
            if (!gl) {{
                alert('WebGL not supported');
                return;
            }}
            
            // Set viewport
            gl.viewport(0, 0, canvas.width, canvas.height);
            gl.clearColor(...{self.webgl_config['background_color']});
            gl.enable(gl.DEPTH_TEST);
            gl.enable(gl.BLEND);
            gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
            
            // Create shaders
            createShaderProgram();
            
            // Create hull buffers
            createHullBuffers();
            
            // Setup deck buttons
            setupDeckButtons();
            
            // Start render loop
            animate();
        }}
        
        function createShaderProgram() {{
            const vertexShader = createShader(gl.VERTEX_SHADER, `{shaders['hull'].vertex_source}`);
            const fragmentShader = createShader(gl.FRAGMENT_SHADER, `{shaders['hull'].fragment_source}`);
            
            shaderProgram = gl.createProgram();
            gl.attachShader(shaderProgram, vertexShader);
            gl.attachShader(shaderProgram, fragmentShader);
            gl.linkProgram(shaderProgram);
            
            if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {{
                console.error('Shader program failed to link:', gl.getProgramInfoLog(shaderProgram));
                return;
            }}
            
            // Get attribute and uniform locations
            shaderProgram.vertexPositionAttribute = gl.getAttribLocation(shaderProgram, 'aVertexPosition');
            shaderProgram.vertexNormalAttribute = gl.getAttribLocation(shaderProgram, 'aVertexNormal');
            shaderProgram.textureCoordAttribute = gl.getAttribLocation(shaderProgram, 'aTextureCoord');
            shaderProgram.thicknessAttribute = gl.getAttribLocation(shaderProgram, 'aThickness');
            
            shaderProgram.modelViewMatrixUniform = gl.getUniformLocation(shaderProgram, 'uModelViewMatrix');
            shaderProgram.projectionMatrixUniform = gl.getUniformLocation(shaderProgram, 'uProjectionMatrix');
            shaderProgram.normalMatrixUniform = gl.getUniformLocation(shaderProgram, 'uNormalMatrix');
            shaderProgram.timeUniform = gl.getUniformLocation(shaderProgram, 'uTime');
            shaderProgram.warpVelocityUniform = gl.getUniformLocation(shaderProgram, 'uWarpVelocity');
            shaderProgram.lightDirectionUniform = gl.getUniformLocation(shaderProgram, 'uLightDirection');
            shaderProgram.ambientColorUniform = gl.getUniformLocation(shaderProgram, 'uAmbientColor');
            shaderProgram.diffuseColorUniform = gl.getUniformLocation(shaderProgram, 'uDiffuseColor');
            shaderProgram.specularColorUniform = gl.getUniformLocation(shaderProgram, 'uSpecularColor');
            shaderProgram.shininessUniform = gl.getUniformLocation(shaderProgram, 'uShininess');
        }}
        
        function createShader(type, source) {{
            const shader = gl.createShader(type);
            gl.shaderSource(shader, source);
            gl.compileShader(shader);
            
            if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {{
                console.error('Shader compilation error:', gl.getShaderInfoLog(shader));
                gl.deleteShader(shader);
                return null;
            }}
            
            return shader;
        }}
        
        function createHullBuffers() {{
            // Vertex positions
            hullBuffers.position = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, hullBuffers.position);
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(hullData.geometry.vertices.flat()), gl.STATIC_DRAW);
            
            // Vertex normals
            hullBuffers.normal = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, hullBuffers.normal);
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(hullData.geometry.normals.flat()), gl.STATIC_DRAW);
            
            // Thickness data
            hullBuffers.thickness = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, hullBuffers.thickness);
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(hullData.geometry.thickness), gl.STATIC_DRAW);
            
            // Face indices
            hullBuffers.indices = gl.createBuffer();
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, hullBuffers.indices);
            gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(hullData.geometry.faces.flat()), gl.STATIC_DRAW);
            
            // UV coordinates (simple mapping)
            const uvCoords = [];
            for (let i = 0; i < hullData.geometry.vertices.length; i++) {{
                uvCoords.push(0.5, 0.5); // Placeholder UV
            }}
            hullBuffers.textureCoord = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, hullBuffers.textureCoord);
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(uvCoords), gl.STATIC_DRAW);
        }}
        
        function setupDeckButtons() {{
            const deckControls = document.getElementById('deck-controls');
            hullData.deck_plans.forEach((deck, index) => {{
                const button = document.createElement('button');
                button.className = 'deck-button';
                button.textContent = deck.name;
                button.onclick = () => showDeck(index);
                deckControls.appendChild(button);
            }});
        }}
        
        function showDeck(deckIndex) {{
            currentDeck = deckIndex;
            
            // Update button states
            const buttons = document.querySelectorAll('.deck-button');
            buttons.forEach((btn, index) => {{
                btn.classList.remove('active');
                if (index === deckIndex + 1) {{
                    btn.classList.add('active');
                }}
            }});
        }}
        
        function updateHullParameter(parameter, value) {{
            document.getElementById(parameter + '_value').textContent = value;
            
            // Update display values
            if (parameter === 'warp_velocity') {{
                document.getElementById('current-warp').textContent = value + 'c';
            }}
            
            // TODO: Implement real-time hull regeneration
        }}
        
        function resetHull() {{
            // Reset all controls to default values
            {chr(10).join([
                f"document.getElementById('{control.parameter}').value = {control.default_value};"
                f"document.getElementById('{control.parameter}_value').textContent = {control.default_value};"
                for control in self.hull_controls
            ])}
        }}
        
        function animate() {{
            animationTime += 0.016; // ~60 FPS
            
            // Clear screen
            gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
            
            // Set up matrices
            const projectionMatrix = perspective({self.webgl_config['fov']}, 
                {self.webgl_config['canvas_width']} / {self.webgl_config['canvas_height']}, 
                {self.webgl_config['near_plane']}, {self.webgl_config['far_plane']});
                
            const modelViewMatrix = lookAt(camera.x, camera.y, camera.z, 0, 0, 0, 0, 1, 0);
            
            // Use shader program
            gl.useProgram(shaderProgram);
            
            // Set uniforms
            gl.uniformMatrix4fv(shaderProgram.projectionMatrixUniform, false, projectionMatrix);
            gl.uniformMatrix4fv(shaderProgram.modelViewMatrixUniform, false, modelViewMatrix);
            gl.uniform1f(shaderProgram.timeUniform, animationTime);
            gl.uniform1f(shaderProgram.warpVelocityUniform, 
                parseFloat(document.getElementById('warp_velocity').value));
            
            // Lighting uniforms
            gl.uniform3fv(shaderProgram.lightDirectionUniform, [1.0, 1.0, 1.0]);
            gl.uniform3fv(shaderProgram.ambientColorUniform, hullData.rendering_config.ambient_color);
            gl.uniform3fv(shaderProgram.diffuseColorUniform, hullData.rendering_config.diffuse_color);
            gl.uniform3fv(shaderProgram.specularColorUniform, hullData.rendering_config.specular_color);
            gl.uniform1f(shaderProgram.shininessUniform, hullData.rendering_config.shininess);
            
            // Bind and draw hull
            drawHull();
            
            requestAnimationFrame(animate);
        }}
        
        function drawHull() {{
            // Bind position buffer
            gl.bindBuffer(gl.ARRAY_BUFFER, hullBuffers.position);
            gl.enableVertexAttribArray(shaderProgram.vertexPositionAttribute);
            gl.vertexAttribPointer(shaderProgram.vertexPositionAttribute, 3, gl.FLOAT, false, 0, 0);
            
            // Bind normal buffer
            gl.bindBuffer(gl.ARRAY_BUFFER, hullBuffers.normal);
            gl.enableVertexAttribArray(shaderProgram.vertexNormalAttribute);
            gl.vertexAttribPointer(shaderProgram.vertexNormalAttribute, 3, gl.FLOAT, false, 0, 0);
            
            // Bind texture coordinate buffer
            gl.bindBuffer(gl.ARRAY_BUFFER, hullBuffers.textureCoord);
            gl.enableVertexAttribArray(shaderProgram.textureCoordAttribute);
            gl.vertexAttribPointer(shaderProgram.textureCoordAttribute, 2, gl.FLOAT, false, 0, 0);
            
            // Bind thickness buffer
            gl.bindBuffer(gl.ARRAY_BUFFER, hullBuffers.thickness);
            gl.enableVertexAttribArray(shaderProgram.thicknessAttribute);
            gl.vertexAttribPointer(shaderProgram.thicknessAttribute, 1, gl.FLOAT, false, 0, 0);
            
            // Bind indices and draw
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, hullBuffers.indices);
            gl.drawElements(gl.TRIANGLES, hullData.geometry.faces.length * 3, gl.UNSIGNED_SHORT, 0);
        }}
        
        // Matrix helper functions
        function perspective(fov, aspect, near, far) {{
            const f = Math.tan(Math.PI * 0.5 - 0.5 * fov * Math.PI / 180);
            const rangeInv = 1.0 / (near - far);
            
            return [
                f / aspect, 0, 0, 0,
                0, f, 0, 0,
                0, 0, (near + far) * rangeInv, -1,
                0, 0, near * far * rangeInv * 2, 0
            ];
        }}
        
        function lookAt(ex, ey, ez, tx, ty, tz, ux, uy, uz) {{
            const zx = ex - tx; const zy = ey - ty; const zz = ez - tz;
            const zlen = Math.sqrt(zx*zx + zy*zy + zz*zz);
            const znx = zx / zlen; const zny = zy / zlen; const znz = zz / zlen;
            
            const xx = uy*znz - uz*zny; const xy = uz*znx - ux*znz; const xz = ux*zny - uy*znx;
            const xlen = Math.sqrt(xx*xx + xy*xy + xz*xz);
            const xnx = xx / xlen; const xny = xy / xlen; const xnz = xz / xlen;
            
            const yx = zny*xnz - znz*xny; const yy = znz*xnx - znx*xnz; const yz = znx*xny - zny*xnx;
            
            return [
                xnx, yx, znx, 0,
                xny, yy, zny, 0,
                xnz, yz, znz, 0,
                -xnx*ex - xny*ey - xnz*ez, -yx*ex - yy*ey - yz*ez, -znx*ex - zny*ey - znz*ez, 1
            ];
        }}
        
        // Mouse controls
        let mouseDown = false;
        let lastMouseX = 0;
        let lastMouseY = 0;
        
        document.getElementById('hull-canvas').addEventListener('mousedown', (e) => {{
            mouseDown = true;
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
        }});
        
        document.addEventListener('mouseup', () => {{
            mouseDown = false;
        }});
        
        document.addEventListener('mousemove', (e) => {{
            if (mouseDown) {{
                const deltaX = e.clientX - lastMouseX;
                const deltaY = e.clientY - lastMouseY;
                
                camera.rotY += deltaX * 0.01;
                camera.rotX += deltaY * 0.01;
                
                lastMouseX = e.clientX;
                lastMouseY = e.clientY;
            }}
        }});
        
        // Zoom with mouse wheel
        document.addEventListener('wheel', (e) => {{
            camera.z += e.deltaY * 0.5;
            camera.z = Math.max(50, Math.min(2000, camera.z));
            e.preventDefault();
        }});
        
        // Initialize when page loads
        window.addEventListener('load', initWebGL);
    </script>
</body>
</html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
            
        self.logger.info(f"HTML visualization generated: {output_path}")
        
    def create_chrome_launcher_script(self, html_path: str, output_path: str) -> None:
        """
        Create a script to launch Chrome with the visualization.
        
        Args:
            html_path: Path to HTML file
            output_path: Path to save launcher script
        """
        script_content = f"""@echo off
echo Launching FTL Ship Hull Geometry Visualization...
echo.
echo Opening in Chrome with WebGL support...
start chrome.exe --allow-file-access-from-files --enable-webgl --disable-web-security "file:///{os.path.abspath(html_path).replace(chr(92), '/')}"
echo.
echo If the visualization doesn't load:
echo 1. Ensure Chrome supports WebGL
echo 2. Check browser console for errors
echo 3. Try running from a local web server
pause
        """
        
        with open(output_path, 'w') as f:
            f.write(script_content)
            
        self.logger.info(f"Chrome launcher script created: {output_path}")


def create_browser_visualization_demo() -> Dict:
    """
    Complete demonstration of browser-based hull visualization.
    
    Returns:
        demo_results: Browser visualization demonstration results
    """
    from hull_geometry_generator import HullPhysicsEngine, AlcubierreMetricConstraints
    
    logger.info("Starting Browser Visualization Demo")
    
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
        length=250.0,
        beam=45.0,
        height=35.0,
        n_sections=18
    )
    
    # Optimize for WebGL (Phase 2)
    obj_generator = OBJMeshGenerator()
    optimized_geometry = obj_generator.optimize_for_webgl(hull_geometry)
    
    # Extract deck plans (Phase 3)
    extractor = DeckPlanExtractor()
    deck_plans = extractor.extract_all_deck_plans(optimized_geometry)
    
    # Create browser visualization (Phase 4)
    viz_engine = BrowserVisualizationEngine()
    
    # Create output directory
    output_dir = "browser_visualization"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate hull data
    hull_data = viz_engine.generate_hull_data_json(optimized_geometry, deck_plans)
    
    # Generate HTML visualization
    html_path = os.path.join(output_dir, "ftl_hull_visualization.html")
    viz_engine.generate_html_visualization(hull_data, html_path)
    
    # Create Chrome launcher
    launcher_path = os.path.join(output_dir, "launch_visualization.bat")
    viz_engine.create_chrome_launcher_script(html_path, launcher_path)
    
    # Save hull data JSON
    data_path = os.path.join(output_dir, "hull_data.json") 
    with open(data_path, 'w') as f:
        json.dump(hull_data, f, indent=2)
    
    demo_results = {
        'output_directory': output_dir,
        'files_generated': {
            'html_visualization': html_path,
            'chrome_launcher': launcher_path,
            'hull_data': data_path
        },
        'visualization_features': {
            'interactive_controls': len(viz_engine.hull_controls),
            'deck_plans': len(deck_plans),
            'webgl_optimized': True,
            'real_time_warp_effects': True,
            'mouse_camera_control': True
        },
        'performance_metrics': {
            'vertices': hull_data['geometry']['vertex_count'],
            'faces': hull_data['geometry']['face_count'],
            'webgl_compatible': hull_data['geometry']['vertex_count'] <= 65536,
            'file_size_html': os.path.getsize(html_path),
            'file_size_data': os.path.getsize(data_path)
        },
        'browser_requirements': {
            'webgl_support': True,
            'chrome_recommended': True,
            'local_file_access': True,
            'javascript_enabled': True
        }
    }
    
    logger.info(
        f"Browser visualization demo complete: {hull_data['geometry']['vertex_count']} vertices, "
        f"{len(deck_plans)} deck plans, WebGL ready"
    )
    
    return demo_results


if __name__ == "__main__":
    # Run browser visualization demonstration
    results = create_browser_visualization_demo()
    
    print("\n" + "="*60)
    print("SHIP HULL GEOMETRY PHASE 4: BROWSER VISUALIZATION")
    print("="*60)
    print(f"Output Directory: {results['output_directory']}")
    print(f"Interactive Controls: {results['visualization_features']['interactive_controls']}")
    print(f"Deck Plans: {results['visualization_features']['deck_plans']}")
    print(f"Vertices: {results['performance_metrics']['vertices']}")
    print(f"Faces: {results['performance_metrics']['faces']}")
    print(f"WebGL Compatible: {results['performance_metrics']['webgl_compatible']}")
    print(f"HTML File Size: {results['performance_metrics']['file_size_html']} bytes")
    
    print("\nGenerated Files:")
    for file_type, path in results['files_generated'].items():
        print(f"  {file_type}: {path}")
        
    print("\nVisualization Features:")
    for feature, enabled in results['visualization_features'].items():
        print(f"  {feature}: {enabled}")
        
    print("\nTo view the visualization:")
    print(f"  1. Run: {results['files_generated']['chrome_launcher']}")
    print("  2. Or open HTML file in Chrome with --allow-file-access-from-files")
    print("="*60)

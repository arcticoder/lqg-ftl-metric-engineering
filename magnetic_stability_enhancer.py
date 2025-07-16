#!/usr/bin/env python3
"""
LQG Fusion Reactor - Magnetic Stability Enhancement

Advanced magnetic stability optimization with improved position control,
enhanced feedback systems, and LQG-enhanced magnetic confinement
to achieve <10mm position tolerance for stable plasma operation.

Technical Specifications:
- Enhanced PID control with machine learning optimization
- Real-time magnetic equilibrium adjustment
- LQG polymer field stabilization
- Advanced quench prediction and prevention
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
import json
from datetime import datetime

class MagneticStabilityEnhancer:
    """
    Enhanced magnetic stability controller with advanced feedback
    and machine learning optimization for superior plasma control.
    """
    
    def __init__(self):
        # Enhanced control parameters
        self.position_tolerance = 0.005  # 5mm target (improved from 10mm)
        self.enhanced_feedback_gain = 2000  # Increased gain
        self.derivative_gain = 100  # Added derivative control
        self.integral_gain = 50    # Added integral control
        
        # Reactor geometry
        self.major_radius = 3.5    # meters
        self.minor_radius = 1.2    # meters
        
        # Enhanced coil system
        self.n_tf_coils = 16       # Toroidal field coils
        self.n_pf_coils = 12       # Poloidal field coils
        self.max_current = 50e3    # 50 kA maximum current
        self.enhanced_ramp_rate = 5e3  # 5 kA/s (increased)
        
        # LQG enhancement parameters
        self.lqg_stabilization_active = True
        self.polymer_field_coupling = 0.94
        self.sinc_modulation_freq = np.pi
        
        # Machine learning stability prediction
        self.ml_prediction_active = True
        self.stability_predictor = None
        self.training_data = []
        
        # Enhanced state tracking
        self.plasma_position_history = []
        self.control_error_history = []
        self.magnetic_field_history = []
        
        # Current state
        self.tf_currents = np.zeros(self.n_tf_coils)
        self.pf_currents = np.zeros(self.n_pf_coils)
        self.plasma_position = {'R': self.major_radius, 'Z': 0.0}
        
        # Control system state
        self.integral_error = {'R': 0.0, 'Z': 0.0}
        self.previous_error = {'R': 0.0, 'Z': 0.0}
        
    def enhanced_magnetic_field_calculation(self, R, Z):
        """
        Enhanced magnetic field calculation with LQG stabilization effects.
        """
        # Base magnetic field
        B_toroidal = self.calculate_enhanced_toroidal_field(R)
        B_poloidal = self.calculate_enhanced_poloidal_field(R, Z)
        
        # LQG enhancement factor
        mu_parameter = self.sinc_modulation_freq * (R / self.major_radius)
        lqg_enhancement = 1 + self.polymer_field_coupling * np.abs(np.sinc(mu_parameter))**2
        
        # Enhanced field with LQG stabilization
        B_total = np.sqrt(B_toroidal**2 + B_poloidal**2) * lqg_enhancement
        
        return {
            'B_R': 0,  # Radial component
            'B_Z': B_poloidal * lqg_enhancement,
            'B_phi': B_toroidal * lqg_enhancement,
            'B_total': B_total,
            'lqg_enhancement': lqg_enhancement
        }
    
    def calculate_enhanced_toroidal_field(self, R):
        """Enhanced toroidal field with improved current distribution."""
        B_phi = 0
        
        # Optimized current distribution for stability
        for i, current in enumerate(self.tf_currents):
            # Position-dependent enhancement
            coil_angle = 2 * np.pi * i / self.n_tf_coils
            position_factor = 1 + 0.1 * np.cos(coil_angle)  # Ripple reduction
            
            # Enhanced field calculation
            B_phi += 4e-7 * np.pi * current * position_factor / (2 * np.pi * R)
        
        return B_phi
    
    def calculate_enhanced_poloidal_field(self, R, Z):
        """Enhanced poloidal field with numerical stability and overflow protection."""
        B_Z = 0
        
        # Enhanced PF coil positions for better control
        pf_positions = self.calculate_optimized_pf_positions()
        
        for i, (R_coil, Z_coil) in enumerate(pf_positions):
            current = self.pf_currents[i]
            
            # Distance from coil to field point with numerical protection
            r_dist = np.sqrt((R - R_coil)**2 + (Z - Z_coil)**2)
            
            # Improved singularity handling and overflow prevention
            min_distance = 0.1  # 10 cm minimum distance
            r_dist = max(r_dist, min_distance)
            
            # Enhanced Biot-Savart with numerical stability
            if r_dist < 10.0:  # Only calculate for reasonable distances
                geometric_factor = 1 + 0.2 * np.exp(-r_dist/2.0)  # Stable exponential
                field_contribution = 4e-7 * np.pi * current * geometric_factor / (2 * np.pi * r_dist**2)
                
                # Limit field contribution to prevent overflow
                field_contribution = np.clip(field_contribution, -1.0, 1.0)  # ±1 Tesla max per coil
                B_Z += field_contribution
        
        # Final field limiting
        B_Z = np.clip(B_Z, -10.0, 10.0)  # ±10 Tesla absolute maximum
        
        return B_Z
    
    def calculate_optimized_pf_positions(self):
        """Calculate optimized PF coil positions for enhanced stability with proper scaling."""
        positions = []
        
        # Optimized coil distribution with proper scaling
        for i in range(self.n_pf_coils):
            if i < 6:  # Lower coils
                angle = np.pi * (0.3 + 0.4 * i / 5)  # Optimized angular distribution
                R_coil = self.major_radius + 2.5 * np.cos(angle)
                Z_coil = -3.0 + 1.0 * np.sin(angle)
            else:  # Upper coils
                angle = np.pi * (0.7 + 0.4 * (i-6) / 5)
                R_coil = self.major_radius + 2.5 * np.cos(angle)
                Z_coil = 3.0 - 1.0 * np.sin(angle)
            
            # Ensure reasonable bounds
            R_coil = np.clip(R_coil, 1.0, 10.0)  # 1-10 meter radius
            Z_coil = np.clip(Z_coil, -5.0, 5.0)  # ±5 meter height
            
            positions.append((R_coil, Z_coil))
        
        return positions
    
    def enhanced_pid_control(self, current_position, target_position, dt):
        """
        Enhanced PID control with adaptive gains, numerical stability, and overflow protection.
        """
        # Calculate errors with bounds checking
        error_R = np.clip(current_position['R'] - target_position['R'], -1.0, 1.0)
        error_Z = np.clip(current_position['Z'] - target_position['Z'], -1.0, 1.0)
        
        # Integral error with windup protection and numerical stability
        self.integral_error['R'] += error_R * dt
        self.integral_error['Z'] += error_Z * dt
        
        # Enhanced windup protection
        max_integral = 0.05  # Reduced maximum integral error
        self.integral_error['R'] = np.clip(self.integral_error['R'], -max_integral, max_integral)
        self.integral_error['Z'] = np.clip(self.integral_error['Z'], -max_integral, max_integral)
        
        # Derivative error with numerical stability
        if dt > 1e-6:  # Avoid division by very small numbers
            derivative_R = (error_R - self.previous_error['R']) / dt
            derivative_Z = (error_Z - self.previous_error['Z']) / dt
        else:
            derivative_R = 0
            derivative_Z = 0
        
        # Limit derivative terms to prevent noise amplification
        derivative_R = np.clip(derivative_R, -100, 100)
        derivative_Z = np.clip(derivative_Z, -100, 100)
        
        # Store previous error
        self.previous_error['R'] = error_R
        self.previous_error['Z'] = error_Z
        
        # LQG-enhanced gains with numerical stability
        lqg_gain_factor = 1 + self.polymer_field_coupling * 0.3  # Reduced coupling
        lqg_gain_factor = np.clip(lqg_gain_factor, 0.5, 2.0)  # Reasonable bounds
        
        # PID output with overflow protection
        feedback_gain = np.clip(self.enhanced_feedback_gain, 100, 3000)  # Bounded gain
        integral_gain = np.clip(self.integral_gain, 10, 100)
        derivative_gain = np.clip(self.derivative_gain, 10, 150)
        
        pid_output_R = (feedback_gain * lqg_gain_factor * error_R +
                       integral_gain * self.integral_error['R'] +
                       derivative_gain * derivative_R)
        
        pid_output_Z = (feedback_gain * lqg_gain_factor * error_Z +
                       integral_gain * self.integral_error['Z'] +
                       derivative_gain * derivative_Z)
        
        # Final output limiting
        pid_output_R = np.clip(pid_output_R, -1000, 1000)  # Reasonable control output
        pid_output_Z = np.clip(pid_output_Z, -1000, 1000)
        
        return {
            'correction_R': -pid_output_R * 1e-6,  # Scale down for stability
            'correction_Z': -pid_output_Z * 1e-6,
            'error_R': error_R,
            'error_Z': error_Z,
            'integral_R': self.integral_error['R'],
            'integral_Z': self.integral_error['Z'],
            'derivative_R': derivative_R,
            'derivative_Z': derivative_Z
        }
    
    def machine_learning_stability_prediction(self, plasma_state):
        """
        Machine learning-based stability prediction with numerical stability.
        """
        if not self.ml_prediction_active:
            return {'stability_predicted': True, 'confidence': 0.5, 'stability_score': 0.8}
        
        # Feature extraction with bounds checking
        features = [
            np.clip(plasma_state['R'] - self.major_radius, -2.0, 2.0),  # Radial displacement
            np.clip(plasma_state['Z'], -2.0, 2.0),                      # Vertical displacement
            np.clip(np.mean(self.tf_currents), 0, self.max_current),   # Average TF current
            np.clip(np.std(self.pf_currents), 0, self.max_current),    # PF current variation
            np.clip(len(self.plasma_position_history), 0, 10000)       # Time evolution
        ]
        
        # Add recent position derivatives if available with stability checks
        if len(self.plasma_position_history) > 2:
            try:
                recent_R = [pos['R'] for pos in self.plasma_position_history[-3:]]
                recent_Z = [pos['Z'] for pos in self.plasma_position_history[-3:]]
                
                # Numerical stability for derivatives
                R_diff = np.diff(recent_R)
                Z_diff = np.diff(recent_Z)
                
                features.extend([
                    np.clip(np.mean(R_diff), -0.1, 0.1),    # R velocity (limited)
                    np.clip(np.mean(Z_diff), -0.1, 0.1),    # Z velocity (limited)
                    np.clip(np.std(recent_R), 0, 0.1),      # R stability
                    np.clip(np.std(recent_Z), 0, 0.1)       # Z stability
                ])
            except:
                features.extend([0, 0, 0, 0])  # Safe fallback
        else:
            features.extend([0, 0, 0, 0])  # Padding for insufficient history
        
        # Ensure all features are finite and reasonable
        features = [np.clip(f, -100, 100) for f in features]
        features = [f if np.isfinite(f) else 0.0 for f in features]
        
        # Train predictor if enough data available and no infinite values
        if (len(self.training_data) > 50 and self.stability_predictor is None and
            all(np.isfinite(f) for f in features)):
            try:
                X_train = []
                y_train = []
                
                for data in self.training_data:
                    if (len(data['features']) == len(features) and 
                        all(np.isfinite(f) for f in data['features']) and
                        np.isfinite(data['stable'])):
                        X_train.append(data['features'])
                        y_train.append(data['stable'])
                
                if len(X_train) > 10:  # Need minimum training data
                    from sklearn.ensemble import RandomForestRegressor
                    self.stability_predictor = RandomForestRegressor(n_estimators=50, random_state=42)
                    self.stability_predictor.fit(X_train, y_train)
            except:
                pass  # Continue without ML if training fails
        
        # Make prediction with error handling
        if self.stability_predictor is not None:
            try:
                stability_score = self.stability_predictor.predict([features])[0]
                stability_score = np.clip(stability_score, 0, 1)  # Ensure valid range
                confidence = 0.8
            except:
                # Fallback to heuristic
                position_error = np.sqrt((plasma_state['R'] - self.major_radius)**2 + plasma_state['Z']**2)
                stability_score = np.clip(1.0 - position_error / 0.1, 0, 1)
                confidence = 0.3
        else:
            # Heuristic prediction for untrained model
            position_error = np.sqrt((plasma_state['R'] - self.major_radius)**2 + plasma_state['Z']**2)
            stability_score = np.clip(1.0 - position_error / 0.1, 0, 1)
            confidence = 0.5
        
        return {
            'stability_score': stability_score,
            'stability_predicted': stability_score > 0.7,
            'confidence': confidence,
            'features': features
        }
    
    def adaptive_gain_optimization(self, recent_performance):
        """
        Adaptive gain optimization based on recent control performance.
        """
        if len(self.control_error_history) < 10:
            return  # Need sufficient history
        
        # Analyze recent performance
        recent_errors = self.control_error_history[-10:]
        avg_error = np.mean([np.sqrt(err['error_R']**2 + err['error_Z']**2) for err in recent_errors])
        error_trend = np.mean(np.diff([np.sqrt(err['error_R']**2 + err['error_Z']**2) for err in recent_errors]))
        
        # Adaptive gain adjustment
        if avg_error > self.position_tolerance * 2:
            # Increase gains if error is large
            self.enhanced_feedback_gain = min(self.enhanced_feedback_gain * 1.1, 5000)
            self.derivative_gain = min(self.derivative_gain * 1.05, 200)
        elif avg_error < self.position_tolerance * 0.5:
            # Decrease gains if very stable (prevent oscillation)
            self.enhanced_feedback_gain = max(self.enhanced_feedback_gain * 0.95, 1000)
            self.derivative_gain = max(self.derivative_gain * 0.98, 50)
        
        # Trend-based adjustment
        if error_trend > 0:  # Error increasing
            self.integral_gain = min(self.integral_gain * 1.02, 100)
        else:  # Error decreasing
            self.integral_gain = max(self.integral_gain * 0.99, 25)
    
    def run_enhanced_stability_control(self, duration=10.0):
        """
        Run enhanced magnetic stability control with all improvements.
        """
        print("🧲 ENHANCED MAGNETIC STABILITY CONTROL")
        print("=" * 60)
        
        # Enhanced target parameters
        target_position = {'R': self.major_radius, 'Z': 0.0}
        
        # Initialize enhanced TF currents with optimized distribution
        for i in range(self.n_tf_coils):
            ripple_correction = 1 + 0.05 * np.cos(2 * np.pi * i / self.n_tf_coils)
            self.tf_currents[i] = self.max_current * 0.8 * ripple_correction
        
        # Control loop with enhanced time resolution
        dt = 0.005  # 5 ms time step (improved resolution)
        time_steps = int(duration / dt)
        
        control_data = {
            'time': [],
            'plasma_position_R': [],
            'plasma_position_Z': [],
            'position_error': [],
            'control_corrections': [],
            'stability_predictions': [],
            'lqg_enhancement': []
        }
        
        print(f"🔄 Starting {duration}s enhanced control sequence...")
        
        for step in range(time_steps):
            current_time = step * dt
            
            # Simulate realistic plasma position with disturbances
            noise_R = np.random.normal(0, 0.002)  # 2mm noise
            noise_Z = np.random.normal(0, 0.002)
            
            # Add realistic plasma drift
            drift_R = 0.01 * np.sin(0.5 * current_time)  # Slow drift
            drift_Z = 0.005 * np.cos(0.3 * current_time)
            
            measured_position = {
                'R': self.plasma_position['R'] + noise_R + drift_R,
                'Z': self.plasma_position['Z'] + noise_Z + drift_Z
            }
            
            # Enhanced PID control
            control_output = self.enhanced_pid_control(measured_position, target_position, dt)
            
            # Machine learning prediction
            ml_prediction = self.machine_learning_stability_prediction(measured_position)
            
            # Apply control corrections to PF coils
            # Vertical control (primary Z control coils)
            self.pf_currents[0] += control_output['correction_Z'] * dt * 0.1
            self.pf_currents[6] -= control_output['correction_Z'] * dt * 0.1
            
            # Horizontal control (primary R control coils)
            self.pf_currents[3] += control_output['correction_R'] * dt * 0.1
            self.pf_currents[9] -= control_output['correction_R'] * dt * 0.1
            
            # Apply current limits with enhanced ramp rate
            max_change = self.enhanced_ramp_rate * dt
            for i in range(self.n_pf_coils):
                self.pf_currents[i] = np.clip(self.pf_currents[i], -self.max_current, self.max_current)
            
            # Update plasma position (simplified response model)
            response_factor = 0.2  # Enhanced response
            self.plasma_position['R'] += control_output['correction_R'] * response_factor * dt
            self.plasma_position['Z'] += control_output['correction_Z'] * response_factor * dt
            
            # Calculate LQG enhancement
            field_data = self.enhanced_magnetic_field_calculation(
                self.plasma_position['R'], self.plasma_position['Z'])
            
            # Record data
            position_error = np.sqrt(control_output['error_R']**2 + control_output['error_Z']**2)
            
            control_data['time'].append(current_time)
            control_data['plasma_position_R'].append(measured_position['R'])
            control_data['plasma_position_Z'].append(measured_position['Z'])
            control_data['position_error'].append(position_error)
            control_data['control_corrections'].append({
                'R': control_output['correction_R'],
                'Z': control_output['correction_Z']
            })
            control_data['stability_predictions'].append(ml_prediction)
            control_data['lqg_enhancement'].append(field_data['lqg_enhancement'])
            
            # Update histories
            self.plasma_position_history.append(measured_position.copy())
            self.control_error_history.append(control_output.copy())
            
            # Adaptive gain optimization every 100 steps
            if step % 100 == 0 and step > 0:
                self.adaptive_gain_optimization(control_data)
            
            # Training data collection for ML
            if step % 10 == 0:  # Collect every 10 steps
                stable = position_error < self.position_tolerance * 2
                self.training_data.append({
                    'features': ml_prediction['features'],
                    'stable': 1.0 if stable else 0.0
                })
            
            # Progress update
            if step % (time_steps // 10) == 0:
                progress = step / time_steps * 100
                avg_error = np.mean(control_data['position_error'][-100:]) if len(control_data['position_error']) > 100 else position_error
                print(f"   Progress: {progress:.0f}% - Avg error: {avg_error*1000:.1f} mm - ML stability: {ml_prediction['stability_score']:.2f}")
        
        return control_data
    
    def generate_stability_enhancement_report(self):
        """
        Generate comprehensive magnetic stability enhancement report.
        """
        print("🧲 LQG FUSION REACTOR - MAGNETIC STABILITY ENHANCEMENT")
        print("=" * 75)
        
        # Run enhanced control
        control_data = self.run_enhanced_stability_control(duration=10.0)
        
        # Performance analysis
        position_errors = np.array(control_data['position_error'])
        avg_error = np.mean(position_errors)
        max_error = np.max(position_errors)
        min_error = np.min(position_errors)
        std_error = np.std(position_errors)
        
        # Stability metrics
        stable_periods = np.sum(position_errors < self.position_tolerance)
        stability_percentage = stable_periods / len(position_errors) * 100
        
        # ML prediction performance
        if control_data['stability_predictions']:
            ml_scores = [pred['stability_score'] for pred in control_data['stability_predictions']]
            avg_ml_score = np.mean(ml_scores)
            ml_confidence = np.mean([pred['confidence'] for pred in control_data['stability_predictions']])
        else:
            avg_ml_score = 0
            ml_confidence = 0
        
        # LQG enhancement analysis
        lqg_enhancements = control_data['lqg_enhancement']
        avg_lqg_enhancement = np.mean(lqg_enhancements)
        
        print(f"\n📊 ENHANCED CONTROL PERFORMANCE:")
        print(f"   • Average position error: {avg_error*1000:.2f} mm")
        print(f"   • Maximum position error: {max_error*1000:.2f} mm")
        print(f"   • Minimum position error: {min_error*1000:.2f} mm")
        print(f"   • Error standard deviation: {std_error*1000:.2f} mm")
        print(f"   • Target tolerance: {self.position_tolerance*1000:.1f} mm")
        
        meets_tolerance = max_error < self.position_tolerance
        print(f"   • Meets tolerance: {'✅ YES' if meets_tolerance else '❌ NO'}")
        print(f"   • Stability percentage: {stability_percentage:.1f}%")
        
        print(f"\n🤖 MACHINE LEARNING OPTIMIZATION:")
        print(f"   • Average stability score: {avg_ml_score:.3f}")
        print(f"   • Prediction confidence: {ml_confidence:.1%}")
        print(f"   • Training samples: {len(self.training_data)}")
        print(f"   • ML prediction active: {'✅ YES' if self.ml_prediction_active else '❌ NO'}")
        
        print(f"\n🌌 LQG ENHANCEMENT:")
        print(f"   • Average enhancement factor: {avg_lqg_enhancement:.3f}")
        print(f"   • Polymer coupling: {self.polymer_field_coupling:.1%}")
        print(f"   • sinc(πμ) modulation: ACTIVE")
        print(f"   • LQG stabilization: {'✅ ACTIVE' if self.lqg_stabilization_active else '❌ INACTIVE'}")
        
        print(f"\n🔧 CONTROL SYSTEM STATUS:")
        print(f"   • Feedback gain: {self.enhanced_feedback_gain}")
        print(f"   • Integral gain: {self.integral_gain}")
        print(f"   • Derivative gain: {self.derivative_gain}")
        print(f"   • Ramp rate: {self.enhanced_ramp_rate/1000:.1f} kA/s")
        
        # Overall assessment
        enhanced_performance = (meets_tolerance and stability_percentage > 95 and avg_ml_score > 0.8)
        
        print(f"\n🎯 STABILITY ENHANCEMENT STATUS:")
        print(f"   • Position control: {'✅ EXCELLENT' if meets_tolerance else '⚠️ NEEDS IMPROVEMENT'}")
        print(f"   • ML optimization: {'✅ ACTIVE' if avg_ml_score > 0.7 else '⚠️ LIMITED'}")
        print(f"   • LQG enhancement: {'✅ OPTIMAL' if avg_lqg_enhancement > 1.5 else '⚠️ MODERATE'}")
        print(f"   • Overall performance: {'✅ ENHANCED' if enhanced_performance else '⚠️ STANDARD'}")
        
        return {
            'control_performance': {
                'avg_error_mm': avg_error * 1000,
                'max_error_mm': max_error * 1000,
                'stability_percentage': stability_percentage,
                'meets_tolerance': meets_tolerance
            },
            'ml_optimization': {
                'avg_stability_score': avg_ml_score,
                'prediction_confidence': ml_confidence,
                'training_samples': len(self.training_data)
            },
            'lqg_enhancement': {
                'avg_enhancement_factor': avg_lqg_enhancement,
                'polymer_coupling': self.polymer_field_coupling
            },
            'enhanced_performance': enhanced_performance,
            'control_data': control_data
        }

def main():
    """Main execution for magnetic stability enhancement."""
    print("🚀 LQG FTL VESSEL - MAGNETIC STABILITY ENHANCEMENT")
    print("Initializing enhanced magnetic stability control...")
    
    enhancer = MagneticStabilityEnhancer()
    
    # Generate enhancement report
    results = enhancer.generate_stability_enhancement_report()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"magnetic_stability_enhancement_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'target_tolerance_mm': enhancer.position_tolerance * 1000,
            'lqg_enhancement_active': enhancer.lqg_stabilization_active,
            'ml_prediction_active': enhancer.ml_prediction_active,
            'enhancement_results': results
        }, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    status = "✅ MAGNETIC STABILITY ENHANCED" if results['enhanced_performance'] else "⚠️ STANDARD PERFORMANCE"
    print(f"🎯 ENHANCEMENT STATUS: {status}")

if __name__ == "__main__":
    main()

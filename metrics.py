#Test wise function containing individual metrics 

import numpy as np
from collections import deque
import statistics

class PerformanceMetrics:
    """
    Comprehensive performance metrics.
    Separates reporting logic for Pushups and Squats.
    """
    
    def __init__(self):
        self.exercise = None
        self.good_reps = 0
        self.bad_reps = 0
        self.bad_form_count = 0
        
        # Angle tracking per rep
        self.rep_angles = []  # List of (max_angle, min_angle) tuples per rep
        self.back_angles = deque(maxlen=100)
        self.elbow_angles = deque(maxlen=100)
        
        self.rep_durations = []  # Time in seconds for each rep
        
        # Squat-specific tracking
        self.squat_depths = []  # Min knee angle per rep
        self.torso_angles = deque(maxlen=100)  # Back inclination angles
        self.shin_angles = deque(maxlen=100)  # Shin angles relative to vertical
        self.knee_positions = deque(maxlen=100)  # For knee stability (valgus/varus)
        self.hip_velocities = []  # Velocity during concentric phase
        self.eccentric_times = []  # Descent time per rep
        self.concentric_times = []  # Ascent time per rep
        self.sticking_points = []  # Knee angle at minimum velocity
        self.rep_start_time = None  # type: float | None
        self.rep_bottom_time = None  # type: float | None
        self.last_hip_y = None  # type: float | None
        self.last_frame_time = None  # type: float | None
        self.current_phase = 'standing'  # standing, descending, bottom, ascending
        self.min_velocity_angle = None  # type: int | None
        self.min_velocity = float('inf')
        
        # Situp-specific tracking
        self.situp_torso_inclinations = []  # Peak torso angles per rep
        self.situp_hip_flexions = []  # Minimum hip flexion per rep
        self.situp_foot_lifts = []  # Foot lift violations per rep
        self.situp_neck_strains = []  # Neck strain detection per rep
        self.situp_concentric_times = []  # Up phase time
        self.situp_eccentric_times = []  # Down phase time
        self.situp_momentum_scores = []  # Jerk/momentum per rep
        self.situp_short_rom_count = 0  # Reps with incomplete ROM
        self.situp_valid_reps = 0  # Strict reps only
        self.ankle_baseline_y = None  # type: float | None
        self.knee_baseline_y = None  # type: float | None
        self.shoulder_positions = deque(maxlen=10)  # For acceleration calc
        self.situp_rep_start_time = None  # type: float | None
        self.situp_peak_time = None  # type: float | None
        self.situp_state = 'rest'  # rest, ascending, peak, descending
        self.max_torso_inclination = 0
        self.min_hip_flexion = 180
        
        # Thresholds
        self.thresholds = {
            'min_hip_angle': 150,
            'ideal_back_angle': 180,
            'max_back_deviation': 45,
            'ideal_rom': 90,
            'max_arm_asymmetry': 30,
            'squat_parallel': 90,
            'ideal_torso_angle': 35,
            'max_knee_deviation': 50,  # pixels
            # Situp thresholds
            'situp_up_angle': 70,  # Minimum torso inclination for "up"
            'situp_down_angle': 20,  # Maximum angle for "down"/reset
            'situp_good_hip_flexion': 50,  # Good crunch angle
            'situp_foot_lift_threshold': 30,  # pixels
            'situp_momentum_threshold': 50,  # Jerk score
        }
    
    def update_angle_data(self, left_elbow, right_elbow, left_hip, right_hip, 
                      left_shoulder, right_shoulder, left_knee, right_knee,
                      left_ankle, right_ankle, left_wrist, right_wrist):
        """Update tracking data with current frame angles."""
        if left_elbow is not None and right_elbow is not None:
            self.elbow_angles.append((left_elbow, right_elbow))
        
        # Calculate back straightness (spine angle)
        if (left_hip is not None) or (right_hip is not None):
            back_angle = self._calculate_back_angle(left_shoulder, left_hip, left_knee, 
                                                  right_shoulder, right_hip, right_knee)
            if back_angle is not None:
                self.back_angles.append(back_angle)
    
    def _calculate_back_angle(self, left_shoulder, left_hip, left_knee, 
                            right_shoulder, right_hip, right_knee):
        """Calculate average back/spine angle from both sides"""
        angles = []
        if left_hip is not None:
            angles.append(left_hip)
        if right_hip is not None:
            angles.append(right_hip)
        
        if angles:
            return int(np.mean(angles))
        return None
    
    def update_squat_data(self, keypoints, angles, torso_angle, shin_angle, current_time, fps=30):
        """Update squat-specific tracking data per frame."""
        if keypoints is None:
            return
        
        # Track torso and shin angles
        if torso_angle is not None:
            self.torso_angles.append(torso_angle)
        if shin_angle is not None:
            self.shin_angles.append(shin_angle)
        
        # Track knee stability (valgus/varus detection)
        left_knee_conf = keypoints[13][2]
        right_knee_conf = keypoints[14][2]
        left_ankle_conf = keypoints[15][2]
        right_ankle_conf = keypoints[16][2]
        
        if left_knee_conf > 0.5 and left_ankle_conf > 0.5 and right_knee_conf > 0.5 and right_ankle_conf > 0.5:
            left_knee_x = keypoints[13][0]
            right_knee_x = keypoints[14][0]
            left_ankle_x = keypoints[15][0]
            right_ankle_x = keypoints[16][0]
            
            left_deviation = abs(left_knee_x - left_ankle_x)
            right_deviation = abs(right_knee_x - right_ankle_x)
            self.knee_positions.append({
                'left_dev': left_deviation,
                'right_dev': right_deviation,
                'left_x': left_knee_x,
                'right_x': right_knee_x
            })
        
        # Calculate hip velocity for concentric phase
        hip_conf = max(keypoints[11][2], keypoints[12][2])
        if hip_conf > 0.5:
            hip_y = (keypoints[11][1] + keypoints[12][1]) / 2
            
            if self.last_hip_y is not None and self.last_frame_time is not None:
                time_delta = current_time - self.last_frame_time
                if time_delta > 0:
                    velocity = (self.last_hip_y - hip_y) / time_delta  # Positive = moving up
                    self.hip_velocities.append(velocity)
                    
                    # Track sticking point (minimum velocity during ascent)
                    if self.current_phase == 'ascending' and velocity < self.min_velocity:
                        self.min_velocity = velocity
                        knee_angle = angles.get('left_knee') or angles.get('right_knee')
                        if knee_angle is not None:
                            self.min_velocity_angle = knee_angle
            
            self.last_hip_y = hip_y
            self.last_frame_time = current_time
    
    def _calculate_torso_inclination_score(self):
        """Calculate score based on torso angle (0-35 degrees is ideal)."""
        if not self.torso_angles:
            return 0
        avg_angle = np.mean(list(self.torso_angles))
        if avg_angle <= self.thresholds['ideal_torso_angle']:
            return 100
        else:
            deviation = avg_angle - self.thresholds['ideal_torso_angle']
            score = max(0, 100 - (deviation * 2))  # Lose 2 points per degree over ideal
            return int(score)
    
    def _calculate_knee_stability_score(self):
        """Calculate score based on knee stability (valgus/varus)."""
        if not self.knee_positions:
            return 100
        
        violations = 0
        for pos in self.knee_positions:
            if pos['left_dev'] > self.thresholds['max_knee_deviation']:
                violations += 1
            if pos['right_dev'] > self.thresholds['max_knee_deviation']:
                violations += 1
        
        violation_rate = violations / (len(self.knee_positions) * 2)
        score = int((1 - violation_rate) * 100)
        return max(0, score)
    
    def _calculate_depth_consistency(self):
        """Calculate ROM consistency based on depth variation."""
        if len(self.squat_depths) < 2:
            return 100
        std_dev = np.std(self.squat_depths)
        mean_depth = np.mean(self.squat_depths)
        if mean_depth == 0:
            return 0
        cv = std_dev / mean_depth  # Coefficient of variation
        score = int(max(0, (1 - cv) * 100))
        return score
    
    def _calculate_tempo_score(self):
        """Calculate tempo consistency for squats."""
        if len(self.eccentric_times) < 2:
            return 100
        combined_times = self.eccentric_times + self.concentric_times
        if not combined_times:
            return 100
        mean_time = np.mean(combined_times)
        if mean_time == 0:
            return 100
        std_time = np.std(combined_times)
        cv = std_time / mean_time
        score = int(max(0, (1 - cv) * 100))
        return score
    
    def _get_avg_concentric_velocity(self):
        """Get average velocity during concentric phase."""
        if not self.hip_velocities:
            return 0
        positive_velocities = [v for v in self.hip_velocities if v > 0]
        if not positive_velocities:
            return 0
        return np.mean(positive_velocities)
    
    # ---------------------------------------------------------
    # SITUP HELPER METHODS
    # ---------------------------------------------------------
    
    def update_situp_data(self, keypoints, angles, torso_inclination, hip_flexion, current_time):
        """Update situp-specific tracking data per frame."""
        if keypoints is None:
            return
        
        # Track peak torso inclination during rep
        if torso_inclination is not None and torso_inclination > self.max_torso_inclination:
            self.max_torso_inclination = torso_inclination
        
        # Track minimum hip flexion during rep
        if hip_flexion is not None and hip_flexion < self.min_hip_flexion:
            self.min_hip_flexion = hip_flexion
        
        # Establish baseline for foot lift detection (first 30 frames)
        knee_conf = max(keypoints[13][2], keypoints[14][2])
        ankle_conf = max(keypoints[15][2], keypoints[16][2])
        
        if knee_conf > 0.5 and ankle_conf > 0.5:
            knee_y = (keypoints[13][1] + keypoints[14][1]) / 2
            ankle_y = (keypoints[15][1] + keypoints[16][1]) / 2
            
            if self.ankle_baseline_y is None:
                self.ankle_baseline_y = ankle_y
                self.knee_baseline_y = knee_y
        
        # Track shoulder positions for momentum detection
        shoulder_conf = max(keypoints[5][2], keypoints[6][2])
        if shoulder_conf > 0.5:
            shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2
            self.shoulder_positions.append((current_time, shoulder_y))
    
    def _detect_foot_lift(self, keypoints):
        """Detect if feet/knees lifted during rep."""
        if self.ankle_baseline_y is None or self.knee_baseline_y is None:
            return False
        
        knee_conf = max(keypoints[13][2], keypoints[14][2])
        ankle_conf = max(keypoints[15][2], keypoints[16][2])
        
        if knee_conf > 0.5 and ankle_conf > 0.5:
            knee_y = (keypoints[13][1] + keypoints[14][1]) / 2
            ankle_y = (keypoints[15][1] + keypoints[16][1]) / 2
            
            knee_lift = abs(knee_y - self.knee_baseline_y)
            ankle_lift = abs(ankle_y - self.ankle_baseline_y)
            
            if knee_lift > self.thresholds['situp_foot_lift_threshold'] or \
               ankle_lift > self.thresholds['situp_foot_lift_threshold']:
                return True
        
        return False
    
    def _detect_neck_strain(self, keypoints):
        """Detect neck strain by measuring ear-shoulder distance change."""
        ear_conf = max(keypoints[3][2], keypoints[4][2])
        shoulder_conf = max(keypoints[5][2], keypoints[6][2])
        
        if ear_conf > 0.5 and shoulder_conf > 0.5:
            # Use the most confident side
            if keypoints[3][2] > keypoints[4][2]:
                ear_pos = np.array(keypoints[3][:2])
                shoulder_pos = np.array(keypoints[5][:2])
            else:
                ear_pos = np.array(keypoints[4][:2])
                shoulder_pos = np.array(keypoints[6][:2])
            
            distance = np.linalg.norm(ear_pos - shoulder_pos)
            return distance
        
        return None
    
    def _calculate_momentum_score(self):
        """Calculate momentum/jerk score based on shoulder acceleration."""
        if len(self.shoulder_positions) < 3:
            return 0
        
        # Calculate acceleration (2nd derivative)
        positions = list(self.shoulder_positions)
        accelerations = []
        
        for i in range(2, len(positions)):
            t0, y0 = positions[i-2]
            t1, y1 = positions[i-1]
            t2, y2 = positions[i]
            
            dt1 = t1 - t0
            dt2 = t2 - t1
            
            if dt1 > 0 and dt2 > 0:
                v1 = (y1 - y0) / dt1
                v2 = (y2 - y1) / dt2
                accel = (v2 - v1) / ((dt1 + dt2) / 2)
                accelerations.append(abs(accel))
        
        if not accelerations:
            return 0
        
        # Check if initial acceleration is much higher (momentum usage)
        if len(accelerations) >= 3:
            initial_accel = np.mean(accelerations[:3])
            avg_accel = np.mean(accelerations)
            
            if avg_accel > 0:
                jerk_ratio = initial_accel / avg_accel
                # Higher ratio = more momentum
                score = int(min(100, jerk_ratio * 30))
                return score
        
        return 0
    
    def _get_situp_form_score(self):
        """Calculate situp form score (0-100)."""
        if self.situp_valid_reps == 0:
            return 0
        
        total_deductions = 0
        
        # 5 points per foot lift
        total_deductions += sum(self.situp_foot_lifts) * 5
        
        # 2 points per short ROM
        total_deductions += self.situp_short_rom_count * 2
        
        # Calculate score
        score = max(0, 100 - total_deductions)
        return int(score)
    
    def _get_situp_tempo_consistency(self):
        """Calculate tempo consistency for situps."""
        if len(self.situp_concentric_times) < 2:
            return 100
        
        combined = self.situp_concentric_times + self.situp_eccentric_times
        if not combined:
            return 100
        
        mean_time = np.mean(combined)
        if mean_time == 0:
            return 100
        
        std_time = np.std(combined)
        cv = std_time / mean_time
        score = int(max(0, (1 - cv) * 100))
        return score
    
    def record_rep(self, rep_max, rep_min, duration_seconds, is_good_form):
        """Record a completed repetition."""
        self.rep_angles.append((rep_max, rep_min))
        self.rep_durations.append(duration_seconds)
        
        if is_good_form:
            self.good_reps += 1
        else:
            self.bad_reps += 1
    
    @property
    def total_reps(self):
        return self.good_reps + self.bad_reps

    # ---------------------------------------------------------
    # HELPER CALCULATIONS (Used by pushup_metrics)
    # ---------------------------------------------------------

    def _get_form_score(self):
        if self.total_reps == 0: return 0
        return int((self.good_reps / self.total_reps) * 100)

    def _get_range_of_motion_score(self):
        if not self.rep_angles: return 0
        rom_values = [max_a - min_a for max_a, min_a in self.rep_angles]
        avg_rom = np.mean(rom_values)
        return int(min(100, (avg_rom / self.thresholds['ideal_rom']) * 100))

    def _get_alignment_score(self):
        if not self.back_angles: return 0
        back_angles = list(self.back_angles)
        avg_error = np.mean([abs(a - self.thresholds['ideal_back_angle']) for a in back_angles])
        score = max(0, 1 - (avg_error / self.thresholds['max_back_deviation']))
        return int(score * 100)

    def _get_arm_symmetry_score(self):
        if not self.elbow_angles: return 0
        symmetry_errors = [abs(l - r) for l, r in self.elbow_angles if l and r]
        if not symmetry_errors: return 0
        avg_error = np.mean(symmetry_errors)
        score = max(0, 1 - (avg_error / self.thresholds['max_arm_asymmetry']))
        return int(score * 100)

    def _get_tempo_consistency_score(self):
        if len(self.rep_durations) < 2: return 100
        mean_dur = np.mean(self.rep_durations)
        if mean_dur == 0: return 100
        variation = np.std(self.rep_durations) / mean_dur
        return int(max(0, 1 - variation) * 100)

    def _get_rating(self, score):
        if score >= 90: return "⭐⭐⭐⭐⭐ EXCELLENT"
        elif score >= 75: return "⭐⭐⭐⭐ VERY GOOD"
        elif score >= 60: return "⭐⭐⭐ GOOD"
        elif score >= 45: return "⭐⭐ FAIR"
        elif score >= 30: return "⭐ NEEDS IMPROVEMENT"
        return "KEEP PRACTICING"

    # ---------------------------------------------------------
    # MAIN METRIC WRAPPERS
    # ---------------------------------------------------------

    def pushup_metrics(self):
        """
        Wraps all logic for calculating and displaying Push-up specific metrics.
        """
        if not self.exercise: self.exercise = 'pushup'
        
        # 1. Calculate individual components
        form = self._get_form_score()
        rom = self._get_range_of_motion_score()
        alignment = self._get_alignment_score()
        symmetry = self._get_arm_symmetry_score()
        tempo = self._get_tempo_consistency_score()
        
        # 2. Calculate Weighted Overall Score
        # Weights: Form(25%), ROM(20%), Alignment(20%), Symmetry(15%), Tempo(10%)
        # Note: Sum is 90% in original code, assumed intended or loose math. 
        overall = (form * 0.25) + (rom * 0.20) + (alignment * 0.20) + \
                  (symmetry * 0.15) + (tempo * 0.10)
        
        # Normalize to 100 scale roughly if weights don't add to 1.0, 
        # or just take raw calculation as per original logic.
        overall = int(overall) 
        
        # 3. Generate Feedback Messages
        messages = []
        if form < 50: messages.append("❌ FORM: Focus on maintaining proper form")
        elif form < 80: messages.append("⚠ FORM: Work on consistency")
        else: messages.append("✓ FORM: Excellent form maintained!")
        
        if rom < 60: messages.append("❌ ROM: Push deeper")
        elif rom < 80: messages.append("⚠ ROM: Get lower")
        else: messages.append("✓ ROM: Great depth!")
        
        if alignment < 50: messages.append("❌ ALIGNMENT: Keep back straight")
        elif alignment < 80: messages.append("⚠ ALIGNMENT: Watch your back sag")
        else: messages.append("✓ ALIGNMENT: Excellent spine stability!")

        if symmetry < 60: messages.append("❌ SYMMETRY: Significant arm imbalance")
        elif symmetry < 80: messages.append("⚠ SYMMETRY: Focus on pushing evenly")
        else: messages.append("✓ SYMMETRY: Good left/right balance")

        # 4. Display Report
        print("\n" + "="*70)
        print(" "*15 + "PUSH-UP PERFORMANCE REPORT")
        print("="*70)
        
        print(f"\n📊 REPETITIONS:")
        print(f"    ✓ Good Reps: {self.good_reps}")
        print(f"    ✗ Bad Reps:  {self.bad_reps}")
        print(f"    Total:       {self.total_reps}")
        
        print(f"\n📈 DETAILED SCORES:")
        print(f"    Form Quality:    {form}/100")
        print(f"    Range of Motion: {rom}/100")
        print(f"    Body Alignment:  {alignment}/100")
        print(f"    Arm Symmetry:    {symmetry}/100")
        print(f"    Tempo:           {tempo}/100")
        
        print(f"\n" + "="*70)
        print(f"🏆 OVERALL SCORE: {overall}/100")
        print(f"⭐ RATING: {self._get_rating(overall)}")
        print("="*70)
        
        print(f"\n💡 FEEDBACK:")
        for msg in messages:
            print(f"    {msg}")
            
        return {
            'exercise': 'pushup',
            'overall_score': overall,
            'details': {'form': form, 'rom': rom, 'alignment': alignment}
        }

    def squat_metrics(self):
        """
        Comprehensive squat analysis with all biomechanical metrics.
        Based on squat.md specifications.
        """
        if not self.exercise:
            self.exercise = 'squat'
        
        # 1. Squat Depth Analysis
        avg_depth = np.mean(self.squat_depths) if self.squat_depths else 0
        parallel_reps = sum(1 for d in self.squat_depths if d <= self.thresholds['squat_parallel'])
        depth_score = int((parallel_reps / len(self.squat_depths)) * 100) if self.squat_depths else 0
        
        # 2. Back Inclination (Torso Angle)
        torso_score = self._calculate_torso_inclination_score()
        avg_torso_angle = np.mean(list(self.torso_angles)) if self.torso_angles else 0
        
        # 3. Shin Angle
        avg_shin_angle = np.mean(list(self.shin_angles)) if self.shin_angles else 0
        
        # 4. Knee Stability
        knee_stability_score = self._calculate_knee_stability_score()
        knee_cave_detected = knee_stability_score < 70
        
        # 5. Concentric Velocity
        avg_velocity = self._get_avg_concentric_velocity()
        
        # 6. Eccentric Tempo
        avg_eccentric_time = np.mean(self.eccentric_times) if self.eccentric_times else 0
        avg_concentric_time = np.mean(self.concentric_times) if self.concentric_times else 0
        
        # 7. Sticking Point
        sticking_point_angle = self.min_velocity_angle if self.min_velocity_angle else "N/A"
        
        # 8. ROM Consistency
        consistency_score = self._calculate_depth_consistency()
        
        # 9. Tempo Consistency
        tempo_score = self._calculate_tempo_score()
        
        # Overall Score Calculation (weighted average)
        overall = (
            depth_score * 0.25 +
            torso_score * 0.20 +
            knee_stability_score * 0.20 +
            consistency_score * 0.15 +
            tempo_score * 0.10 +
            (100 if not knee_cave_detected else 70) * 0.10
        )
        overall = int(overall)
        
        # Generate Feedback
        messages = []
        if depth_score < 50:
            messages.append("❌ DEPTH: Squat deeper - aim for parallel or below")
        elif depth_score < 80:
            messages.append("⚠ DEPTH: Almost there - get your hips to knee level")
        else:
            messages.append("✓ DEPTH: Excellent depth consistency!")
        
        if torso_score < 50:
            messages.append("❌ TORSO: Excessive forward lean - strengthen core")
        elif torso_score < 80:
            messages.append("⚠ TORSO: Slight forward lean detected")
        else:
            messages.append("✓ TORSO: Great upright posture!")
        
        if knee_cave_detected:
            messages.append("❌ KNEES: Knee valgus detected - push knees outward")
        else:
            messages.append("✓ KNEES: Good knee tracking!")
        
        if consistency_score < 70:
            messages.append("⚠ CONSISTENCY: Depth varies between reps")
        else:
            messages.append("✓ CONSISTENCY: Very consistent depth!")
        
        # Display Report
        print("\n" + "="*70)
        print(" "*15 + "SQUAT PERFORMANCE REPORT")
        print("="*70)
        
        print(f"\n📊 REPETITIONS:")
        print(f"    Total Reps:        {self.total_reps}")
        print(f"    Parallel+ Reps:    {parallel_reps}/{len(self.squat_depths)}")
        print(f"    Good Form Reps:    {self.good_reps}")
        print(f"    Poor Form Reps:    {self.bad_reps}")
        
        print(f"\n📐 BIOMECHANICS (Category A - Form):")
        print(f"    Avg Depth:         {avg_depth:.1f}° (Target: ≤{self.thresholds['squat_parallel']}°)")
        print(f"    Depth Score:       {depth_score}/100")
        print(f"    Avg Torso Angle:   {avg_torso_angle:.1f}° (Ideal: 0-{self.thresholds['ideal_torso_angle']}°)")
        print(f"    Torso Score:       {torso_score}/100")
        print(f"    Avg Shin Angle:    {avg_shin_angle:.1f}°")
        print(f"    Knee Stability:    {knee_stability_score}/100")
        if knee_cave_detected:
            print(f"    ⚠️  Knee Cave:       DETECTED")
        
        print(f"\n⚡ PHYSICS (Category B - Power):")
        print(f"    Avg Velocity:      {avg_velocity:.2f} px/s")
        print(f"    Avg Descent Time:  {avg_eccentric_time:.2f}s")
        print(f"    Avg Ascent Time:   {avg_concentric_time:.2f}s")
        if sticking_point_angle != "N/A":
            print(f"    Sticking Point:    {sticking_point_angle}°")
        
        print(f"\n📈 CONSISTENCY (Category C):")
        print(f"    ROM Consistency:   {consistency_score}/100")
        print(f"    Tempo Consistency: {tempo_score}/100")
        
        print(f"\n" + "="*70)
        print(f"🏆 OVERALL SCORE: {overall}/100")
        print(f"⭐ RATING: {self._get_rating(overall)}")
        print("="*70)
        
        print(f"\n💡 FEEDBACK:")
        for msg in messages:
            print(f"    {msg}")
        
        print("\n" + "="*70)
        
        return {
            'exercise': 'squat',
            'overall_score': overall,
            'depth_score': depth_score,
            'torso_score': torso_score,
            'knee_stability': knee_stability_score,
            'consistency': consistency_score
        }
    
    def situp_metrics(self):
        """
        Comprehensive situp analysis with all biomechanical metrics.
        Based on situp.md specifications.
        """
        if not self.exercise:
            self.exercise = 'situp'
        
        # 1. Torso Inclination Analysis
        avg_torso_inclination = np.mean(self.situp_torso_inclinations) if self.situp_torso_inclinations else 0
        good_inclination_reps = sum(1 for t in self.situp_torso_inclinations if t >= self.thresholds['situp_up_angle'])
        inclination_score = int((good_inclination_reps / len(self.situp_torso_inclinations)) * 100) if self.situp_torso_inclinations else 0
        
        # 2. Hip Flexion (Crunch Quality)
        avg_hip_flexion = np.mean(self.situp_hip_flexions) if self.situp_hip_flexions else 0
        good_flexion_reps = sum(1 for h in self.situp_hip_flexions if h <= self.thresholds['situp_good_hip_flexion'])
        flexion_score = int((good_flexion_reps / len(self.situp_hip_flexions)) * 100) if self.situp_hip_flexions else 0
        
        # 3. Foot Lift Detection
        total_foot_lifts = sum(self.situp_foot_lifts)
        foot_lift_rate = (total_foot_lifts / len(self.situp_foot_lifts)) * 100 if self.situp_foot_lifts else 0
        
        # 4. Neck Strain
        avg_neck_distance = np.mean(self.situp_neck_strains) if self.situp_neck_strains else 0
        
        # 5. Tempo Analysis
        avg_up_time = np.mean(self.situp_concentric_times) if self.situp_concentric_times else 0
        avg_down_time = np.mean(self.situp_eccentric_times) if self.situp_eccentric_times else 0
        tempo_consistency = self._get_situp_tempo_consistency()
        
        # 6. Momentum Detection
        avg_momentum_score = np.mean(self.situp_momentum_scores) if self.situp_momentum_scores else 0
        high_momentum_reps = sum(1 for m in self.situp_momentum_scores if m > self.thresholds['situp_momentum_threshold'])
        
        # 7. Valid Rep Count
        valid_rep_rate = (self.situp_valid_reps / self.total_reps) * 100 if self.total_reps > 0 else 0
        
        # 8. Form Score
        form_score = self._get_situp_form_score()
        
        # Overall Score
        overall = (
            inclination_score * 0.20 +
            flexion_score * 0.20 +
            (100 - foot_lift_rate) * 0.15 +
            form_score * 0.20 +
            tempo_consistency * 0.10 +
            (100 - min(100, avg_momentum_score)) * 0.15
        )
        overall = int(overall)
        
        # Generate Feedback
        messages = []
        if inclination_score < 50:
            messages.append("❌ ROM: Sit up higher - reach toward your knees")
        elif inclination_score < 80:
            messages.append("⚠ ROM: Almost there - lift your chest higher")
        else:
            messages.append("✓ ROM: Excellent full range of motion!")
        
        if flexion_score < 50:
            messages.append("❌ CRUNCH: Focus on crunching/folding tighter")
        elif flexion_score < 80:
            messages.append("⚠ CRUNCH: Good, but try to close the gap more")
        else:
            messages.append("✓ CRUNCH: Perfect core engagement!")
        
        if total_foot_lifts > 0:
            messages.append(f"❌ ANCHORING: Feet lifted {total_foot_lifts} times - keep them down")
        else:
            messages.append("✓ ANCHORING: Perfect foot placement!")
        
        if high_momentum_reps > 0:
            messages.append(f"⚠ MOMENTUM: {high_momentum_reps} reps used excessive swing")
        else:
            messages.append("✓ CONTROL: Great controlled movement!")
        
        # Display Report
        print("\n" + "="*70)
        print(" "*15 + "SIT-UP PERFORMANCE REPORT")
        print("="*70)
        
        print(f"\n📊 REPETITIONS:")
        print(f"    Total Reps:        {self.total_reps}")
        print(f"    Valid Reps:        {self.situp_valid_reps}")
        print(f"    Good Form Reps:    {self.good_reps}")
        print(f"    Poor Form Reps:    {self.bad_reps}")
        
        print(f"\n📐 BIOMECHANICS (Category A - Form):")
        print(f"    Avg Torso Angle:   {avg_torso_inclination:.1f}° (Target: ≥{self.thresholds['situp_up_angle']}°)")
        print(f"    Inclination Score: {inclination_score}/100")
        print(f"    Avg Hip Flexion:   {avg_hip_flexion:.1f}° (Target: ≤{self.thresholds['situp_good_hip_flexion']}°)")
        print(f"    Flexion Score:     {flexion_score}/100")
        if total_foot_lifts > 0:
            print(f"    ⚠️  Foot Lifts:      {total_foot_lifts} violations")
        else:
            print(f"    ✓  Foot Stability:  Perfect")
        
        print(f"\n⚡ PHYSICS (Category B - Power):")
        print(f"    Avg Up Time:       {avg_up_time:.2f}s")
        print(f"    Avg Down Time:     {avg_down_time:.2f}s")
        print(f"    Tempo Consistency: {tempo_consistency}/100")
        print(f"    Avg Momentum:      {avg_momentum_score:.1f}/100")
        if high_momentum_reps > 0:
            print(f"    ⚠️  High Momentum:   {high_momentum_reps} reps")
        
        print(f"\n📈 SCORING (Category C):")
        print(f"    Form Score:        {form_score}/100")
        print(f"    Valid Rep Rate:    {valid_rep_rate:.1f}%")
        print(f"    Short ROM Count:   {self.situp_short_rom_count}")
        
        print(f"\n" + "="*70)
        print(f"🏆 OVERALL SCORE: {overall}/100")
        print(f"⭐ RATING: {self._get_rating(overall)}")
        print("="*70)
        
        print(f"\n💡 FEEDBACK:")
        for msg in messages:
            print(f"    {msg}")
        
        print("\n" + "="*70)
        
        return {
            'exercise': 'situp',
            'overall_score': overall,
            'inclination_score': inclination_score,
            'flexion_score': flexion_score,
            'form_score': form_score
        }

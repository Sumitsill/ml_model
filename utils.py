import cv2
import numpy as np
from ultralytics import YOLO

class PoseCalibrator:
    def __init__(self, model_path='yolov8n-pose.pt'):
        self.model = YOLO(model_path)
        self.joint_angles = {
            'left_elbow': (5, 7, 9), 'right_elbow': (6, 8, 10),
            'left_hip': (5, 11, 13), 'right_hip': (6, 12, 14),
            'left_knee': (11, 13, 15), 'right_knee': (12, 14, 16),
        }
    
    def detect_pose(self, frame):
        results = self.model(frame, verbose=False)
        if len(results[0].keypoints) > 0:
            return results[0].keypoints.data[0].cpu().numpy()
        return None
    
    def calculate_angle(self, pt1, pt2, pt3):
        pt1, pt2, pt3 = np.array(pt1), np.array(pt2), np.array(pt3)
        v1, v2 = pt1 - pt2, pt3 - pt2
        v1_norm, v2_norm = np.linalg.norm(v1), np.linalg.norm(v2)
        if v1_norm == 0 or v2_norm == 0: return 0
        cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
        return int(round(np.degrees(np.arccos(cos_angle))))
    
    def get_all_joint_angles(self, keypoints):
        angles = {}
        for joint_name, (idx1, idx2, idx3) in self.joint_angles.items():
            if keypoints[idx1][2] > 0.5 and keypoints[idx2][2] > 0.5 and keypoints[idx3][2] > 0.5:
                angles[joint_name] = self.calculate_angle(
                    keypoints[idx1][:2], keypoints[idx2][:2], keypoints[idx3][:2]
                )
            else:
                angles[joint_name] = None
        
        angles['torso_angle'] = self.calculate_torso_angle(keypoints)
        angles['shin_angle_left'] = self.calculate_shin_angle(keypoints, 13, 15)
        angles['shin_angle_right'] = self.calculate_shin_angle(keypoints, 14, 16)
        angles['torso_inclination_horizontal'] = self.calculate_torso_horizontal(keypoints)
        angles['hip_flexion_angle'] = self.calculate_hip_flexion(keypoints)
        return angles
    
    def calculate_torso_angle(self, keypoints):
        if all(keypoints[i][2] > 0.5 for i in [5, 6, 11, 12]):
            shoulder_mid = np.mean([keypoints[5][:2], keypoints[6][:2]], axis=0)
            hip_mid = np.mean([keypoints[11][:2], keypoints[12][:2]], axis=0)
            torso_vector = shoulder_mid - hip_mid
            torso_norm = np.linalg.norm(torso_vector)
            if torso_norm == 0: return None
            cos_angle = np.clip(np.dot(torso_vector, [0, -1]) / torso_norm, -1.0, 1.0)
            return int(round(np.degrees(np.arccos(cos_angle))))
        return None
    
    def calculate_shin_angle(self, keypoints, knee_idx, ankle_idx):
        if keypoints[knee_idx][2] > 0.5 and keypoints[ankle_idx][2] > 0.5:
            shin_vector = keypoints[knee_idx][:2] - keypoints[ankle_idx][:2]
            shin_norm = np.linalg.norm(shin_vector)
            if shin_norm == 0: return None
            cos_angle = np.clip(np.dot(shin_vector, [0, -1]) / shin_norm, -1.0, 1.0)
            return int(round(np.degrees(np.arccos(cos_angle))))
        return None
    
    def calculate_torso_horizontal(self, keypoints):
        if all(keypoints[i][2] > 0.5 for i in [5, 6, 11, 12]):
            shoulder_mid = np.mean([keypoints[5][:2], keypoints[6][:2]], axis=0)
            hip_mid = np.mean([keypoints[11][:2], keypoints[12][:2]], axis=0)
            dy, dx = shoulder_mid[1] - hip_mid[1], shoulder_mid[0] - hip_mid[0]
            return int(round(abs(np.degrees(np.arctan2(-dy, dx)))))
        return None
    
    def calculate_hip_flexion(self, keypoints):
        left_conf = min(keypoints[5][2], keypoints[11][2], keypoints[13][2])
        right_conf = min(keypoints[6][2], keypoints[12][2], keypoints[14][2])
        
        if left_conf > right_conf and left_conf > 0.5:
            shoulder, hip, knee = keypoints[5][:2], keypoints[11][:2], keypoints[13][:2]
        elif right_conf > 0.5:
            shoulder, hip, knee = keypoints[6][:2], keypoints[12][:2], keypoints[14][:2]
        else:
            return None
        
        v1, v2 = shoulder - hip, knee - hip
        v1_norm, v2_norm = np.linalg.norm(v1), np.linalg.norm(v2)
        if v1_norm == 0 or v2_norm == 0: return None
        cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
        return int(round(np.degrees(np.arccos(cos_angle))))
    
    def process_frame(self, frame, show_angles_panel=False):
        keypoints = self.detect_pose(frame)
        angles = {}
        if keypoints is not None:
            angles = self.get_all_joint_angles(keypoints)
        return frame, keypoints, angles

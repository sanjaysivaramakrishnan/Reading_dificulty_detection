"""
Reading Difficulty Detector - Main Detection Engine
Author: Sanjay Sivaramakrishnan M
Date: September 2025
"""

import cv2
import numpy as np
import mediapipe as mp
import time
from typing import Dict, Tuple, Optional
import pandas as pd

class ReadingDifficultyDetector:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Eye landmarks
        self.left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

        # Feature storage
        self.feature_history = []
        self.max_history = 300
        self.blink_threshold = 0.2

        # Performance tracking
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0

        print("ReadingDifficultyDetector initialized")

    def extract_facial_landmarks(self, frame: np.ndarray) -> Optional[Dict]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape

            landmark_points = []
            for landmark in landmarks.landmark:     
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                z = landmark.z
                landmark_points.append([x, y, z])

            return {
                'landmarks': landmark_points,
                'frame_shape': frame.shape
            }

        return None

    def extract_eye_features(self, landmark_data: Dict) -> Dict:
        landmarks = landmark_data['landmarks']
        features = {}

        try:
            left_eye_points = [landmarks[i] for i in self.left_eye_indices]
            right_eye_points = [landmarks[i] for i in self.right_eye_indices]

            features['left_ear'] = self._calculate_ear(left_eye_points)
            features['right_ear'] = self._calculate_ear(right_eye_points)
            features['avg_ear'] = (features['left_ear'] + features['right_ear']) / 2

            left_center = np.mean(left_eye_points, axis=0)
            right_center = np.mean(right_eye_points, axis=0)

            features['left_eye_center'] = left_center[:2]
            features['right_eye_center'] = right_center[:2]

            avg_gaze = (left_center + right_center) / 2
            features['gaze_x'] = avg_gaze[0]
            features['gaze_y'] = avg_gaze[1]

            features['eye_distance'] = np.linalg.norm(left_center - right_center)

        except Exception as e:
            features.update({
                'left_ear': 0.3, 'right_ear': 0.3, 'avg_ear': 0.3,
                'gaze_x': 0, 'gaze_y': 0, 'eye_distance': 0
            })

        return features

    def _calculate_ear(self, eye_points) -> float:
        try:
            points = np.array(eye_points)
            A = np.linalg.norm(points[1][:2] - points[5][:2])
            B = np.linalg.norm(points[2][:2] - points[4][:2])
            C = np.linalg.norm(points[0][:2] - points[3][:2])

            if C > 0:
                ear = (A + B) / (2.0 * C)
            else:
                ear = 0.3

            return max(0.0, min(1.0, ear))
        except:
            return 0.3

    def extract_head_pose(self, landmark_data: Dict) -> Dict:
        landmarks = landmark_data['landmarks']
        h, w, _ = landmark_data['frame_shape']

        try:
            model_points = np.array([
                (0.0, 0.0, 0.0),
                (0.0, -330.0, -65.0),
                (-165.0, 170.0, -135.0),
                (165.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0),
                (150.0, -150.0, -125.0)
            ])

            landmark_indices = [1, 152, 33, 263, 61, 291]
            points_2d = np.array([[landmarks[i][0], landmarks[i][1]] for i in landmark_indices], dtype=np.float64)

            focal_length = w
            center = (w/2, h/2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)

            dist_coeffs = np.zeros((4,1))

            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, points_2d, camera_matrix, dist_coeffs
            )

            if success:
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                angles = cv2.RQDecomp3x3(rotation_matrix)[0]

                return {
                    'pitch': angles[0],
                    'yaw': angles[1], 
                    'roll': angles[2],
                    'distance': np.linalg.norm(translation_vector)
                }
        except:
            pass

        return {'pitch': 0, 'yaw': 0, 'roll': 0, 'distance': 0}

    def extract_temporal_features(self) -> Dict:
        if len(self.feature_history) < 30:
            return {}

        recent_features = self.feature_history[-300:]

        try:
            df = pd.DataFrame(recent_features)
            temporal_features = {}

            if 'gaze_x' in df.columns and 'gaze_y' in df.columns:
                temporal_features['gaze_stability_x'] = df['gaze_x'].var()
                temporal_features['gaze_stability_y'] = df['gaze_y'].var()

                gaze_diff_x = df['gaze_x'].diff().dropna()
                if len(gaze_diff_x) > 0:
                    temporal_features['reading_progression'] = gaze_diff_x.mean()
                    direction_changes = np.sum(np.diff(np.sign(gaze_diff_x.values)) != 0)
                    temporal_features['gaze_direction_changes'] = direction_changes

            if 'avg_ear' in df.columns:
                blinks = (df['avg_ear'] < self.blink_threshold).astype(int)
                if len(blinks) > 1:
                    blink_changes = np.diff(blinks.values)
                    blink_count = np.sum(blink_changes == 1)
                    temporal_features['blink_rate'] = blink_count * 2

            return temporal_features

        except:
            return {}

    def predict_reading_difficulty(self, frame: np.ndarray) -> Tuple[float, Dict]:
        self._update_fps_counter()

        landmark_data = self.extract_facial_landmarks(frame)

        if landmark_data is None:
            return 0.0, {'error': 'No face detected', 'fps': self.current_fps}

        eye_features = self.extract_eye_features(landmark_data)
        head_pose_features = self.extract_head_pose(landmark_data)

        current_features = {
            **eye_features,
            **head_pose_features,
            'fps': self.current_fps
        }

        self.feature_history.append(current_features)
        if len(self.feature_history) > self.max_history:
            self.feature_history.pop(0)

        temporal_features = self.extract_temporal_features()
        all_features = {**current_features, **temporal_features}

        if temporal_features:
            difficulty_score = self._calculate_rule_based_score(all_features)
        else:
            difficulty_score = self._calculate_basic_score(current_features)

        return difficulty_score, all_features

    def _calculate_rule_based_score(self, features: Dict) -> float:
        score = 0.0

        blink_rate = features.get('blink_rate', 15)
        if blink_rate < 8:
            score += 0.25
        elif blink_rate < 12:
            score += 0.1
        elif blink_rate > 25:
            score += 0.15

        gaze_stability_x = features.get('gaze_stability_x', 0)
        if gaze_stability_x > 2000:
            score += 0.3
        elif gaze_stability_x > 1000:
            score += 0.15

        direction_changes = features.get('gaze_direction_changes', 0)
        if direction_changes > 80:
            score += 0.25
        elif direction_changes > 50:
            score += 0.1

        pitch = features.get('pitch', 0)
        if pitch < -15:
            score += 0.15
        elif pitch < -8:
            score += 0.05

        avg_ear = features.get('avg_ear', 0.3)
        if avg_ear < 0.15:
            score += 0.15
        elif avg_ear < 0.2:
            score += 0.05

        return min(score, 1.0)

    def _calculate_basic_score(self, features: Dict) -> float:
        score = 0.0

        avg_ear = features.get('avg_ear', 0.3)
        if avg_ear < 0.15:
            score += 0.3
        elif avg_ear < 0.2:
            score += 0.1

        pitch = features.get('pitch', 0)
        if pitch < -15:
            score += 0.2

        return min(score, 1.0)

    def _update_fps_counter(self):
        self.fps_counter += 1
        current_time = time.time()

        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time

    def reset(self):
        self.feature_history.clear()
        self.fps_counter = 0

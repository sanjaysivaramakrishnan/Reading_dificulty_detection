"""
Data Logger - Data collection and storage
Author: Sanjay Sivaramakrishnan M
Date: September 2025
"""

import json
import os
import time
import uuid
from datetime import datetime
from typing import Dict, List

class DataLogger:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir

        self.session_dir = os.path.join(data_dir, "sessions")
        self.logs_dir = os.path.join(data_dir, "logs")
        self.exports_dir = os.path.join(data_dir, "exports")

        for directory in [self.session_dir, self.logs_dir, self.exports_dir]:
            os.makedirs(directory, exist_ok=True)

        self.current_sessions = {}
        print(f"DataLogger initialized: {data_dir}")

    def start_session(self, user_info: Dict = None) -> str:
        session_id = str(uuid.uuid4())[:8]
        start_time = datetime.now()

        session_data = {
            'session_id': session_id,
            'start_time': start_time.isoformat(),
            'user_info': user_info or {},
            'detections': [],
            'metadata': {'version': '1.0'}
        }

        self.current_sessions[session_id] = session_data
        print(f"Started session: {session_id}")
        return session_id

    def log_detection(self, session_id: str, difficulty_score: float, features: Dict):
        if session_id not in self.current_sessions:
            print(f"Unknown session: {session_id}")
            return

        timestamp = datetime.now()

        detection_data = {
            'timestamp': timestamp.isoformat(),
            'difficulty_score': difficulty_score,
            'features': self._sanitize_features(features)
        }

        self.current_sessions[session_id]['detections'].append(detection_data)

    def _sanitize_features(self, features: Dict) -> Dict:
        sanitized = {}

        for key, value in features.items():
            try:
                if isinstance(value, (int, float, str, bool, type(None))):
                    sanitized[key] = value
                elif hasattr(value, '__iter__') and not isinstance(value, str):
                    try:
                        import numpy as np
                        if isinstance(value, np.ndarray):
                            sanitized[key] = value.tolist()
                        else:
                            sanitized[key] = list(value)
                    except:
                        sanitized[key] = str(value)
                else:
                    sanitized[key] = str(value)
            except:
                sanitized[key] = None

        return sanitized

    def end_session(self, session_id: str, detection_history: List[Dict] = None):
        if session_id not in self.current_sessions:
            print(f"Unknown session: {session_id}")
            return

        session_data = self.current_sessions[session_id]
        end_time = datetime.now()

        detections = detection_history or session_data['detections']

        if detection_history and len(detection_history) > 0:
            converted_detections = []
            for detection in detection_history:
                converted_detection = {
                    'timestamp': datetime.fromtimestamp(detection.get('timestamp', time.time())).isoformat(),
                    'difficulty_score': detection.get('difficulty_score', 0),
                    'features': self._sanitize_features(detection.get('features', {}))
                }
                converted_detections.append(converted_detection)
            detections = converted_detections

        if detections:
            difficulty_scores = [d.get('difficulty_score', 0) for d in detections]
            avg_difficulty = sum(difficulty_scores) / len(difficulty_scores)
            max_difficulty = max(difficulty_scores)
        else:
            avg_difficulty = 0.0
            max_difficulty = 0.0

        start_time = datetime.fromisoformat(session_data['start_time'])
        duration = (end_time - start_time).total_seconds()

        session_data.update({
            'end_time': end_time.isoformat(),
            'duration': duration,
            'total_detections': len(detections),
            'avg_difficulty_score': avg_difficulty,
            'max_difficulty_score': max_difficulty,
            'detections': detections
        })

        session_file = os.path.join(self.session_dir, f"session_{session_id}.json")
        try:
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            print(f"Session saved: {session_file}")
        except Exception as e:
            print(f"Error saving session: {e}")

        del self.current_sessions[session_id]
        print(f"Session ended: {session_id}")

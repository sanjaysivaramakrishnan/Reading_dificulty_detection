"""
Settings Configuration - Application settings
Author: Sanjay Sivaramakrishnan M
Date: September 2025
"""

import json
import os
from typing import Dict, Any

class Settings:
    def __init__(self, config_file: str = "config/settings.json"):
        self.config_file = config_file

        self.defaults = {
            "video_resolution": "1280x720",
            "target_fps": 30,
            "camera_id": 0,
            "detection_sensitivity": 0.5,
            "difficulty_threshold": 0.5,
            "save_detection_data": True,
            "data_directory": "data"
        }

        self.settings = self.defaults.copy()
        self.load()

    def load(self):
        try:
            config_dir = os.path.dirname(self.config_file)
            if config_dir:
                os.makedirs(config_dir, exist_ok=True)

            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_settings = json.load(f)

                self.settings.update(user_settings)
                print(f"Settings loaded from {self.config_file}")
            else:
                self.save()
                print(f"Created default settings: {self.config_file}")
        except Exception as e:
            print(f"Failed to load settings: {e}")
            self.settings = self.defaults.copy()

    def save(self):
        try:
            config_dir = os.path.dirname(self.config_file)
            if config_dir:
                os.makedirs(config_dir, exist_ok=True)

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2)

            print(f"Settings saved: {self.config_file}")
        except Exception as e:
            print(f"Failed to save settings: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        return self.settings.get(key, default)

    def set(self, key: str, value: Any):
        self.settings[key] = value

    @property
    def detection_sensitivity(self) -> float:
        return self.get("detection_sensitivity", 0.5)

    @detection_sensitivity.setter
    def detection_sensitivity(self, value: float):
        self.set("detection_sensitivity", value)

    @property
    def save_detection_data(self) -> bool:
        return self.get("save_detection_data", True)

    @save_detection_data.setter
    def save_detection_data(self, value: bool):
        self.set("save_detection_data", value)

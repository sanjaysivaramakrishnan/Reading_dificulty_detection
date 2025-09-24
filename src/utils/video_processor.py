"""
Video Processor - Video capture and preprocessing
Author: Sanjay Sivaramakrishnan M
Date: September 2025
"""

import cv2
import numpy as np
import threading
import time
from typing import Optional

class VideoProcessor:
    def __init__(self, camera_id: int = 0, target_fps: int = 30):
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps

        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()

        self.frame_width = 1280
        self.frame_height = 720
        self.actual_fps = 0

        self.frame_count = 0
        self.last_fps_time = time.time()

    def initialize_camera(self) -> bool:
        try:
            for camera_idx in [self.camera_id, 0, 1, 2]:
                self.cap = cv2.VideoCapture(camera_idx)

                if self.cap.isOpened():
                    self.camera_id = camera_idx
                    break
                else:
                    if self.cap:
                        self.cap.release()
                        self.cap = None

            if not self.cap or not self.cap.isOpened():
                print("Cannot open camera")
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            ret, test_frame = self.cap.read()
            if not ret:
                print("Failed to capture test frame")
                self.cap.release()
                self.cap = None
                return False

            print(f"Camera {self.camera_id} initialized")
            return True

        except Exception as e:
            print(f"Camera init failed: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
            return False

    def start_capture(self) -> bool:
        if not self.initialize_camera():
            return False

        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        print("Video capture started")
        return True

    def stop_capture(self):
        self.is_running = False

        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=1.0)

        if self.cap:
            self.cap.release()
            self.cap = None

        print("Video capture stopped")

    def _capture_loop(self):
        consecutive_failures = 0
        max_failures = 5

        while self.is_running and self.cap and self.cap.isOpened():
            start_time = time.time()

            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    consecutive_failures = 0

                    processed_frame = self._preprocess_frame(frame)

                    with self.frame_lock:
                        self.current_frame = processed_frame.copy()

                    self._update_fps_counter()

                    elapsed = time.time() - start_time
                    sleep_time = max(0, self.frame_interval - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        print("Too many failures, stopping")
                        break
                    time.sleep(0.1)

            except Exception as e:
                consecutive_failures += 1
                print(f"Capture error: {e}")
                if consecutive_failures >= max_failures:
                    break
                time.sleep(0.1)

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        try:
            frame = cv2.flip(frame, 1)

            if frame.shape[0] < 240 or frame.shape[1] < 320:
                frame = cv2.resize(frame, (640, 480))

            if len(frame.shape) == 3 and frame.shape[2] == 3:
                return frame
            elif len(frame.shape) == 3 and frame.shape[2] == 4:
                return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            else:
                return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return frame

    def get_current_frame(self) -> Optional[np.ndarray]:
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def _update_fps_counter(self):
        self.frame_count += 1
        current_time = time.time()

        if current_time - self.last_fps_time >= 1.0:
            self.actual_fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = current_time

    def get_fps(self) -> float:
        return self.actual_fps

    def is_camera_available(self) -> bool:
        return self.cap is not None and self.cap.isOpened() and self.is_running

#!/usr/bin/env python3
"""
Reading Difficulty Detection System - Main Application
Author: Sanjay Sivaramakrishnan M
Date: September 2025
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from datetime import datetime
import os
from PIL import Image, ImageTk
import sys

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.models.reading_detector import ReadingDifficultyDetector
    from src.utils.video_processor import VideoProcessor
    from src.utils.data_logger import DataLogger
    from config.settings import Settings
except ImportError as e:
    print(f"Import error: {e}")
    messagebox.showerror("Import Error", f"Missing dependencies: {e}\nPlease run: pip install -r requirements.txt")

class ReadingDifficultyApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Reading Difficulty Detection System v1.0")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')

        try:
            self.settings = Settings()
            self.detector = ReadingDifficultyDetector()
            self.video_processor = VideoProcessor()
            self.data_logger = DataLogger()
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize: {e}")
            return

        self.is_analyzing = False
        self.current_session = None
        self.detection_history = []
        self.session_start_time = None

        self.setup_gui()
        self.setup_video_capture()

    def setup_gui(self):
        # Configure main layout
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(header_frame, text="Reading Difficulty Detection System v1.0", 
                 font=("Arial", 16, "bold")).pack(side=tk.LEFT)

        # Status
        status_frame = ttk.Frame(header_frame)
        status_frame.pack(side=tk.RIGHT)

        self.camera_status_label = ttk.Label(status_frame, text="Camera: Initializing...")
        self.camera_status_label.pack(anchor=tk.E)

        self.detection_status_label = ttk.Label(status_frame, text="Detection: Stopped")
        self.detection_status_label.pack(anchor=tk.E)

        # Video display
        video_frame = ttk.LabelFrame(main_frame, text="Live Video Feed", padding="5")
        video_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=(0, 5))

        self.video_label = ttk.Label(video_frame, text="Camera initializing...")
        self.video_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        video_frame.rowconfigure(0, weight=1)
        video_frame.columnconfigure(0, weight=1)

        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls & Results", padding="5")
        control_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)

        self.start_btn = ttk.Button(button_frame, text="Start Detection", command=self.start_detection)
        self.start_btn.grid(row=0, column=0, padx=2, pady=2, sticky=(tk.W, tk.E))

        self.stop_btn = ttk.Button(button_frame, text="Stop Detection", command=self.stop_detection, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=2, pady=2, sticky=(tk.W, tk.E))

        self.export_btn = ttk.Button(button_frame, text="Export Data", command=self.export_data)
        self.export_btn.grid(row=1, column=0, columnspan=2, padx=2, pady=2, sticky=(tk.W, tk.E))

        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        # Difficulty display
        score_frame = ttk.LabelFrame(control_frame, text="Difficulty Score", padding="5")
        score_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)

        self.difficulty_score_var = tk.StringVar(value="0.00")
        self.difficulty_score_label = ttk.Label(score_frame, textvariable=self.difficulty_score_var,
                                               font=("Arial", 32, "bold"), foreground="green")
        self.difficulty_score_label.pack()

        self.difficulty_progress = ttk.Progressbar(score_frame, length=200, mode='determinate')
        self.difficulty_progress.pack(fill=tk.X, pady=5)

        self.status_text_var = tk.StringVar(value="Normal Reading")
        ttk.Label(score_frame, textvariable=self.status_text_var).pack()

        # Metrics
        metrics_frame = ttk.LabelFrame(control_frame, text="Live Metrics", padding="5")
        metrics_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        self.metrics_text = tk.Text(metrics_frame, height=10, width=35, font=("Consolas", 8))
        self.metrics_text.pack(fill=tk.BOTH, expand=True)

        # Session info
        session_frame = ttk.LabelFrame(main_frame, text="Session Info", padding="5")
        session_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        info_grid = ttk.Frame(session_frame)
        info_grid.pack(fill=tk.X)

        ttk.Label(info_grid, text="Duration:").grid(row=0, column=0)
        self.duration_var = tk.StringVar(value="00:00:00")
        ttk.Label(info_grid, textvariable=self.duration_var).grid(row=0, column=1, padx=10)

        ttk.Label(info_grid, text="Samples:").grid(row=0, column=2)
        self.samples_var = tk.StringVar(value="0")
        ttk.Label(info_grid, textvariable=self.samples_var).grid(row=0, column=3, padx=10)

    def setup_video_capture(self):
        try:
            if self.video_processor.start_capture():
                self.camera_status_label.config(text="Camera: Ready", foreground="green")
                self.video_thread = threading.Thread(target=self.update_video_feed, daemon=True)
                self.video_thread.start()
            else:
                self.camera_status_label.config(text="Camera: Error", foreground="red")
        except Exception as e:
            self.camera_status_label.config(text="Camera: Error", foreground="red")

    def update_video_feed(self):
        while True:
            try:
                frame = self.video_processor.get_current_frame()
                if frame is not None:
                    if self.is_analyzing:
                        processed_frame, metrics = self.process_frame(frame)
                        self.root.after(0, lambda: self.update_metrics_display(metrics))
                    else:
                        processed_frame = frame

                    display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    display_frame = cv2.resize(display_frame, (640, 480))

                    image = Image.fromarray(display_frame)
                    photo = ImageTk.PhotoImage(image)
                    self.root.after(0, lambda p=photo: self.update_video_display(p))

                time.sleep(0.033)
            except Exception as e:
                print(f"Video error: {e}")
                time.sleep(0.1)

    def update_video_display(self, photo):
        try:
            self.video_label.config(image=photo, text="")
            self.video_label.image = photo
        except:
            pass

    def process_frame(self, frame):
        try:
            difficulty_score, features = self.detector.predict_reading_difficulty(frame)

            self.difficulty_score_var.set(f"{difficulty_score:.2f}")
            self.difficulty_progress.config(value=difficulty_score * 100)

            if difficulty_score < 0.3:
                color, status = "green", "Normal Reading"
            elif difficulty_score < 0.7:
                color, status = "orange", "Slight Difficulty"
            else:
                color, status = "red", "Reading Difficulty"

            self.difficulty_score_label.config(foreground=color)
            self.status_text_var.set(status)

            detection_data = {
                'timestamp': time.time(),
                'difficulty_score': difficulty_score,
                'features': features
            }
            self.detection_history.append(detection_data)
            self.samples_var.set(str(len(self.detection_history)))

            if self.current_session:
                self.data_logger.log_detection(self.current_session, difficulty_score, features)

            return frame, features
        except Exception as e:
            return frame, {}

    def update_metrics_display(self, features):
        try:
            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(tk.END, f"Last Update: {datetime.now().strftime('%H:%M:%S')}\n\n")

            for key, value in features.items():
                if isinstance(value, (int, float)) and not key.startswith('left_eye'):
                    self.metrics_text.insert(tk.END, f"{key.replace('_', ' ').title()}: {value:.4f}\n")
        except:
            pass

    def start_detection(self):
        try:
            self.is_analyzing = True
            self.current_session = self.data_logger.start_session()
            self.detection_history.clear()

            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.detection_status_label.config(text="Detection: Running", foreground="green")

            self.session_start_time = time.time()
            self.update_session_timer()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start: {e}")

    def stop_detection(self):
        try:
            self.is_analyzing = False

            if self.current_session:
                self.data_logger.end_session(self.current_session, self.detection_history)

            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            self.detection_status_label.config(text="Detection: Stopped", foreground="red")

            samples = len(self.detection_history)
            self.current_session = None
            self.duration_var.set("00:00:00")
            self.samples_var.set("0")

            messagebox.showinfo("Stopped", f"Session completed! Samples: {samples}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop: {e}")

    def update_session_timer(self):
        if self.is_analyzing and self.session_start_time:
            try:
                duration = time.time() - self.session_start_time
                hours = int(duration // 3600)
                minutes = int((duration % 3600) // 60)
                seconds = int(duration % 60)
                self.duration_var.set(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
                self.root.after(1000, self.update_session_timer)
            except:
                pass

    def export_data(self):
        try:
            if not self.detection_history:
                messagebox.showwarning("No Data", "No data to export.")
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_data_{timestamp}.csv"

            import csv
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'difficulty_score'])
                for data in self.detection_history:
                    writer.writerow([data['timestamp'], data['difficulty_score']])

            messagebox.showinfo("Export Complete", f"Data exported to {filename}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export: {e}")

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        try:
            if self.is_analyzing:
                if messagebox.askyesno("Active Detection", "Stop detection and close?"):
                    self.stop_detection()
                else:
                    return
            self.video_processor.stop_capture()
            cv2.destroyAllWindows()
        except:
            pass
        self.root.destroy()

def main():
    if not os.path.exists('src'):
        messagebox.showerror("Error", "Please run from project directory")
        return

    app = ReadingDifficultyApp()
    app.run()

if __name__ == "__main__":
    main()

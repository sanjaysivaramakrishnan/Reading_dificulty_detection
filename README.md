# Reading Difficulty Detection System v1.0

A complete computer vision system for detecting reading difficulties using webcam input in real-time.

## 🚀 Quick Start

1. **Install Python 3.7+**
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Run the application**: `python main.py`
4. **Click "Start Detection"** and begin using!

## 🌟 Features

- **Real-time Detection**: 30+ FPS webcam analysis
- **MediaPipe Integration**: Advanced facial landmark detection  
- **Professional GUI**: User-friendly interface with live video feed
- **Session Management**: Data logging and export capabilities
- **Privacy-First**: All processing happens locally

## 📋 System Requirements

- **Python**: 3.7 or higher
- **Camera**: Webcam or USB camera device
- **RAM**: 4GB+ recommended
- **OS**: Windows, macOS, or Linux

## 🎯 How to Use

1. **Launch**: Run `python main.py`
2. **Grant camera permissions** when prompted by your system
3. **Click "Start Detection"** to begin real-time analysis
4. **View results** in the interface:
   - **Green (0.0-0.3)**: Normal reading behavior
   - **Orange (0.3-0.7)**: Some difficulty detected
   - **Red (0.7-1.0)**: Significant reading difficulty
5. **Stop detection** when finished and export data if needed

## 🔧 Technical Details

- **Computer Vision**: OpenCV + MediaPipe Face Mesh
- **Detection Method**: Rule-based analysis with temporal patterns
- **Performance**: Real-time processing at 30+ FPS
- **Data Storage**: JSON sessions with CSV export capability

## 🚨 Troubleshooting

### Camera Issues
- Check camera permissions in system settings
- Close other applications using the camera
- Try running as administrator (Windows)

### Installation Issues
- Ensure Python 3.7+ is installed
- Update pip: `pip install --upgrade pip`
- Install packages individually if batch fails

### Performance Issues
- Reduce video resolution in settings
- Close resource-intensive applications
- Ensure adequate lighting for face detection

## ⚖️ Privacy & Ethics

- **100% Local Processing**: No data transmission to external servers
- **User Control**: All data collection is optional and user-controlled
- **Privacy Protection**: Only facial landmarks analyzed, no personal data stored
- **Professional Use**: Results should supplement, not replace, professional assessment

## 📊 Use Cases

- Educational assessment and research
- Reading difficulty identification
- Accessibility tool development
- Tutoring application integration
- Personal reading behavior analysis

---

**Author**: Punda Prakash  
**Version**: 1.0  
**Date**: September 2025  
**License**: Educational and Research Use

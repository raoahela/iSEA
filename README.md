# iSEA: Intelligent Seafloor & Animal Image Annotator
## Exploring the deep sea with intelligence
---

iSEA is a video annotation platform specifically designed for marine biology research. It combines automated object detection with manual annotation tools to accelerate the analysis of deep-sea imagery.

---

## 📋 Overview

iSEA (Intelligent Seafloor & Animal Image Annotator) is a PyQt6-based graphical interface that integrates state-of-the-art computer vision models to assist marine researchers in annotating underwater imagery. The tool supports both automatic detection using YOLO and interactive segmentation using SAM 2.

---

## 🚀 Key Features

### **Core Functionality**
- **Dual-mode operation**: Automatic YOLO detection + manual annotation
- **SAM 2 Integration**: Click-based segmentation refinement inside bounding boxes
- **Live mode**: Direct camera feed support with recording
- **Training pipeline**: Export annotations to YOLO format and train custom models
- **Georeferencing**: Merge annotations with navigation data


### **Performance & UX**
- Dark/Light mode for long annotation sessions
- Detection history with filtering by taxon/confidence
- Keyboard shortcuts for efficient navigation

---

### **Dependencies**
```bash
pip install -r requirements.txt
# iSEA: Intelligent Seafloor & Animal Image Annotator

> **A PyQt6-based video annotation platform for marine biology research, combining YOLO object detection, SAM 2 interactive segmentation, and seafloor classification.**

---

## 📋 Overview

iSEA is a video annotation platform specifically designed for marine biology research. It integrates state-of-the-art computer vision models to assist researchers in annotating underwater imagery efficiently.

The tool supports:
- **Automatic detection** using YOLO
- **Interactive segmentation** using SAM 
- **Manual bounding box annotation** with class selection
- **Continuous seafloor classification** with keyboard shortcuts
- **Live camera feed** support with recording
- **Training pipeline** to export annotations and train custom YOLO models
- **Georeferencing** by merging annotations with navigation data

---

## 🚀 Key Features

### **Core Functionality**
| Feature | Description |
|---------|-------------|
| **YOLO Auto-Detection** | Automatic object detection with confidence filtering and object tracking |
| **SAM Segmentation** | Click-based interactive segmentation with hover preview and mask confirmation |
| **Manual Annotation** | Draw bounding boxes manually with class selection from taxon grid |
| **Live Mode** | Direct camera feed support with recording |
| **Seafloor Classification** | Real-time and offline classification with dynamic class management |
| **Training Pipeline** | Export to YOLO format (detection & segmentation) and train custom models |
| **Georeferencing** | Merge annotation timestamps with navigation CSV data |

### **Annotation Workflow**
- **Detection history dock** with filtering by taxon, confidence, and type
- **Taxon grid** for quick class selection
- **Keyboard shortcuts** for all major operations
- **Dark/Light mode** toggle for long sessions
- **Session persistence** via dataset import/export

---

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended for YOLO tracking and SAM)

### Setup

```bash
# Clone the repository
git clone https://github.com/raoahela/iSEA.git
cd iSEA

# Install dependencies
pip install -r requirements.txt

```

---

## 🎮 Usage

### Starting the Application
```bash
python main.py
```

### Loading a Video
1. **File → Open Video** (`Ctrl+O`) or drag & drop a video file
2. Supported formats: MP4, AVI, MOV, MKV, M4V, FLV, WMV

### Playback Controls
| Action | Shortcut |
|--------|----------|
| Play/Pause | `Space` |
| Previous 30 frames | `←` |
| Next 30 frames | `→` |
| Seek | Slider drag |

### Annotation Modes

#### **YOLO Auto-Detection**
- Press `D` for single-frame detection
- Press `T` to toggle **continuous detection** (runs on every frame during playback)
- Press **2.0×** button for accelerated detection (every N frames via async thread)

#### **SAM Segmentation**
- Press `A` to toggle hover segmentation mode
- Move mouse over video — SAM generates mask preview in real-time
- Click to confirm and save segmentation mask
- Supports both video and dataset (image) modes

#### **Manual Bounding Boxes**
- Press `M` to enable manual annotation mode
- Select class from **Taxon Grid** (`Ctrl+T` to show/hide)
- Click and drag on video to draw bounding boxes

#### **Seafloor Classification**
Real-time classification during video playback:

| Shortcut | Class | Action |
|----------|-------|--------|
| `S` | Sediment | Start collecting soft sediment frames |
| `F` | Coral_Fragment | Start collecting coral fragment frames |
| `R` | Coral_Reef | Start collecting coral reef frames |
| `Shift+S` | — | Stop current annotation & save segment |

- Frames are saved every 30 frames to `seafloor_training_data/<class>/`
- Classify single frame: **Seafloor → Classify Current Frame** (`Ctrl+F`)
- Train classifier from collected data: **Seafloor → Train from Collected Data**
- Manage custom categories: **Seafloor → Manage Categories**

### Detection History
- **Dock panel** (`Ctrl+H`) lists all detections with:
  - Frame number, timestamp, class, confidence, type (auto/manual/segmentation/seafloor)
  - Track ID for tracked objects
  - Click to jump to frame
- Filter by class and confidence threshold

---

## 📤 Export & Training

### YOLO Detection Export
**Training → Export YOLO Annotations**
- Exports manual bounding boxes to `images/train`, `images/val`, `labels/train`, `labels/val`
- Automatically splits 80/20
- Generates/updates `dataset.yaml`
- Preserves existing class IDs when appending to existing dataset

### YOLO Segmentation Export
**Training → Export Segmentation**
- Exports SAM 2 confirmed masks as YOLO segmentation format (polygons)
- Same directory structure as detection export

### Training Custom Models
**Training → Train YOLO Model**
- Configure epochs, batch size, image size, learning rate, device (CPU/GPU)
- Trains from exported dataset or existing dataset folder
- Model saved to `models/<name>/weights/best.pt`

**Training → Train Segmentation Model**
- Trains YOLO segmentation model from exported segmentation dataset

### Dataset Import
**Training → Import YOLO Dataset**
- Load existing YOLO dataset with `dataset.yaml`
- Parses `labels/train/` and `labels/val/` annotations
- Populates taxon grid with dataset classes

---

## ⌨️ Complete Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Space` | Play / Pause video |
| `←` / `→` | Previous / Next 30 frames |
| `Ctrl+O` | Open video |
| `Ctrl+M` | Load custom YOLO model |
| `Ctrl+Y` | Load dataset YAML |
| `Ctrl+W` | Toggle live camera mode |
| `Ctrl+S` | Save annotations to CSV |
| `Ctrl+R` | Start recording (live mode) |
| `Ctrl+Shift+R` | Stop recording |
| `Ctrl+L` | Load annotations from CSV |
| `Ctrl+H` | Show/hide detection history |
| `Ctrl+T` | Show/hide taxon grid |
| `Ctrl+Q` | Exit application |
| `D` | Detect objects (single frame) |
| `T` | Toggle continuous detection |
| `M` | Toggle manual annotation mode |
| `A` | Toggle SAM 2 hover segmentation |
| `E` | Open taxonomy enrichment dialog |
| `S` | Seafloor: Sediment |
| `F` | Seafloor: Coral_Fragment |
| `R` | Seafloor: Coral_Reef |
| `Shift+S` | Stop seafloor annotation |
| `Ctrl+F` | Classify current frame (seafloor) |
| `F1` | Show shortcuts help |

---

## 📊 Data Export Format

### Annotations CSV
Saved annotations include:
- `Video` — Source video filename
- `Timestamp` — Video timestamp (HH:MM:SS)
- `System_Date`, `System_Time`, `System_Timezone` — System metadata
- `Taxon` — Class name
- `Confidence` — Detection confidence (1.0 for manual)
- `Type` — `auto`, `manual`, `segmentation`, or `seafloor`
- `Track_ID` — Object tracking ID (auto-detections)
- `x1`, `y1`, `x2`, `y2` — Bounding box coordinates
- `Frame_Number` — Frame index
- `Seafloor` — Seafloor class for that frame (if annotated)
- `Photo` — Path to extracted frame image

### Recording Export
When saving a live recording:
- Video file (AVI)
- Frame images with detections drawn
- `_annotations.csv` with all detections from the recording session

---


## 🤝 Contributing

This project is under active development. For bug reports or feature requests, please open an issue on GitHub.

---

## 📚 Citation

If you use iSEA in your research, please cite:

```bibtex
@software{isea2026,
  author = {Lopes, R. N.},
  title = {iSEA: Intelligent Seafloor and Animal Image Annotator},
  url = {https://github.com/raoahela/iSEA},
  year = {2026},
}
```
---
## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE).

This application integrates Ultralytics YOLO, which is also licensed under AGPL-3.0.

> **Status**: Work in Progress — Features and interface may evolve.

"""
Seafloor Classification Module for iSEA
Integrates AI-SCW workflow into the iSEA video annotation platform
Classes: Sedimento, Coral_Fragmento, Coral_Vivo
"""

import cv2
import os
import shutil
import colorsys
import json
import numpy as np
from pathlib import Path
from collections import Counter
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt backend so it works inside PyQt6 app
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition, QMutexLocker, Qt, QSize
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QProgressBar, QTextEdit, QMessageBox,
                             QComboBox, QSpinBox, QDoubleSpinBox, QFormLayout,
                             QDialogButtonBox, QFileDialog, QGroupBox, QApplication,
                             QGridLayout, QScrollArea, QWidget, QListWidget, QListWidgetItem,
                             QLineEdit, QColorDialog)
from PyQt6.QtGui import QImage, QPixmap, QColor, QIcon


class CenterCropTo299:
    """Crop central para 299×299, removendo bordas escuras."""
    def __init__(self, crop_ratio=0.85):
        self.crop_ratio = np.clip(crop_ratio, 0.3, 1.0)
        self.target_size = 299
    
    def crop(self, img):
        h, w = img.shape[:2]
        region_h = int(h * self.crop_ratio)
        region_w = int(w * self.crop_ratio)
        y1 = (h - region_h) // 2
        x1 = (w - region_w) // 2
        y2 = y1 + region_h
        x2 = x1 + region_w
        
        region = img[y1:y2, x1:x2].copy()
        rh, rw = region.shape[:2]
        
        if rh >= self.target_size and rw >= self.target_size:
            cy = (rh - self.target_size) // 2
            cx = (rw - self.target_size) // 2
            result = region[cy:cy+self.target_size, cx:cx+self.target_size].copy()
            method = "crop"
        else:
            scale = min(self.target_size / rw, self.target_size / rh)
            new_w = int(rw * scale)
            new_h = int(rh * scale)
            resized = cv2.resize(region, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            result = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
            y_off = (self.target_size - new_h) // 2
            x_off = (self.target_size - new_w) // 2
            result[y_off:y_off+new_h, x_off:x_off+new_w] = resized
            method = "resize+pad"
        
        return result, {
            'original': (w, h), 'region': (rw, rh),
            'method': method, 'crop_box': (x1, y1, x2, y2)
        }
    
    def crop_batch(self, images, image_names):
        cropped = []
        infos = []
        for img, name in zip(images, image_names):
            c, info = self.crop(img)
            cropped.append(c)
            infos.append(info)
        return cropped, infos


# =============================================================================
# COLOR NORMALIZATION 
# =============================================================================

class ColorNormalizer:
    def __init__(self, reference_path=None, auto_select=True):
        self.reference_path = reference_path
        self.auto_select = auto_select
        self.ref_mean = None
        self.ref_std = None

    def select_reference(self, image_paths, folder_path):
        if self.reference_path and os.path.exists(self.reference_path):
            return cv2.imread(self.reference_path)

        entropies = []
        valid_paths = []
        for name in image_paths:
            img = cv2.imread(os.path.join(folder_path, name))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ent = shannon_entropy(gray)
            entropies.append(ent)
            valid_paths.append(name)

        median_idx = np.argsort(entropies)[len(entropies) // 2]
        ref_name = valid_paths[median_idx]
        return cv2.imread(os.path.join(folder_path, ref_name))

    def compute_reference_stats(self, ref_img):
        ref_lab = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        self.ref_mean = np.mean(ref_lab, axis=(0, 1))
        self.ref_std = np.std(ref_lab, axis=(0, 1))
        return self.ref_mean, self.ref_std

    def normalize(self, img):
        if self.ref_mean is None:
            raise ValueError("Reference stats not computed.")

        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        src_mean = np.mean(img_lab, axis=(0, 1))
        src_std = np.std(img_lab, axis=(0, 1))
        src_std = np.where(src_std == 0, 1, src_std)

        normalized = (img_lab - src_mean) * (self.ref_std / src_std) + self.ref_mean
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        return cv2.cvtColor(normalized, cv2.COLOR_LAB2BGR)

    def normalize_dataset(self, images, image_names, output_dir=None):
        normalized = []
        for i, (img, name) in enumerate(zip(images, image_names)):
            norm_img = self.normalize(img)
            normalized.append(norm_img)
            if output_dir:
                out_path = os.path.join(output_dir, f"norm_{name}")
                cv2.imwrite(out_path, norm_img)
        return normalized


# =============================================================================
# DATA AUGMENTATION
# =============================================================================

def augment_image(img):
    aug_img = img.copy()
    if np.random.rand() > 0.5:
        aug_img = cv2.flip(aug_img, 1)
    angle = np.random.uniform(-15, 15)
    h, w = aug_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aug_img = cv2.warpAffine(aug_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    alpha = np.random.uniform(0.9, 1.1)
    beta = np.random.uniform(-15, 15)
    aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)
    aug_img = np.clip(aug_img, 0, 255).astype(np.uint8)
    return aug_img


# =============================================================================
# ENTROPY CLASSIFIER (NOVO - de image_clustering2.py)
# =============================================================================

class EntropyClassifier:
    """
    Classificador baseado em entropia com thresholds otimizados para S/F/R.
    Usado para pseudo-labeling e classificador hierárquico.
    """
    def __init__(self, thresholds=(6.471, 6.980), class_names=None):
        self.thresholds = thresholds
        self.class_names = class_names or ["Sedimento", "Coral_Fragmento", "Coral_Vivo"]
        self.entropies = None
        self.pseudo_labels = None
        self.confidence_scores = None
        self.entropy_features = None

    def compute_entropies(self, images):
        self.entropies = []
        for img in images:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            self.entropies.append(shannon_entropy(gray))
        self.entropies = np.array(self.entropies)
        return self.entropies

    def predict(self, images=None):
        if images is not None:
            self.compute_entropies(images)
        labels = np.zeros(len(self.entropies), dtype=int)
        labels[self.entropies >= self.thresholds[0]] = 1
        labels[self.entropies >= self.thresholds[1]] = 2
        return labels

    def get_pseudo_labels(self, images=None, margin=0.25):
        """Retorna pseudo-labels apenas para amostras com alta confiança."""
        if images is not None:
            self.compute_entropies(images)
        labels = np.full(len(self.entropies), -1, dtype=int)
        confidences = np.zeros(len(self.entropies))
        class_centers = [6.192, 6.750, 7.209]  # Centros conhecidos S/F/R
        
        for i, e in enumerate(self.entropies):
            dists = [abs(e - c) for c in class_centers]
            min_dist = min(dists)
            sorted_dists = sorted(dists)
            # Só aceita se estiver próximo do centro E longe do vizinho mais próximo
            if min_dist < margin and (sorted_dists[1] - min_dist) > margin * 0.5:
                labels[i] = np.argmin(dists)
                confidences[i] = 1.0 - (min_dist / margin)
        
        self.pseudo_labels = labels
        self.confidence_scores = confidences
        return labels, confidences

    def get_entropy_color(self, entropy):
        if entropy < self.thresholds[0]:
            return '#2ecc71'
        elif entropy < self.thresholds[1]:
            return '#f39c12'
        else:
            return '#e74c3c'

    def get_stats(self):
        if self.entropies is None:
            return None
        return {
            'mean': np.mean(self.entropies),
            'std': np.std(self.entropies),
            'min': np.min(self.entropies),
            'max': np.max(self.entropies),
            'median': np.median(self.entropies)
        }

    def print_report(self, labels=None):
        print("\n" + "=" * 60)
        print("RELATORIO: ENTROPIA E CLASSE")
        print("=" * 60)
        print(f"Total de imagens: {len(self.entropies)}")
        stats = self.get_stats()
        print(f"  Entropia global: media={stats['mean']:.4f}, std={stats['std']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        if labels is not None:
            print("\n  Por classe:")
            for i, name in enumerate(self.class_names):
                mask = labels == i
                n = np.sum(mask)
                if n > 0:
                    mean = np.mean(self.entropies[mask])
                    std = np.std(self.entropies[mask])
                    print(f"    Classe {name}: n={n}, media={mean:.4f}, std={std:.4f}")
        if self.pseudo_labels is not None:
            n_safe = np.sum(self.pseudo_labels != -1)
            print(f"\n  Pseudo-labels seguros: {n_safe}/{len(self.entropies)} ({n_safe/len(self.entropies):.1%})")
        print("=" * 60)


# =============================================================================
# SEMI-AUTOMATIC LABELER 
# =============================================================================

class SemiAutoLabeler:
    def __init__(self, n_examples_per_class=10):
        self.n_examples = n_examples_per_class
        self.labels = {}
        self.class_names = {}

    def extract_domain_features(self, images):
        """
        Features AVANÇADAS otimizadas para separar S/F/R.
        Entropia replicada para dar peso dominante no PCA.
        """
        features = []
        for img in images:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img

            h, w = gray.shape

            # ===== ENTROPIA (peso máximo - replicada 4x) =====
            global_entropy = shannon_entropy(gray)

            # Entropia em múltiplas escalas
            if h > 64 and w > 64:
                small = cv2.resize(gray, (w//4, h//4), interpolation=cv2.INTER_AREA)
                med = cv2.resize(gray, (w//2, h//2), interpolation=cv2.INTER_AREA)
                ent_small = shannon_entropy(small)
                ent_med = shannon_entropy(med)
            else:
                ent_small = global_entropy
                ent_med = global_entropy

            # Entropia por quadrantes (heterogeneidade espacial)
            q1 = gray[:h//2, :w//2]
            q2 = gray[:h//2, w//2:]
            q3 = gray[h//2:, :w//2]
            q4 = gray[h//2:, w//2:]
            quad_ents = [shannon_entropy(q) for q in [q1, q2, q3, q4] if q.size > 0]

            # Entropia do gradiente (bordas = complexidade)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(sobelx**2 + sobely**2)
            grad_entropy = shannon_entropy((grad_mag / (grad_mag.max() + 1e-8) * 255).astype(np.uint8))

            # ===== ESTATÍSTICAS DE INTENSIDADE =====
            mean_int = np.mean(gray)
            std_int = np.std(gray)

            # Histograma normalizado (percentis compactos)
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
            hist = hist / (hist.sum() + 1e-10)
            cumhist = np.cumsum(hist)
            p25 = np.searchsorted(cumhist, 0.25) * 8
            p50 = np.searchsorted(cumhist, 0.50) * 8
            p75 = np.searchsorted(cumhist, 0.75) * 8

            # ===== GLCM (textura com 2 distâncias/ângulos) =====
            glcm = graycomatrix(gray, distances=[1, 2], angles=[0, np.pi/4], 
                               levels=256, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            energy = graycoprops(glcm, 'energy').mean()

            # ===== MONTAR FEATURE VECTOR =====
            entropy_features = [global_entropy] * 4 + [ent_small, ent_med, grad_entropy]
            spatial_features = quad_ents + [np.mean(quad_ents), np.std(quad_ents)]
            intensity_features = [mean_int, std_int, p25, p50, p75]
            texture_features = [contrast, homogeneity, energy]

            feat = np.concatenate([
                entropy_features,      # 7 features (peso máximo)
                spatial_features,      # 6 features
                intensity_features,    # 5 features
                texture_features       # 3 features
            ])
            features.append(feat)

        features = np.array(features)
        print(f"  Domain features: {features.shape[1]} dims (entropy={7}, spatial={6}, intensity={5}, texture={3})")
        return features

    def interactive_labeling_qt(self, images, image_names, pca_2d, n_classes, 
                                 class_names_list=None, parent_widget=None):
        """
        INTERACTIVE labeling usando Qt Dialog (não bloqueante).
        """
        if class_names_list is None:
            class_names_list = [f"Class_{i}" for i in range(n_classes)]

        all_labels = {}
        
        for cls in range(n_classes):
            cls_name = class_names_list[cls] if cls < len(class_names_list) else f"Class_{cls}"
            self.class_names[cls] = cls_name
            
            dialog = InteractiveLabelingDialog(
                images, image_names, pca_2d, cls_name, cls,
                self.n_examples, parent=parent_widget
            )
            
            result = dialog.exec()
            
            if result == QDialog.DialogCode.Accepted and dialog.selected_indices:
                for idx in dialog.selected_indices:
                    all_labels[idx] = cls
                print(f"  '{cls_name}': {len(dialog.selected_indices)} examples selected by user")
            else:
                # Fallback to auto-selection
                auto = self._auto_select_uniform(pca_2d, images, cls,
                                                 exclude=list(all_labels.keys()))
                for idx in auto[:self.n_examples]:
                    all_labels[idx] = cls
                print(f"  '{cls_name}': {len(auto[:self.n_examples])} examples (auto-selected)")

        self.labels = all_labels
        return self.labels

    def _auto_select_uniform(self, pca_2d, images, cls_id, exclude=None):
        if exclude is None:
            exclude = list(self.labels.keys())
        available = [i for i in range(len(images)) if i not in exclude]
        if not available:
            return []

        selected = []
        n_pick = min(self.n_examples, len(available))
        kmeans = KMeans(n_clusters=n_pick, random_state=42 + cls_id, n_init=10)
        clusters = kmeans.fit_predict(pca_2d[available])

        for c in range(n_pick):
            cluster_indices = [available[i] for i, cl in enumerate(clusters) if cl == c]
            if cluster_indices:
                centroid = kmeans.cluster_centers_[c]
                dists = np.linalg.norm(pca_2d[cluster_indices] - centroid, axis=1)
                best = cluster_indices[np.argmin(dists)]
                selected.append(best)
        return selected

    def expand_labels_nearest_neighbors(self, images, n_neighbors=50):
        features = self.extract_domain_features(images)
        features_scaled = StandardScaler().fit_transform(features)
        features_pca = PCA(n_components=min(50, features_scaled.shape[1])).fit_transform(features_scaled)

        labeled_indices = list(self.labels.keys())
        if len(labeled_indices) == 0:
            return self.labels
            
        labeled_features = features_pca[labeled_indices]

        nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(labeled_indices)), 
                             metric='euclidean')
        nn.fit(labeled_features)

        new_labels = dict(self.labels)
        for i in range(len(images)):
            if i in self.labels:
                continue
            distances, indices = nn.kneighbors(features_pca[i:i+1])
            neighbor_labels = []
            neighbor_weights = []
            for dist, idx in zip(distances[0], indices[0]):
                real_idx = labeled_indices[idx]
                neighbor_labels.append(self.labels[real_idx])
                neighbor_weights.append(1.0 / (dist + 1e-6))

            if len(neighbor_labels) > 0:
                votes = {}
                for lbl, w in zip(neighbor_labels, neighbor_weights):
                    votes[lbl] = votes.get(lbl, 0) + w
                new_labels[i] = max(votes, key=votes.get)

        self.labels = new_labels
        return new_labels

    def get_training_data_balanced(self, images, image_names, augment_factor=3):
        X_train, y_train, names_train = [], [], []
        for idx, label in self.labels.items():
            X_train.append(images[idx])
            y_train.append(label)
            names_train.append(image_names[idx])

        y_train = np.array(y_train)
        unique_labels = sorted(np.unique(y_train))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        y_train = np.array([label_map[l] for l in y_train])

        class_counts = Counter(y_train)
        max_count = max(class_counts.values())

        X_balanced, y_balanced = [], []
        for cls in unique_labels:
            cls_indices = np.where(y_train == cls)[0]
            cls_images = [X_train[i] for i in cls_indices]
            cls_count = len(cls_images)

            X_balanced.extend(cls_images)
            y_balanced.extend([cls] * cls_count)

            n_augment = min((max_count - cls_count) * augment_factor, max_count * 2)
            for _ in range(n_augment):
                img = cls_images[np.random.randint(len(cls_images))]
                aug_img = augment_image(img)
                X_balanced.append(aug_img)
                y_balanced.append(cls)

        return np.array(X_balanced), np.array(y_balanced), names_train


# =============================================================================
# INTERACTIVE LABELING DIALOG 
# =============================================================================

class InteractiveLabelingDialog(QDialog):
    """Dialog for interactive selection of training examples per class."""
    
    def __init__(self, images, image_names, pca_2d, class_name, class_id,
                 n_examples, parent=None):
        super().__init__(parent)
        self.images = images
        self.image_names = image_names
        self.pca_2d = pca_2d
        self.class_name = class_name
        self.class_id = class_id
        self.n_examples = n_examples
        self.selected_indices = []
        
        self.setWindowTitle(f"Selecionar exemplos: {class_name}")
        self.resize(900, 700)
        
        self._build_ui()
        self._populate_grid()
        
    def _build_ui(self):
        layout = QVBoxLayout(self)
        
        # Instructions
        self.instruction = QLabel(
            f"<b>Clique nas imagens para selecionar exemplos de '{self.class_name}'</b><br>"
            f"Selecionados: <span style='color: red;'>0/{self.n_examples}</span><br>"
            f"Selecione imagens que representam bem a classe, espalhadas por diferentes regiões."
        )
        self.instruction.setWordWrap(True)
        layout.addWidget(self.instruction)
        
        # Scroll area for thumbnails
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(5)
        scroll.setWidget(self.grid_widget)
        layout.addWidget(scroll)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.auto_btn = QPushButton("Auto-selecionar")
        self.auto_btn.setToolTip("Deixar o algoritmo escolher automaticamente")
        self.auto_btn.clicked.connect(self._auto_select)
        
        self.done_btn = QPushButton("Concluir")
        self.done_btn.setEnabled(False)
        self.done_btn.clicked.connect(self.accept)
        
        btn_layout.addWidget(self.auto_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.done_btn)
        layout.addLayout(btn_layout)
        
    def _populate_grid(self):
        """Create thumbnail buttons from images distributed in PCA space."""
        n_cols = 6
        n_rows_grid = (len(self.images) + n_cols - 1) // n_cols
        
        x_min, x_max = self.pca_2d[:, 0].min(), self.pca_2d[:, 0].max()
        y_min, y_max = self.pca_2d[:, 1].min(), self.pca_2d[:, 1].max()
        
        # Grid-based distribution
        grid_to_idx = {}
        used = set()
        
        for row in range(n_rows_grid):
            for col in range(n_cols):
                cx_norm = (col + 0.5) / n_cols
                cy_norm = (row + 0.5) / n_rows_grid
                cx = x_min + cx_norm * (x_max - x_min)
                cy = y_min + cy_norm * (y_max - y_min)
                
                best_idx = None
                best_dist = float('inf')
                for idx in range(len(self.images)):
                    if idx in used:
                        continue
                    dist = (self.pca_2d[idx, 0] - cx)**2 + (self.pca_2d[idx, 1] - cy)**2
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = idx
                
                if best_idx is not None:
                    grid_to_idx[(row, col)] = best_idx
                    used.add(best_idx)
        
        # Add remaining
        for idx in range(len(self.images)):
            if idx not in used:
                for row in range(n_rows_grid):
                    for col in range(n_cols):
                        if (row, col) not in grid_to_idx:
                            grid_to_idx[(row, col)] = idx
                            used.add(idx)
                            break
                    if idx in used:
                        break
        
        # Create buttons
        self.thumb_buttons = {}
        for (row, col), idx in grid_to_idx.items():
            img = self.images[idx]
            thumb = cv2.resize(img, (120, 120), interpolation=cv2.INTER_AREA)
            thumb_rgb = thumb  # images already in RGB, no conversion needed
            
            h, w, ch = thumb_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(thumb_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            btn = QPushButton()
            btn.setFixedSize(130, 130)
            btn.setIcon(pixmap)
            btn.setIconSize(QSize(120, 120))
            btn.setProperty("image_idx", idx)
            btn.setProperty("image_name", self.image_names[idx])
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton {
                    border: 3px solid transparent;
                    padding: 2px;
                }
                QPushButton:checked {
                    border: 3px solid #e74c3c;
                    background-color: #ffeaea;
                }
                QPushButton:hover {
                    border: 3px solid #3498db;
                }
            """)
            btn.clicked.connect(lambda checked, b=btn: self._on_thumbnail_clicked(b))
            
            self.grid_layout.addWidget(btn, row, col)
            self.thumb_buttons[idx] = btn
            
    def _on_thumbnail_clicked(self, btn):
        if btn is None:
            return
            
        idx = btn.property("image_idx")
        
        if btn.isChecked():
            if idx not in self.selected_indices:
                self.selected_indices.append(idx)
        else:
            if idx in self.selected_indices:
                self.selected_indices.remove(idx)
        
        count = len(self.selected_indices)
        color = 'green' if count >= self.n_examples else 'red'
        self.instruction.setText(
            f"<b>Clique nas imagens para selecionar exemplos de '{self.class_name}'</b><br>"
            f"Selecionados: <span style='color: {color};'>{count}/{self.n_examples}</span><br>"
            f"{'✓ Pronto para concluir!' if count >= self.n_examples else 'Continue selecionando...'}"
        )
        
        self.done_btn.setEnabled(count >= self.n_examples)
            
    def _auto_select(self):
        """Auto-select examples using k-means on PCA space."""
        available = [i for i in range(len(self.images)) if i not in self.selected_indices]
        n_pick = min(self.n_examples - len(self.selected_indices), len(available))
        
        if n_pick > 0:
            kmeans = KMeans(n_clusters=n_pick, random_state=42 + self.class_id, n_init=10)
            clusters = kmeans.fit_predict(self.pca_2d[available])
            
            for c in range(n_pick):
                cluster_indices = [available[i] for i, cl in enumerate(clusters) if cl == c]
                if cluster_indices:
                    centroid = kmeans.cluster_centers_[c]
                    dists = np.linalg.norm(self.pca_2d[cluster_indices] - centroid, axis=1)
                    best = cluster_indices[np.argmin(dists)]
                    
                    if best not in self.selected_indices:
                        self.selected_indices.append(best)
                        if best in self.thumb_buttons:
                            self.thumb_buttons[best].setChecked(True)
        
        self._on_thumbnail_clicked(self.thumb_buttons.get(self.selected_indices[0]) 
                                   if self.selected_indices else None)


# =============================================================================
# FAST DATASET & DATALOADER 
# =============================================================================

class FastImageDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif img.shape[2] == 3 and isinstance(img, np.ndarray):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_tensor = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img_tensor, label


# =============================================================================
# FAST INCEPTIONV3 CLASSIFIER 
# =============================================================================

class FastInceptionV3Classifier:
    def __init__(self, n_classes, device=None, freeze_backbone=False):
        self.n_classes = n_classes
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.freeze_backbone = freeze_backbone

        self.model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)

        if self.freeze_backbone:
            print("  Freezing backbone (transfer learning)...")
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            print("  Fine-tuning: last 3 blocks unfrozen...")
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.Mixed_7c.parameters():
                param.requires_grad = True
            for param in self.model.Mixed_7b.parameters():
                param.requires_grad = True
            for param in self.model.Mixed_7a.parameters():
                param.requires_grad = True

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, n_classes)
        self.model = self.model.to(self.device)

        # SEM RESIZE - imagens já são 299×299 (crop feito antes)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.scaler = GradScaler() if torch.cuda.is_available() else None

    def train(self, X_train, y_train, epochs=30, batch_size=64, val_split=0.2,
              patience=5, num_workers=0, class_weights=None):
        
        print(f"\n Training InceptionV3...")
        print(f" Device: {self.device}")
        print(f" Samples: {len(X_train)} | Epochs: {epochs} | Batch: {batch_size}")
        print(f" Backbone frozen: {self.freeze_backbone}")
        print(f" Input size: 299×299 (pre-cropped)")

        n_val = int(len(X_train) * val_split)
        indices = np.random.permutation(len(X_train))
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        train_dataset = FastImageDataset(X_train[train_idx], y_train[train_idx], self.transform)
        val_dataset = FastImageDataset(X_train[val_idx], y_train[val_idx], self.transform)

        pin_memory = self.device.type == 'cuda'
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        )

        if class_weights is None:
            counts = Counter(y_train[train_idx])
            total = len(train_idx)
            weights = torch.tensor([
                total / (self.n_classes * counts.get(c, 1))
                for c in range(self.n_classes)
            ], dtype=torch.float32).to(self.device)
            print(f" Class weights: {weights.cpu().numpy()}")
        else:
            weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=weights)

        params = [{'params': self.model.fc.parameters(), 'lr': 0.001}]
        if not self.freeze_backbone:
            params.append({
                'params': list(self.model.Mixed_7a.parameters()) +
                          list(self.model.Mixed_7b.parameters()) +
                          list(self.model.Mixed_7c.parameters()),
                'lr': 0.0001
            })

        optimizer = torch.optim.Adam(params)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2
        )

        best_val_acc, epochs_no_improve = 0, 0

        for epoch in range(epochs):
            self.model.train()
            train_loss, train_correct, train_total = 0, 0, 0

            for batch_imgs, batch_labels in train_loader:
                batch_imgs = batch_imgs.to(self.device, non_blocking=pin_memory)
                batch_labels = batch_labels.to(self.device, non_blocking=pin_memory)

                optimizer.zero_grad()

                if self.scaler:
                    with autocast():
                        outputs = self.model(batch_imgs)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        loss = criterion(outputs, batch_labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(batch_imgs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item() * batch_imgs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == batch_labels).sum().item()
                train_total += batch_labels.size(0)

            self.model.eval()
            val_correct, val_total = 0, 0
            val_preds, val_true = [], []

            with torch.no_grad():
                for batch_imgs, batch_labels in val_loader:
                    batch_imgs = batch_imgs.to(self.device, non_blocking=pin_memory)
                    batch_labels = batch_labels.to(self.device, non_blocking=pin_memory)

                    outputs = self.model(batch_imgs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]

                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == batch_labels).sum().item()
                    val_total += batch_labels.size(0)
                    val_preds.extend(predicted.cpu().numpy())
                    val_true.extend(batch_labels.cpu().numpy())

            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            val_dist = Counter(val_preds)
            true_dist = Counter(val_true)

            print(f" Epoch {epoch+1}/{epochs} | "
                  f"Loss: {train_loss/train_total:.4f} | "
                  f"Train: {train_acc:.3f} | Val: {val_acc:.3f}")
            print(f"   Val PREDICTED: {dict(val_dist)}")
            print(f"   Val TRUE:      {dict(true_dist)}")

            scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), "best_model_fast.pth")
                print(f"  -> New best! Saved.")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\n Early stopping at epoch {epoch+1}")
                    break

        print(f"\n Best validation accuracy: {best_val_acc:.3f}")
        self.model.load_state_dict(torch.load("best_model_fast.pth", weights_only=True))

    def predict(self, images, batch_size=64, num_workers=0):
        self.model.eval()
        dummy_labels = np.zeros(len(images), dtype=np.int64)
        dataset = FastImageDataset(images, dummy_labels, self.transform)
        pin_memory = self.device.type == 'cuda'
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        )

        predictions, confidences = [], []
        with torch.no_grad():
            for batch_imgs, _ in loader:
                batch_imgs = batch_imgs.to(self.device, non_blocking=pin_memory)
                outputs = self.model(batch_imgs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
                predictions.extend(pred.cpu().numpy())
                confidences.extend(conf.cpu().numpy())

        return np.array(predictions), np.array(confidences)

    def extract_features(self, images, batch_size=64, num_workers=0):
        self.model.eval()
        original_fc = self.model.fc
        self.model.fc = nn.Identity()

        dummy_labels = np.zeros(len(images), dtype=np.int64)
        dataset = FastImageDataset(images, dummy_labels, self.transform)
        pin_memory = self.device.type == 'cuda'
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        )

        features = []
        with torch.no_grad():
            for batch_imgs, _ in loader:
                batch_imgs = batch_imgs.to(self.device, non_blocking=pin_memory)
                outputs = self.model(batch_imgs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                features.extend(outputs.cpu().numpy())

        self.model.fc = original_fc
        return np.array(features)


# =============================================================================
# HIERARCHICAL CLASSIFIER 
# =============================================================================

class HierarchicalClassifier:
    """
    Combina EntropyClassifier + CNNClassifier com pesos adaptativos.
    Quanto mais próximo da fronteira de entropia, mais confia no CNN.
    """
    def __init__(self, entropy_classifier, cnn_classifier,
                 entropy_weight=0.4, margin=0.3):
        self.entropy_clf = entropy_classifier
        self.cnn_clf = cnn_classifier
        self.entropy_weight = entropy_weight
        self.margin = margin
        self.thresholds = entropy_classifier.thresholds

    def predict(self, images, cnn_probs=None):
        entropies = self.entropy_clf.compute_entropies(images)
        entropy_probs = np.zeros((len(images), 3))
        
        for i, e in enumerate(entropies):
            if e < self.thresholds[0] - self.margin:
                entropy_probs[i, 0] = 1.0
            elif e < self.thresholds[0] + self.margin:
                w = (self.thresholds[0] + self.margin - e) / (2 * self.margin)
                entropy_probs[i, 0] = w
                entropy_probs[i, 1] = 1 - w
            elif e < self.thresholds[1] - self.margin:
                entropy_probs[i, 1] = 1.0
            elif e < self.thresholds[1] + self.margin:
                w = (self.thresholds[1] + self.margin - e) / (2 * self.margin)
                entropy_probs[i, 1] = w
                entropy_probs[i, 2] = 1 - w
            else:
                entropy_probs[i, 2] = 1.0
        
        if cnn_probs is None:
            return np.argmax(entropy_probs, axis=1), entropy_probs
        
        # Peso adaptativo: longe da fronteira = confia mais na entropia
        weights = np.zeros(len(images))
        for i, e in enumerate(entropies):
            dist_to_boundary = min(abs(e - self.thresholds[0]), abs(e - self.thresholds[1]))
            weights[i] = min(dist_to_boundary / self.margin, 1.0)
        
        final_probs = np.zeros_like(cnn_probs)
        for i in range(len(images)):
            w = weights[i]
            final_probs[i] = w * entropy_probs[i] + (1 - w) * cnn_probs[i]
        
        return np.argmax(final_probs, axis=1), final_probs


# =============================================================================
# MAIN SEAFLOOR CLASSIFICATION CLASS 
# =============================================================================

class SeafloorClassifier:
    """
    Classificador de fundo marinho com classes dinâmicas.

    Classes fixas (atalhos de teclado):
        S → Sedimento
        F → Coral_Fragmento  
        R → Coral_Vivo

    Classes customizadas (atalhos numéricos):
        1, 2, 3... 9, 0 → adicionadas pelo usuário via menu
    """

    # Classes fixas e seus atalhos
    FIXED_CLASSES = {
        "S": "Sedimento",
        "F": "Coral_Fragmento", 
        "R": "Coral_Vivo"
    }

    FIXED_COLORS = {
        "Sedimento": "#8B4513",
        "Coral_Fragmento": "#FF8C00",
        "Coral_Vivo": "#00CED1"
    }

    # Atalhos disponíveis para classes customizadas
    CUSTOM_SHORTCUTS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]

    def __init__(self, n_clusters=3, max_examples=None,
                 normalize_colors=True, reference_path=None,
                 n_examples_per_class=10, use_augmentation=True,
                 fast_mode=True, output_dir="seafloor_output",
                 crop_to_299=True,
                 crop_ratio=0.85,
                 use_hierarchical=False,
                 entropy_margin=0.3,
                 custom_classes=None,        # NOVO: lista de nomes
                 custom_colors=None):          # NOVO: dict nome→cor
        """
        Args:
            custom_classes: lista de nomes de classes adicionais (ex: ["Areia", "Rocha"])
            custom_colors: dict com cores para classes customizadas (ex: {"Areia": "#FFD700"})
        """

        # Inicializar classes fixas
        self.fixed_class_names = list(self.FIXED_CLASSES.values())
        self.fixed_colors = dict(self.FIXED_COLORS)

        # Inicializar classes customizadas
        self.custom_classes = {}  # {atalho: nome}  ex: {"1": "Areia", "2": "Rocha"}
        self.custom_colors = {}   # {nome: cor}     ex: {"Areia": "#FFD700"}

        if custom_classes:
            for i, name in enumerate(custom_classes):
                if i < len(self.CUSTOM_SHORTCUTS):
                    shortcut = self.CUSTOM_SHORTCUTS[i]
                    self.custom_classes[shortcut] = name
                    if custom_colors and name in custom_colors:
                        self.custom_colors[name] = custom_colors[name]
                    else:
                        self.custom_colors[name] = self._generate_color(name)

        # Lista completa de classes para o classificador
        self.class_names = self.fixed_class_names + list(self.custom_classes.values())
        self.class_colors = {**self.fixed_colors, **self.custom_colors}

        # Atualizar n_clusters
        self.n_clusters = len(self.class_names)

        # ... resto do __init__ existente (max_examples, normalize_colors, etc.) ...
        self.max_examples = max_examples
        self.normalize_colors = normalize_colors
        self.n_examples_per_class = n_examples_per_class
        self.use_augmentation = use_augmentation
        self.fast_mode = fast_mode
        self.output_dir = Path(output_dir)
        self.crop_to_299 = crop_to_299
        self.crop_ratio = crop_ratio
        self.cropper = CenterCropTo299(crop_ratio=crop_ratio) if crop_to_299 else None
        self.use_hierarchical = use_hierarchical
        self.entropy_margin = entropy_margin

        self.normalizer = None
        if self.normalize_colors:
            self.normalizer = ColorNormalizer(reference_path=reference_path)

        self.labeler = SemiAutoLabeler(n_examples_per_class=n_examples_per_class)
        self.trained_classifier = None

        # Atualizar entropy classifier com classes dinâmicas
        self.entropy_classifier = EntropyClassifier(
            thresholds=(6.471, 6.980),
            class_names=self.class_names
        )
        self.hierarchical_classifier = None

        # Criar diretórios
        self.output_dir.mkdir(exist_ok=True)
        for sub in ["clusters", "normalized", "supervised", "pca_plots", "cropped"]:
            (self.output_dir / sub).mkdir(exist_ok=True)
        self._refresh_cluster_dirs()

    # -------------------------------------------------------------------------
    # MÉTODOS NOVOS PARA GERENCIAMENTO DE CLASSES
    # -------------------------------------------------------------------------

    def _generate_color(self, name):
        """Gera cor automática baseada no hash do nome."""
        hue = hash(name) % 360 / 360.0
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

    def _refresh_cluster_dirs(self):
        """Atualiza diretórios de clusters para número atual de classes."""
        for i in range(self.n_clusters):
            (self.output_dir / "clusters" / f"cluster{i}").mkdir(exist_ok=True)
            (self.output_dir / "supervised" / f"class{i}").mkdir(exist_ok=True)

    def get_class_by_shortcut(self, shortcut):
        """
        Retorna o nome da classe para um atalho de teclado.

        Args:
            shortcut: str - "S", "F", "R", "1", "2", ... "9", "0"

        Returns:
            str: nome da classe ou None se atalho inválido
        """
        shortcut = shortcut.upper() if shortcut in ["s", "f", "r"] else shortcut

        if shortcut in self.FIXED_CLASSES:
            return self.FIXED_CLASSES[shortcut]

        if shortcut in self.custom_classes:
            return self.custom_classes[shortcut]

        return None

    def get_shortcut_for_class(self, class_name):
        """Retorna o atalho para uma classe (ex: "Sedimento" → "S")."""
        # Verificar fixas
        for shortcut, name in self.FIXED_CLASSES.items():
            if name == class_name:
                return shortcut
        # Verificar customizadas
        for shortcut, name in self.custom_classes.items():
            if name == class_name:
                return shortcut
        return None

    def add_custom_class(self, name, color=None):
        """
        Adiciona nova classe customizada.

        Args:
            name: nome da nova classe
            color: cor opcional (hex), senão gera automática

        Returns:
            tuple: (sucesso: bool, atalho: str ou None, mensagem: str)
        """
        # Verificar se nome já existe
        if name in self.class_names:
            return False, None, f"Classe '{name}' já existe!"

        # Verificar se há atalho disponível
        available = [s for s in self.CUSTOM_SHORTCUTS if s not in self.custom_classes]
        if not available:
            return False, None, "Limite de 10 classes customizadas atingido!"

        shortcut = available[0]

        # Adicionar
        self.custom_classes[shortcut] = name
        self.custom_colors[name] = color if color else self._generate_color(name)

        # Atualizar listas consolidadas
        self.class_names = self.fixed_class_names + list(self.custom_classes.values())
        self.class_colors[name] = self.custom_colors[name]
        self.n_clusters = len(self.class_names)

        # Atualizar entropy classifier
        self.entropy_classifier.class_names = self.class_names

        # Criar diretórios
        self._refresh_cluster_dirs()

        return True, shortcut, f"Classe '{name}' adicionada com atalho '{shortcut}'"

    def remove_custom_class(self, name_or_shortcut):
        """
        Remove uma classe customizada.

        Args:
            name_or_shortcut: nome da classe ou atalho (ex: "Areia" ou "1")

        Returns:
            tuple: (sucesso: bool, mensagem: str)
        """
        # Resolver nome se recebeu atalho
        name = name_or_shortcut
        if name_or_shortcut in self.custom_classes:
            name = self.custom_classes[name_or_shortcut]

        if name not in self.custom_classes.values():
            return False, f"Classe '{name}' não encontrada ou é fixa (não pode ser removida)"

        # Remover
        shortcut_to_remove = None
        for s, n in self.custom_classes.items():
            if n == name:
                shortcut_to_remove = s
                break

        if shortcut_to_remove:
            del self.custom_classes[shortcut_to_remove]

        self.custom_colors.pop(name, None)

        # Atualizar listas consolidadas
        self.class_names = self.fixed_class_names + list(self.custom_classes.values())
        self.n_clusters = len(self.class_names)

        # Atualizar entropy classifier
        self.entropy_classifier.class_names = self.class_names

        # Reorganizar atalhos (compactar)
        self._reorganize_shortcuts()

        return True, f"Classe '{name}' removida"

    def _reorganize_shortcuts(self):
        """Reorganiza atalhos para ficarem sequenciais (1,2,3...)."""
        classes = list(self.custom_classes.values())
        self.custom_classes = {}
        for i, name in enumerate(classes):
            if i < len(self.CUSTOM_SHORTCUTS):
                self.custom_classes[self.CUSTOM_SHORTCUTS[i]] = name

        self.class_names = self.fixed_class_names + list(self.custom_classes.values())
        self.n_clusters = len(self.class_names)

    def list_all_classes(self):
        """Retorna lista de todas as classes com seus atalhos."""
        result = []
        for shortcut, name in self.FIXED_CLASSES.items():
            result.append({
                "shortcut": shortcut,
                "name": name,
                "color": self.fixed_colors[name],
                "type": "fixed"
            })
        for shortcut, name in self.custom_classes.items():
            result.append({
                "shortcut": shortcut,
                "name": name,
                "color": self.custom_colors.get(name, "#808080"),
                "type": "custom"
            })
        return result

    def save_config(self, path=None):
        """Salva configuração atual (classes e cores)."""
        config = {
            "fixed_classes": self.FIXED_CLASSES,
            "fixed_colors": self.fixed_colors,
            "custom_classes": self.custom_classes,
            "custom_colors": self.custom_colors,
            "class_names": self.class_names,
            "class_colors": self.class_colors,
            "n_clusters": self.n_clusters,
            "crop_to_299": self.crop_to_299,
            "crop_ratio": self.crop_ratio,
            "normalize_colors": self.normalize_colors,
            "use_hierarchical": self.use_hierarchical
        }
        path = path or (self.output_dir / "classifier_config.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return path

    def load_config(self, path):
        """Carrega configuração salva."""
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        self.custom_classes = config.get("custom_classes", {})
        self.custom_colors = config.get("custom_colors", {})
        self.class_names = config.get("class_names", self.fixed_class_names)
        self.class_colors = {**self.fixed_colors, **self.custom_colors}
        self.n_clusters = config.get("n_clusters", len(self.class_names))

        # Atualizar entropy classifier
        self.entropy_classifier.class_names = self.class_names

        self._refresh_cluster_dirs()
        return config


    # -------------------------------------------------------------------------
    # MÉTODOS NOVOS: Carregar imagens de pasta
    # -------------------------------------------------------------------------

    def load_images_from_folder(self, folder_path):
        """Carrega imagens de uma pasta para classificação."""
        folder = Path(folder_path)
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]:
            image_paths.extend(folder.glob(ext))
            image_paths.extend(folder.glob(ext.upper()))

        image_paths = sorted(image_paths)
        if not image_paths:
            raise ValueError(f"Nenhuma imagem encontrada em: {folder_path}")

        images_raw = []
        images_rgb = []
        image_names = []

        for p in image_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            images_raw.append(img)
            images_rgb.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            image_names.append(p.name)

        if not images_raw:
            raise ValueError(f"Nenhuma imagem válida carregada de: {folder_path}")

        print(f"  Carregadas {len(images_raw)} imagens de {folder_path}")
        return images_raw, images_rgb, image_names

    # -------------------------------------------------------------------------
    # MÉTODOS NOVOS: Persistência do modelo treinado
    # -------------------------------------------------------------------------

    def save_model(self, path=None):
        """Salva o modelo treinado + configuração para uso posterior."""
        if self.trained_classifier is None:
            raise ValueError("Nenhum modelo treinado para salvar!")

        path = Path(path) if path else (self.output_dir / "seafloor_model.pt")
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.trained_classifier.model.state_dict(),
            "n_classes": self.trained_classifier.n_classes,
            "class_names": self.class_names,
            "class_colors": self.class_colors,
            "custom_classes": self.custom_classes,
            "custom_colors": self.custom_colors,
            "fixed_class_names": self.fixed_class_names,
            "fixed_colors": self.fixed_colors,
            "n_clusters": self.n_clusters,
            "crop_to_299": self.crop_to_299,
            "crop_ratio": self.crop_ratio,
            "normalize_colors": self.normalize_colors,
            "use_hierarchical": self.use_hierarchical,
            "entropy_margin": self.entropy_margin,
            "ref_mean": self.normalizer.ref_mean.tolist() if (self.normalizer and self.normalizer.ref_mean is not None) else None,
            "ref_std": self.normalizer.ref_std.tolist() if (self.normalizer and self.normalizer.ref_std is not None) else None,
            "reference_path": str(self.normalizer.reference_path) if self.normalizer else None,
        }

        torch.save(checkpoint, str(path))
        print(f"  Modelo salvo em: {path}")
        return str(path)

    def load_model(self, path):
        """Carrega um modelo treinado previamente salvo."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {path}")

        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)

        # Restaurar configuração
        self.class_names = checkpoint.get("class_names", self.class_names)
        self.class_colors = checkpoint.get("class_colors", self.class_colors)
        self.custom_classes = checkpoint.get("custom_classes", {})
        self.custom_colors = checkpoint.get("custom_colors", {})
        self.fixed_class_names = checkpoint.get("fixed_class_names", self.fixed_class_names)
        self.fixed_colors = checkpoint.get("fixed_colors", self.fixed_colors)
        self.n_clusters = checkpoint.get("n_clusters", self.n_clusters)
        self.crop_to_299 = checkpoint.get("crop_to_299", self.crop_to_299)
        self.crop_ratio = checkpoint.get("crop_ratio", self.crop_ratio)
        self.normalize_colors = checkpoint.get("normalize_colors", self.normalize_colors)
        self.use_hierarchical = checkpoint.get("use_hierarchical", self.use_hierarchical)
        self.entropy_margin = checkpoint.get("entropy_margin", self.entropy_margin)

        # Restaurar normalizador
        ref_mean = checkpoint.get("ref_mean")
        ref_std = checkpoint.get("ref_std")
        ref_path = checkpoint.get("reference_path")

        if ref_mean is not None and ref_std is not None:
            if self.normalizer is None:
                self.normalizer = ColorNormalizer(reference_path=ref_path)
            self.normalizer.ref_mean = np.array(ref_mean, dtype=np.float32)
            self.normalizer.ref_std = np.array(ref_std, dtype=np.float32)

        # Recriar classificador
        n_classes = checkpoint.get("n_classes", len(self.class_names))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.trained_classifier = FastInceptionV3Classifier(
            n_classes=n_classes,
            device=device,
            freeze_backbone=False
        )
        self.trained_classifier.model.load_state_dict(checkpoint["model_state_dict"])
        self.trained_classifier.model.eval()

        self.entropy_classifier.class_names = self.class_names

        print(f"  Modelo carregado de: {path}")
        print(f"  Classes: {self.class_names}")
        return True

    def has_trained_model(self):
        """Verifica se existe um modelo treinado carregado."""
        return self.trained_classifier is not None

    # -------------------------------------------------------------------------
    # MÉTODO NOVO: Treinar a partir de dados coletados
    # -------------------------------------------------------------------------

    def train_from_collected_data(self, data_dir, min_samples_per_class=5,
                                   parent_widget=None, epochs_phase1=8,
                                   epochs_phase2=15):
        """
        Treina o classificador a partir de frames coletados durante anotação.
        """
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise ValueError(f"Diretório não encontrado: {data_dir}")

        # Descobrir classes a partir das subpastas
        class_dirs = [d for d in data_dir.iterdir()
                      if d.is_dir() and any(d.glob("*.jpg"))]

        if len(class_dirs) < 2:
            raise ValueError(
                f"São necessárias pelo menos 2 classes com imagens. "
                f"Encontradas: {len(class_dirs)}"
            )

        # Atualizar classes do classificador
        class_names_from_dirs = sorted([d.name for d in class_dirs])

        # Separar fixas de customizadas
        new_custom = [n for n in class_names_from_dirs if n not in self.fixed_class_names]

        # Resetar classes customizadas
        self.custom_classes = {}
        self.custom_colors = {}
        for i, name in enumerate(new_custom):
            if i < len(self.CUSTOM_SHORTCUTS):
                shortcut = self.CUSTOM_SHORTCUTS[i]
                self.custom_classes[shortcut] = name
                self.custom_colors[name] = self._generate_color(name)

        self.class_names = self.fixed_class_names + list(self.custom_classes.values())
        self.class_colors = {**self.fixed_colors, **self.custom_colors}
        self.n_clusters = len(self.class_names)
        self.entropy_classifier.class_names = self.class_names
        self._refresh_cluster_dirs()

        # Carregar todas as imagens
        images_raw = []
        images_rgb = []
        image_names = []
        labels = []

        print(f"\n--- Carregando dados de treinamento ---")
        for class_id, class_name in enumerate(self.class_names):
            class_dir = data_dir / class_name
            if not class_dir.exists():
                continue

            class_images = []
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                class_images.extend(class_dir.glob(ext))

            if len(class_images) < min_samples_per_class:
                print(f"  AVISO: Classe '{class_name}' tem apenas {len(class_images)} imagens")

            for img_path in sorted(class_images):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                images_raw.append(img)
                images_rgb.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                image_names.append(img_path.name)
                labels.append(class_id)

        if len(images_raw) < min_samples_per_class * 2:
            raise ValueError(f"Dados insuficientes: apenas {len(images_raw)} imagens totais")

        print(f"  Total: {len(images_raw)} imagens, classes: {self.class_names}")

        labels = np.array(labels)

        # Executar workflow completo
        supervised_preds, unsupervised_preds, confidences = self.classify_images(
            images_rgb=np.array(images_rgb),
            images_raw=np.array(images_raw),
            image_names=image_names,
            folder_path=str(data_dir),
            parent_widget=parent_widget
        )

        # Salvar modelo
        model_path = self.save_model(data_dir / "seafloor_model.pt")
        config_path = self.save_config(data_dir / "classifier_config.json")

        return {
            "model_path": model_path,
            "config_path": str(config_path),
            "class_names": self.class_names,
            "n_samples": len(images_raw),
        }

    # -------------------------------------------------------------------------
    # MÉTODO MODIFICADO: predict_single_frame
    # -------------------------------------------------------------------------

    def predict_single_frame(self, frame_bgr):
        if self.trained_classifier is None:
            raise ValueError("Classifier not trained yet!")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self.crop_to_299 and self.cropper is not None:
            cropped, _ = self.cropper.crop(frame_bgr)
            frame_bgr = cropped
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self.normalize_colors and self.normalizer and self.normalizer.ref_mean is not None:
            frame_norm_bgr = self.normalizer.normalize(frame_bgr)
            frame_rgb = cv2.cvtColor(frame_norm_bgr, cv2.COLOR_BGR2RGB)

        preds, confs = self.trained_classifier.predict([frame_rgb], batch_size=1)

        class_id = int(preds[0])
        confidence = float(confs[0])

        # Garantir que class_id está no range válido
        if class_id >= len(self.class_names):
            class_id = len(self.class_names) - 1

        class_name = self.class_names[class_id]

        return {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "color": self.class_colors.get(class_name, "#FFFFFF"),
            "shortcut": self.get_shortcut_for_class(class_name)
        }


    def classify_images(self, images_rgb, images_raw, image_names, folder_path=None,
                       parent_widget=None):
        """
        Run full classification workflow WITH INTERACTIVE LABELING via Qt.
        ATUALIZADO: Normalização DEPOIS do labeling, crop para 299×299.
        """
        # ===== Step 1: Domain features + PCA (ORIGINAL colors, full resolution) =====
        features = self.labeler.extract_domain_features(images_rgb)
        features_scaled = StandardScaler().fit_transform(features)
        pca_2d = PCA(n_components=2).fit_transform(features_scaled)

        # ===== Step 2: INTERACTIVE labeling via Qt Dialog =====
        self.labeler.interactive_labeling_qt(
            images_rgb, image_names, pca_2d,
            self.n_clusters, self.class_names,
            parent_widget=parent_widget
        )
        self.labeler.expand_labels_nearest_neighbors(images_rgb, n_neighbors=50)

        # ===== Step 2.5: CROP to 299×299 (NOVO - remove dark borders) =====
        if self.crop_to_299 and self.cropper is not None:
            print("\n--- Cropping to 299×299 (removing dark borders) ---")
            cropped_raw, crop_infos = self.cropper.crop_batch(images_raw, image_names)
            cropped_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in cropped_raw]
            
            methods = Counter([info['method'] for info in crop_infos])
            print(f"  Methods: {dict(methods)}")
            
            # Salvar cropped para debug
            for img, name in zip(cropped_raw, image_names):
                out_path = self.output_dir / "cropped" / f"crop_{name}"
                cv2.imwrite(str(out_path), img)
            
            images_cropped_raw = cropped_raw
            images_cropped_rgb = cropped_rgb
        else:
            images_cropped_raw = images_raw
            images_cropped_rgb = images_rgb

        # ===== Step 3: Color normalization (AGORA DEPOIS do labeling!) =====
        if self.normalize_colors and self.normalizer and folder_path:
            ref_img = self.normalizer.select_reference(image_names, folder_path)
            self.normalizer.compute_reference_stats(ref_img)
            normalized_bgr = self.normalizer.normalize_dataset(
                images_cropped_raw, image_names,
                output_dir=str(self.output_dir / "normalized")
            )
            images_normalized_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                                      for img in normalized_bgr]
        else:
            images_normalized_rgb = images_cropped_rgb

        # ===== Step 4: Supervised training =====
        if self.use_augmentation:
            X_train, y_train, _ = self.labeler.get_training_data_balanced(
                images_normalized_rgb, image_names, augment_factor=3
            )
        else:
            X_train, y_train, _ = self.labeler.get_training_data_balanced(
                images_normalized_rgb, image_names, augment_factor=0
            )

        n_classes = len(np.unique(y_train))

        if self.fast_mode:
            # Phase 1: Transfer learning (frozen backbone)
            print("\n  Phase 1: Transfer learning (frozen backbone)...")
            classifier = FastInceptionV3Classifier(
                n_classes=n_classes, freeze_backbone=True
            )
            classifier.train(X_train, y_train, epochs=8, batch_size=64, patience=2)

            # Phase 2: Fine-tuning (partial unfreeze)
            print("\n  Phase 2: Fine-tuning (partial unfreeze)...")
            classifier = FastInceptionV3Classifier(
                n_classes=n_classes, freeze_backbone=False
            )
            classifier.model.load_state_dict(
                torch.load("best_model_fast.pth", weights_only=True)
            )
            classifier.train(X_train, y_train, epochs=15, batch_size=64, patience=4)
            self.trained_classifier = classifier
        else:
            classifier = FastInceptionV3Classifier(
                n_classes=n_classes, freeze_backbone=False
            )
            classifier.train(X_train, y_train, epochs=20, batch_size=64, patience=5)
            self.trained_classifier = classifier

        supervised_preds, confidences = classifier.predict(
            images_normalized_rgb, batch_size=64
        )
        
        # NOVO: Classificador hierárquico (opcional)
        if self.use_hierarchical:
            print("\n--- Hierarchical classification (entropy + CNN) ---")
            cnn_probs = torch.nn.functional.softmax(
                torch.tensor(self.trained_classifier.model(
                    torch.stack([self.trained_classifier.transform(img) for img in images_normalized_rgb])
                )), dim=1
            ).numpy() if False else None  # Simplificado - na prática extrair probs do predict
            
            self.hierarchical_classifier = HierarchicalClassifier(
                self.entropy_classifier, self.trained_classifier,
                entropy_weight=0.4, margin=self.entropy_margin
            )
            # Para uso futuro em predict_single_frame

        # ===== Step 5: Unsupervised clustering =====
        unsupervised_preds = self._run_unsupervised(images_normalized_rgb)

        # ===== Step 6: Comparison =====
        self._compare_methods(supervised_preds, unsupervised_preds, confidences)

        # ===== Step 7: Save ORIGINAL images to result folders (NOVO) =====
        print("\n--- Saving ORIGINAL images to result folders ---")
        for i, name in enumerate(image_names):
            # Supervised: imagem ORIGINAL (não cropped/normalizada)
            dst = self.output_dir / "supervised" / f"class{supervised_preds[i]}" / name
            cv2.imwrite(str(dst), images_raw[i])  # images_raw = original!
            
        for i, name in enumerate(image_names):
            # Unsupervised: imagem ORIGINAL
            dst = self.output_dir / "clusters" / f"cluster{unsupervised_preds[i]}" / name
            cv2.imwrite(str(dst), images_raw[i])

        # ===== Step 8: Visualizations =====
        self._visualize_pca_supervised(images_normalized_rgb, supervised_preds, confidences)
        self._visualize_pca_unsupervised(images_normalized_rgb, unsupervised_preds)

        return supervised_preds, unsupervised_preds, confidences

    def _run_unsupervised(self, images_normalized_rgb):
        """Unsupervised clustering com InceptionV3 features + entropy features."""
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        model.fc = nn.Identity()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Imagens já são 299×299, sem resize necessário
        cnn_features = []
        with torch.no_grad():
            for i in range(0, len(images_normalized_rgb), 32):
                batch = images_normalized_rgb[i:i + 32]
                tensors = torch.stack([preprocess(img) for img in batch]).to(device)
                cnn_features.extend(model(tensors).cpu().numpy())

        cnn_features = np.array(cnn_features)
        entropy_features = self.labeler.extract_domain_features(images_normalized_rgb)
        
        scaler_e = StandardScaler()
        entropy_scaled = scaler_e.fit_transform(entropy_features)
        scaler_c = StandardScaler()
        cnn_scaled = scaler_c.fit_transform(cnn_features)

        combined = np.hstack([cnn_scaled, entropy_scaled])

        n_components = min(512, combined.shape[0], combined.shape[1])
        images_new = PCA(n_components=n_components).fit_transform(
            StandardScaler().fit_transform(combined)
        )

        predictions = KMeans(n_clusters=self.n_clusters, random_state=728, n_init=10).fit_predict(images_new)
        return predictions

    def _compare_methods(self, supervised, unsupervised, confidences):
        if len(set(supervised)) == len(set(unsupervised)):
            cm = confusion_matrix(supervised, unsupervised)
            row_ind, col_ind = linear_sum_assignment(-cm)
            mapped = np.zeros_like(unsupervised)
            for r, c in zip(row_ind, col_ind):
                mapped[unsupervised == c] = r
            kappa = cohen_kappa_score(supervised, mapped)
            print(f"\nCohen's Kappa: {kappa:.3f}")

        print(f"Confidence > 0.6: {np.mean(confidences > 0.6):.1%}")
        print("\nSupervised distribution:", Counter(supervised))
        print("Unsupervised distribution:", Counter(unsupervised))

    def _visualize_pca_supervised(self, images, predictions, confidences):
        """Visualização PCA supervisionada com features do classificador treinado."""
        if self.trained_classifier is None:
            return

        trained_features = self.trained_classifier.extract_features(images, batch_size=64)
        scaled = StandardScaler().fit_transform(trained_features)
        pca = PCA(n_components=2)
        pca_2d = pca.fit_transform(scaled)
        var = pca.explained_variance_ratio_

        colors_list = ['#2ecc71', '#9b59b6', '#e74c3c']

        fig, ax = plt.subplots(figsize=(16, 12))

        for idx in range(len(images)):
            img = images[idx].astype(np.uint8)
            thumb = cv2.resize(img, (50, 50))
            imagebox = OffsetImage(thumb, zoom=0.8)
            ab = AnnotationBbox(imagebox, (pca_2d[idx, 0], pca_2d[idx, 1]),
                            frameon=True, pad=0.05, boxcoords="data", zorder=1,
                            bboxprops=dict(
                                edgecolor=colors_list[predictions[idx] % len(colors_list)],
                                linewidth=2.5, facecolor='white', alpha=0.9
                            ))
            ax.add_artist(ab)

        for i in range(self.n_clusters):
            mask = predictions == i
            n_points = np.sum(mask)
            if n_points > 0:
                cls_name = self.class_names[i] if i < len(self.class_names) else f"Class_{i}"
                ax.scatter(pca_2d[mask, 0], pca_2d[mask, 1],
                        c=colors_list[i % len(colors_list)],
                        s=80, alpha=0.6, edgecolors='black', linewidth=0.8,
                        zorder=10, label=f'{cls_name} ({n_points})')

        info_text = ""
        for i in range(self.n_clusters):
            mask = predictions == i
            if np.sum(mask) > 0:
                cls_name = self.class_names[i] if i < len(self.class_names) else f"Class_{i}"
                avg_conf = np.mean(confidences[mask])
                info_text += f"{cls_name}: avg conf={avg_conf:.2f}\n"

        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel('PCA 1', fontsize=12)
        ax.set_ylabel('PCA 2', fontsize=12)
        title = 'Supervised Classification - PCA (TRAINED features)'
        if self.normalize_colors:
            title += ' (Color Normalized)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        out = self.output_dir / "pca_plots" / "pca_supervised_trained.png"
        plt.savefig(out, dpi=200, bbox_inches='tight')
        plt.close()

    def _visualize_pca_unsupervised(self, images, predictions):
        """NOVO: Visualização PCA não-supervisionada."""
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        model.fc = nn.Identity()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        cnn_features = []
        with torch.no_grad():
            for i in range(0, len(images), 32):
                batch = images[i:i + 32]
                tensors = torch.stack([preprocess(img) for img in batch]).to(device)
                cnn_features.extend(model(tensors).cpu().numpy())

        cnn_features = np.array(cnn_features)
        entropy_features = self.labeler.extract_domain_features(images)
        
        combined = np.hstack([
            StandardScaler().fit_transform(cnn_features),
            StandardScaler().fit_transform(entropy_features)
        ])
        
        scaled = StandardScaler().fit_transform(combined)
        pca = PCA(n_components=2)
        pca_2d = pca.fit_transform(scaled)
        var = pca.explained_variance_ratio_

        colors_list = ['#2ecc71', '#9b59b6', '#e74c3c']

        fig, ax = plt.subplots(figsize=(14, 10))

        for idx in range(len(images)):
            img = images[idx].astype(np.uint8)
            thumb = cv2.resize(img, (50, 50))
            imagebox = OffsetImage(thumb, zoom=0.8)
            ab = AnnotationBbox(imagebox, (pca_2d[idx, 0], pca_2d[idx, 1]),
                            frameon=True, pad=0.05, boxcoords="data", zorder=1,
                            bboxprops=dict(
                                edgecolor=colors_list[predictions[idx] % len(colors_list)],
                                linewidth=2, facecolor='white', alpha=0.9
                            ))
            ax.add_artist(ab)

        for i in range(self.n_clusters):
            mask = predictions == i
            n_points = np.sum(mask)
            if n_points > 0:
                ax.scatter(pca_2d[mask, 0], pca_2d[mask, 1],
                        c=colors_list[i % len(colors_list)],
                        s=60, alpha=0.9, edgecolors='black', linewidth=0.8,
                        zorder=10, label=f'Cluster {i+1} ({n_points})')

        ax.set_xlabel('PCA 1', fontsize=12)
        ax.set_ylabel('PCA 2', fontsize=12)
        title = 'Unsupervised Classification - PCA'
        if self.normalize_colors:
            title += ' (Color Normalized)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        out = self.output_dir / "pca_plots" / "pca_unsupervised.png"
        plt.savefig(out, dpi=200, bbox_inches='tight')
        plt.close()

# =============================================================================
# QTHREAD - CLASSIFICAÇÃO EM TEMPO REAL
# =============================================================================

class SeafloorClassificationThread(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, object, str)
    frame_classified = pyqtSignal(dict, int)
    request_interactive_labeling = pyqtSignal(object, object, object, int, list)
    labeling_result_ready = pyqtSignal(object)

    def __init__(self, classifier: SeafloorClassifier, parent=None):
        super().__init__(parent)
        self.classifier = classifier
        self.frames = []
        self.mode = "batch"
        self.running = True
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self._interactive_result = None
        self.labeling_result_ready.connect(self._on_labeling_result)

    def _on_labeling_result(self, labels):
        self._interactive_result = labels

    def add_frame(self, frame_bgr, frame_num):
        with QMutexLocker(self.mutex):
            self.frames.append((frame_bgr.copy(), frame_num))
            self.condition.wakeOne()

    def set_batch_mode(self, folder_path):
        self.mode = "batch"
        self.batch_folder = folder_path

    def set_realtime_mode(self):
        self.mode = "realtime"

    def stop(self):
        with QMutexLocker(self.mutex):
            self.running = False
            self.condition.wakeOne()

    def run(self):
        if self.mode == "batch":
            self._run_batch()
        elif self.mode == "realtime":
            self._run_realtime()

    def _run_batch(self):
        try:
            self.status.emit("Carregando imagens...")
            images_raw, images_rgb, image_names = self.classifier.load_images_from_folder(
                self.batch_folder
            )

            self.status.emit("Computando features + PCA...")
            features = self.classifier.labeler.extract_domain_features(images_rgb)
            features_scaled = StandardScaler().fit_transform(features)
            pca_2d = PCA(n_components=2).fit_transform(features_scaled)

            self.status.emit("Aguardando seleção do usuário...")
            self._interactive_result = None

            self.request_interactive_labeling.emit(
                images_rgb, image_names, pca_2d, 
                self.classifier.n_clusters, self.classifier.class_names
            )

            import time
            for _ in range(300):
                if self._interactive_result is not None:
                    break
                time.sleep(0.1)

            if self._interactive_result is None:
                self.finished_signal.emit(False, None, "Timeout aguardando seleção")
                return

            self.classifier.labeler.labels = self._interactive_result
            self.classifier.labeler.expand_labels_nearest_neighbors(images_rgb, n_neighbors=50)

            self.status.emit("Treinando classificador...")

            results = {
                "labels": self.classifier.labeler.labels,
                "class_names": self.classifier.class_names
            }
            self.finished_signal.emit(True, results, "Workflow completo!")

        except Exception as e:
            import traceback
            self.finished_signal.emit(False, None, f"Erro: {str(e)}\n{traceback.format_exc()}")

    def _run_realtime(self):
        """Classifica frames da fila usando modelo já treinado."""
        if not self.classifier.has_trained_model():
            self.status.emit("ERRO: Nenhum modelo treinado carregado!")
            self.finished_signal.emit(False, None, "Modelo não carregado")
            return

        self.status.emit("Classificação em tempo real iniciada")
        self._paused = False  # Flag for pause/resume with video
        processed_count = 0

        while self.running:
            # Check if paused (video is paused)
            if getattr(self, '_paused', False):
                import time
                time.sleep(0.1)
                continue

            with QMutexLocker(self.mutex):
                if not self.frames:
                    self.condition.wait(self.mutex, 100)
                    continue
                frame_bgr, frame_num = self.frames.pop(0)

            try:
                result = self.classifier.predict_single_frame(frame_bgr)
                processed_count += 1
                if processed_count <= 3:
                    self.status.emit(f"DEBUG: Frame {frame_num} classificado como {result['class_name']} ({result['confidence']:.2f})")
                self.frame_classified.emit(result, frame_num)
            except Exception as e:
                import traceback
                self.status.emit(f"Erro na classificação: {str(e)}\n{traceback.format_exc()}")

        self.status.emit(f"Classificação finalizada. {processed_count} frames processados.")
        self.finished_signal.emit(True, None, "Classificação em tempo real finalizada")
class SeafloorClassificationDialog(QDialog):
    def __init__(self, parent=None, language="pt"):
        super().__init__(parent)
        self.language = language
        self.setWindowTitle("Classificação de Fundo Marinho (AI-SCW)")
        self.resize(500, 450)  # Aumentado para novas opções
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        
        info = QLabel(
            "Classificação de fundo marinho usando AI-SCW<br>"
            "<b>Classes:</b> Sedimento, Coral Fragmento, Coral Vivo<br><br>"
            "O processo é <b>semi-automático</b>: você selecionará exemplos "
            "de cada classe visualmente antes do treinamento."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        folder_layout = QHBoxLayout()
        self.folder_label = QLabel("(nenhuma pasta selecionada)")
        self.folder_label.setStyleSheet("color: gray; font-style: italic;")
        browse_btn = QPushButton("Selecionar Pasta...")
        browse_btn.clicked.connect(self._select_folder)
        folder_layout.addWidget(self.folder_label, 1)
        folder_layout.addWidget(browse_btn)
        layout.addLayout(folder_layout)

        settings_group = QGroupBox("Configurações")
        settings_layout = QFormLayout(settings_group)

        self.examples_spin = QSpinBox()
        self.examples_spin.setRange(5, 50)
        self.examples_spin.setValue(10)
        settings_layout.addRow("Exemplos/classe:", self.examples_spin)

        # NOVO: Crop ratio
        self.crop_spin = QDoubleSpinBox()
        self.crop_spin.setRange(0.3, 1.0)
        self.crop_spin.setSingleStep(0.05)
        self.crop_spin.setValue(0.85)
        settings_layout.addRow("Crop ratio:", self.crop_spin)

        self.normalize_check = QPushButton("Normalização de cor: ATIVADA")
        self.normalize_check.setCheckable(True)
        self.normalize_check.setChecked(True)
        self.normalize_check.clicked.connect(self._toggle_normalize)
        settings_layout.addRow("Normalização:", self.normalize_check)
        
        # NOVO: Hierarchical classifier toggle
        self.hierarchical_check = QPushButton("Classificador hierárquico: DESATIVADO")
        self.hierarchical_check.setCheckable(True)
        self.hierarchical_check.setChecked(False)
        self.hierarchical_check.clicked.connect(self._toggle_hierarchical)
        settings_layout.addRow("Hierárquico:", self.hierarchical_check)

        layout.addWidget(settings_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_text = QLabel("")
        self.status_text.setVisible(False)
        layout.addWidget(self.status_text)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Iniciar Classificação")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start_classification)
        cancel_btn = QPushButton("Cancelar")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self.normalize_enabled = True
        self.hierarchical_enabled = False
        self.folder_path = None

    def _toggle_normalize(self):
        self.normalize_enabled = not self.normalize_enabled
        text = "ATIVADA" if self.normalize_enabled else "DESATIVADA"
        self.normalize_check.setText(f"Normalização de cor: {text}")

    def _toggle_hierarchical(self):
        self.hierarchical_enabled = not self.hierarchical_enabled
        text = "ATIVADO" if self.hierarchical_enabled else "DESATIVADO"
        self.hierarchical_check.setText(f"Classificador hierárquico: {text}")

    def _select_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Selecionar pasta de imagens")
        if path:
            self.folder_path = path
            self.folder_label.setText(Path(path).name)
            self.folder_label.setStyleSheet("color: black; font-style: normal;")
            self.start_btn.setEnabled(True)

    def _start_classification(self):
        if not self.folder_path:
            return

        self.status_text.setVisible(True)
        self.status_text.setText("Carregando imagens...")
        QApplication.processEvents()

        classifier = SeafloorClassifier(
            n_clusters=3,
            normalize_colors=self.normalize_enabled,
            n_examples_per_class=self.examples_spin.value(),
            use_augmentation=True,
            fast_mode=True,
            output_dir=os.path.join(self.folder_path, "seafloor_output"),
            crop_to_299=True,
            crop_ratio=self.crop_spin.value(),
            use_hierarchical=self.hierarchical_enabled
        )

        try:
            images_raw, images_rgb, image_names = classifier.load_images_from_folder(self.folder_path)
            
            self.status_text.setText("Computando features + PCA...")
            QApplication.processEvents()
            
            features = classifier.labeler.extract_domain_features(images_rgb)
            features_scaled = StandardScaler().fit_transform(features)
            pca_2d = PCA(n_components=2).fit_transform(features_scaled)

            # INTERACTIVE LABELING
            self.status_text.setText("Aguardando seleção do usuário...")
            QApplication.processEvents()

            labels = classifier.labeler.interactive_labeling_qt(
                images_rgb, image_names, pca_2d,
                classifier.n_clusters, classifier.class_names,
                parent_widget=self
            )

            if not labels or len(labels) == 0:
                QMessageBox.warning(self, "Aviso", 
                    "Nenhum exemplo selecionado. Usando auto-seleção.")
                # Fallback seria implementado aqui
            else:
                classifier.labeler.labels = labels

            # Expand labels
            self.status_text.setText("Expandindo labels (k-NN)...")
            QApplication.processEvents()
            classifier.labeler.expand_labels_nearest_neighbors(images_rgb, n_neighbors=50)

            # Full classification
            self.status_text.setText("Executando classificação completa...")
            QApplication.processEvents()
            
            supervised_preds, unsupervised_preds, confidences = classifier.classify_images(
                images_rgb, images_raw, image_names,
                folder_path=self.folder_path,
                parent_widget=self
            )

            # Show results
            dist = Counter(supervised_preds)
            msg = "Classificação concluída!\n\nDistribuição:\n"
            for cls_id, count in dist.items():
                name = classifier.class_names[cls_id]
                msg += f"  {name}: {count}\n"
            
            msg += f"\nConfiança média: {np.mean(confidences):.2f}"
            
            QMessageBox.information(self, "Concluído", msg)

        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Erro", f"{str(e)}\n\n{traceback.format_exc()}")

    def reject(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1000)
        super().reject()

class SeafloorClassManager(QDialog):
    """Diálogo para gerenciar categorias do classificador de fundo."""

    classes_changed = pyqtSignal()  # Sinal emitido quando classes mudam

    def __init__(self, classifier, parent=None):
        super().__init__(parent)
        self.classifier = classifier
        self.setWindowTitle("Gerenciar Categorias de Fundo")
        self.resize(450, 550)
        self._build_ui()
        self._refresh_list()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Instruções
        info = QLabel(
            "<b>Classes fixas:</b> S=Sedimento, F=Fragmento, R=Recife<br>"
            "<b>Classes custom:</b> atalhos 1-9, 0<br>"
            "Máximo 10 classes adicionais."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Lista de classes
        layout.addWidget(QLabel("<b>Categorias:</b>"))
        self.class_list = QListWidget()
        self.class_list.setStyleSheet("""
            QListWidget {
                font-size: 13px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #ddd;
            }
        """)
        layout.addWidget(self.class_list)

        # Adicionar nova
        add_group = QGroupBox("Adicionar Nova Categoria")
        add_layout = QFormLayout(add_group)

        self.new_name = QLineEdit()
        self.new_name.setPlaceholderText("Ex: Areia, Rocha, Algá...")
        add_layout.addRow("Nome:", self.new_name)

        color_layout = QHBoxLayout()
        self.color_preview = QLabel("    ")
        self.color_preview.setFixedSize(30, 30)
        self.color_preview.setStyleSheet("background-color: #808080; border: 1px solid #333;")
        self.selected_color = None

        color_btn = QPushButton("Escolher Cor")
        color_btn.clicked.connect(self._choose_color)
        color_layout.addWidget(self.color_preview)
        color_layout.addWidget(color_btn)
        color_layout.addStretch()
        add_layout.addRow("Cor:", color_layout)

        add_btn = QPushButton("Adicionar")
        add_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        add_btn.clicked.connect(self._add_class)
        add_layout.addRow(add_btn)

        layout.addWidget(add_group)

        # Botão remover
        remove_btn = QPushButton("🗑 Remover Selecionada")
        remove_btn.setStyleSheet("background-color: #f44336; color: white;")
        remove_btn.clicked.connect(self._remove_class)
        layout.addWidget(remove_btn)

        # Fechar
        layout.addStretch()
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Salvar Configuração")
        save_btn.clicked.connect(self._save_config)
        close_btn = QPushButton("Fechar")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(save_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def _choose_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.selected_color = color.name()
            self.color_preview.setStyleSheet(
                f"background-color: {self.selected_color}; border: 1px solid #333;"
            )

    def _add_class(self):
        name = self.new_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Erro", "Digite um nome para a categoria")
            return

        success, shortcut, msg = self.classifier.add_custom_class(
            name, self.selected_color
        )

        if success:
            self._refresh_list()
            self.new_name.clear()
            self.selected_color = None
            self.color_preview.setStyleSheet("background-color: #808080; border: 1px solid #333;")
            self.classes_changed.emit()
        else:
            QMessageBox.warning(self, "Erro", msg)

    def _remove_class(self):
        current = self.class_list.currentItem()
        if not current:
            return

        # Extrair nome da string formatada
        text = current.text()
        # Formato: "[1] Areia (#FFD700)" ou "[S] Sedimento (#8B4513) [FIXO]"
        if "[FIXO]" in text:
            QMessageBox.warning(self, "Erro", "Classes fixas (S, F, R) não podem ser removidas!")
            return

        # Pegar o nome entre ] e (
        import re
        match = re.search(r'\] (.+?) \(', text)
        if match:
            name = match.group(1)
            reply = QMessageBox.question(self, "Confirmar", 
                                       f"Remover categoria '{name}'?")
            if reply == QMessageBox.StandardButton.Yes:
                success, msg = self.classifier.remove_custom_class(name)
                if success:
                    self._refresh_list()
                    self.classes_changed.emit()
                else:
                    QMessageBox.warning(self, "Erro", msg)

    def _refresh_list(self):
        self.class_list.clear()
        classes = self.classifier.list_all_classes()

        for cls in classes:
            shortcut = cls["shortcut"]
            name = cls["name"]
            color = cls["color"]
            type_ = cls["type"]

            # Criar item com ícone de cor
            item_text = f"[{shortcut}] {name} ({color})"
            if type_ == "fixed":
                item_text += " [FIXO]"

            item = QListWidgetItem(item_text)

            # Ícone colorido
            pixmap = QPixmap(16, 16)
            pixmap.fill(QColor(color))
            item.setIcon(QIcon(pixmap))

            self.class_list.addItem(item)

    def _save_config(self):
        path = self.classifier.save_config()
        QMessageBox.information(self, "Salvo", f"Configuração salva em:\n{path}")
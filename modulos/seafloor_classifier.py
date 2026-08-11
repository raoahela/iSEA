"""
Seafloor Classification Module for iSEA 
Classes: Sedimento, Coral_Fragmento, Recife_Coral + custom classes
"""

import cv2
import os
import shutil
import colorsys
import json
import numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import matplotlib
matplotlib.use('qtagg')
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

from .translations import TEXTS


def _t(key, lang="pt", *args):
    """Translation helper."""
    text = TEXTS.get(lang, TEXTS["pt"]).get(key, key)
    if args:
        return text.format(*args)
    return text


def _tc(name, lang="pt"):
    """Translate class name for display."""
    if lang == "en":
        mapping = {
            "Sedimento": "Sediment",
            "Coral_Fragmento": "Coral Fragment",
            "Recife_Coral": "Coral Reef"
        }
        return mapping.get(name, name)
    return name


# =============================================================================
# CENTER CROP TO 299x299
# =============================================================================

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
            raise ValueError(_t("reference_stats_not_computed", self.language))

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
# DATA AUGMENTATION (ONLY applied to training set AFTER split)
# =============================================================================

def augment_image(img):
    """Apply random augmentation to a single image."""
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


def balance_classes_by_augmentation(X_train, y_train, augment_factor=3, max_oversample_ratio=2.0):
    """
    Balance training classes by creating augmented copies.
    ONLY called AFTER train/val/test split, on training set only.
    """
    counts = Counter(y_train)
    max_count = max(counts.values())

    # Cap the target to avoid extreme oversampling
    target_count = min(max_count, int(min(counts.values()) * max_oversample_ratio))
    target_count = max(target_count, max_count)

    X_balanced, y_balanced = [], []

    for cls in sorted(counts.keys()):
        cls_indices = np.where(y_train == cls)[0]
        cls_images = [X_train[i] for i in cls_indices]
        cls_count = len(cls_images)

        # Add originals
        X_balanced.extend(cls_images)
        y_balanced.extend([cls] * cls_count)

        # Calculate how many augmented samples needed
        n_needed = target_count - cls_count
        if n_needed <= 0:
            continue

        n_augment = min(n_needed * augment_factor, int(cls_count * max_oversample_ratio))
        n_augment = max(n_augment, n_needed)

        for _ in range(n_augment):
            img = cls_images[np.random.randint(len(cls_images))]
            aug_img = augment_image(img)
            X_balanced.append(aug_img)
            y_balanced.append(cls)

    return np.array(X_balanced), np.array(y_balanced)


# =============================================================================
# DEEP FEATURE EXTRACTOR (for PCA visualization and k-NN expansion)
# =============================================================================

class DeepFeatureExtractor:
    """
    Extracts 2048-dimensional features from InceptionV3 avg_pool layer.
    These features represent what the network "sees" and are used for:
    1. PCA visualization for user example selection
    2. k-NN label expansion
    3. Unsupervised clustering

    CRITICAL: This is the SAME feature space the classifier will learn,
    ensuring user selection is relevant to the model.
    """
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_model()

    def _setup_model(self):
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        # Replace fc with Identity to get 2048-dim features from avg_pool
        self.model.fc = nn.Identity()
        self.model = self.model.to(self.device).eval()

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, images, batch_size=32, progress_callback=None):
        """Extract deep features from images."""
        features = []
        total = len(images)
        with torch.no_grad():
            for i in range(0, total, batch_size):
                batch = images[i:i + batch_size]
                tensors = []
                for img in batch:
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    elif img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                    elif img.shape[2] == 3 and isinstance(img, np.ndarray):
                        # Assume BGR from OpenCV, convert to RGB
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    tensors.append(self.preprocess(img))

                batch_tensor = torch.stack(tensors).to(self.device)
                feats = self.model(batch_tensor).cpu().numpy()
                features.extend(feats)

                if progress_callback:
                    pct = min(100, int((i + len(batch)) / total * 100))
                    progress_callback(pct)

        return np.array(features)

    def extract_single(self, img):
        """Extract features from a single image."""
        return self.extract([img])[0]


# =============================================================================
# ENTROPY CLASSIFIER (Independent baseline model)
# =============================================================================

class EntropyClassifier:
    """
    Classificador baseado em entropia com thresholds otimizados para S/F/R.
    Usado como modelo baseline INDEPENDENTE — NÃO guia a seleção do CNN.
    As 21 features manuais pertencem a este classificador, não ao CNN.
    """
    def __init__(self, thresholds=(6.471, 6.980), class_names=None):
        self.thresholds = thresholds
        self.class_names = class_names or ["Sedimento", "Coral_Fragmento", "Recife_Coral"]
        self.entropies = None
        self.pseudo_labels = None
        self.confidence_scores = None

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
        class_centers = [6.192, 6.750, 7.209]

        for i, e in enumerate(self.entropies):
            dists = [abs(e - c) for c in class_centers]
            min_dist = min(dists)
            sorted_dists = sorted(dists)
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
        print(_t("entropy_class_report", self.language))
        print("=" * 60)
        print(_t("total_images", self.language, len(self.entropies)))
        stats = self.get_stats()
        print(_t("global_entropy", self.language, stats["mean"], stats["std"]))
        print(_t("range_format", self.language, stats["min"], stats["max"]))
        if labels is not None:
            print("\n  " + _t("per_class", self.language))
            for i, name in enumerate(self.class_names):
                mask = labels == i
                n = np.sum(mask)
                if n > 0:
                    mean = np.mean(self.entropies[mask])
                    std = np.std(self.entropies[mask])
                    print(_t("class_stats", self.language, name, n, mean, std))
        if self.pseudo_labels is not None:
            n_safe = np.sum(self.pseudo_labels != -1)
            print(_t("safe_pseudo_labels", self.language, n_safe, len(self.entropies), n_safe/len(self.entropies)))
        print("=" * 60)


# =============================================================================
# MANUAL FEATURE EXTRACTOR (21 features - ONLY for EntropyClassifier analysis)
# =============================================================================

class ManualFeatureExtractor:
    """
    Extrai as 21 características manuais (entropia, brilho, textura, etc.)
    Usadas APENAS para:
    - Análise exploratória do EntropyClassifier
    - Baseline independente
    - NUNCA para guiar seleção de exemplos do CNN
    """
    def __init__(self):
        pass

    def extract(self, images):
        features = []
        for img in images:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img

            h, w = gray.shape

            # Entropia (replicada para peso)
            global_entropy = shannon_entropy(gray)

            if h > 64 and w > 64:
                small = cv2.resize(gray, (w//4, h//4), interpolation=cv2.INTER_AREA)
                med = cv2.resize(gray, (w//2, h//2), interpolation=cv2.INTER_AREA)
                ent_small = shannon_entropy(small)
                ent_med = shannon_entropy(med)
            else:
                ent_small = global_entropy
                ent_med = global_entropy

            # Quadrantes
            q1 = gray[:h//2, :w//2]
            q2 = gray[:h//2, w//2:]
            q3 = gray[h//2:, :w//2]
            q4 = gray[h//2:, w//2:]
            quad_ents = [shannon_entropy(q) for q in [q1, q2, q3, q4] if q.size > 0]

            # Gradiente
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(sobelx**2 + sobely**2)
            grad_entropy = shannon_entropy((grad_mag / (grad_mag.max() + 1e-8) * 255).astype(np.uint8))

            # Estatísticas
            mean_int = np.mean(gray)
            std_int = np.std(gray)
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
            hist = hist / (hist.sum() + 1e-10)
            cumhist = np.cumsum(hist)
            p25 = np.searchsorted(cumhist, 0.25) * 8
            p50 = np.searchsorted(cumhist, 0.50) * 8
            p75 = np.searchsorted(cumhist, 0.75) * 8

            # GLCM
            glcm = graycomatrix(gray, distances=[1, 2], angles=[0, np.pi/4], 
                               levels=256, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            energy = graycoprops(glcm, 'energy').mean()

            # Feature vector
            entropy_features = [global_entropy] * 4 + [ent_small, ent_med, grad_entropy]
            spatial_features = quad_ents + [np.mean(quad_ents), np.std(quad_ents)]
            intensity_features = [mean_int, std_int, p25, p50, p75]
            texture_features = [contrast, homogeneity, energy]

            feat = np.concatenate([
                entropy_features, spatial_features, intensity_features, texture_features
            ])
            features.append(feat)

        return np.array(features)


# =============================================================================
# SEMI-AUTOMATIC LABELER (REVISED - uses DEEP FEATURES for PCA and k-NN)
# =============================================================================

class SemiAutoLabeler:
    def __init__(self, n_examples_per_class=10):
        self.n_examples = n_examples_per_class
        self.labels = {}
        self.class_names = {}
        self.deep_feature_extractor = None

    def set_deep_feature_extractor(self, extractor):
        """Set the deep feature extractor for PCA and k-NN."""
        self.deep_feature_extractor = extractor

    def interactive_labeling_qt(self, images, image_names, pca_2d, n_classes, 
                                 class_names_list=None, parent_widget=None, language="pt"):
        """
        INTERACTIVE labeling usando Qt Dialog.
        pca_2d MUST be computed from DEEP FEATURES (InceptionV3), not manual features.
        """
        if class_names_list is None:
            class_names_list = [f"Class_{i}" for i in range(n_classes)]

        all_labels = {}

        for cls in range(n_classes):
            cls_name = class_names_list[cls] if cls < len(class_names_list) else f"Class_{cls}"
            self.class_names[cls] = cls_name

            dialog = InteractiveLabelingDialog(
                images, image_names, pca_2d, cls_name, cls,
                self.n_examples, parent=parent_widget, language=language
            )

            result = dialog.exec()

            if result == QDialog.DialogCode.Accepted and dialog.selected_indices:
                for idx in dialog.selected_indices:
                    all_labels[idx] = cls
                print(_t("examples_selected_by_user", language, _tc(cls_name, language), len(dialog.selected_indices)))
            else:
                auto = self._auto_select_uniform(pca_2d, images, cls,
                                                 exclude=list(all_labels.keys()))
                for idx in auto[:self.n_examples]:
                    all_labels[idx] = cls
                print(_t("examples_auto_selected", language, _tc(cls_name, language), len(auto[:self.n_examples])))

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
        """
        Expand labels using k-NN in DEEP FEATURE space (not manual features).
        This ensures expanded labels are consistent with what the CNN sees.
        """
        if self.deep_feature_extractor is None:
            raise ValueError(_t("deep_feature_extractor_not_set", self.language))

        # Extract deep features for ALL images
        print(_t("extracting_for_knn_expansion", self.language))
        deep_features = self.deep_feature_extractor.extract(images)

        # Scale deep features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(deep_features)

        # Optional: reduce to 50 dims with PCA for k-NN efficiency
        n_components = min(50, features_scaled.shape[0], features_scaled.shape[1])
        if n_components < features_scaled.shape[1]:
            pca = PCA(n_components=n_components)
            features_pca = pca.fit_transform(features_scaled)
        else:
            features_pca = features_scaled

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

    def get_training_data(self, images, image_names):
        """
        Get labeled training data WITHOUT augmentation.
        Augmentation is applied LATER, after train/val split.
        """
        X_train, y_train, names_train = [], [], []
        for idx, label in self.labels.items():
            X_train.append(images[idx])
            y_train.append(label)
            names_train.append(image_names[idx])

        y_train = np.array(y_train)
        unique_labels = sorted(np.unique(y_train))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        y_train = np.array([label_map[l] for l in y_train])

        return np.array(X_train), y_train, names_train


# =============================================================================
# INTERACTIVE LABELING DIALOG
# =============================================================================

class InteractiveLabelingDialog(QDialog):
    """Dialog for interactive selection of training examples per class."""

    def __init__(self, images, image_names, pca_2d, class_name, class_id,
                 n_examples, parent=None, language="pt"):
        super().__init__(parent)
        self.images = images
        self.image_names = image_names
        self.pca_2d = pca_2d
        self.class_name = class_name
        self.class_id = class_id
        self.n_examples = n_examples
        self.selected_indices = []

        self.setWindowTitle(_t("select_examples_title", self.language, _tc(class_name, self.language)))
        self.resize(900, 700)

        self._build_ui()
        self._populate_grid()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        self.instruction = QLabel(
            _t("click_images_select_examples", self.language, _tc(self.class_name, self.language)) + "<br>" +
            _t("selected_count", self.language, 0, self.n_examples) + "<br>" +
            "<i>" + _t("pca_deep_features_hint", self.language) + "</i>"
        )
        self.instruction.setWordWrap(True)
        layout.addWidget(self.instruction)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(5)
        scroll.setWidget(self.grid_widget)
        layout.addWidget(scroll)

        btn_layout = QHBoxLayout()

        self.auto_btn = QPushButton(_t("auto_select", self.language))
        self.auto_btn.setToolTip(_t("auto_select_tooltip", self.language))
        self.auto_btn.clicked.connect(self._auto_select)

        self.done_btn = QPushButton(_t("done", self.language))
        self.done_btn.setEnabled(False)
        self.done_btn.clicked.connect(self.accept)

        btn_layout.addWidget(self.auto_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.done_btn)
        layout.addLayout(btn_layout)

    def _populate_grid(self):
        n_cols = 6
        n_rows_grid = (len(self.images) + n_cols - 1) // n_cols

        x_min, x_max = self.pca_2d[:, 0].min(), self.pca_2d[:, 0].max()
        y_min, y_max = self.pca_2d[:, 1].min(), self.pca_2d[:, 1].max()

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

        self.thumb_buttons = {}
        for (row, col), idx in grid_to_idx.items():
            img = self.images[idx]
            thumb = cv2.resize(img, (120, 120), interpolation=cv2.INTER_AREA)
            thumb_rgb = thumb

            h, w, ch = thumb_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(thumb_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)

            btn = QPushButton()
            btn.setFixedSize(130, 130)
            btn.setIcon(QIcon(pixmap))
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
        ready_text = _t("ready_to_finish", self.language) if count >= self.n_examples else _t("continue_selecting", self.language)
        self.instruction.setText(
            _t("click_images_select_examples", self.language, _tc(self.class_name, self.language)) + "<br>" +
            _t("selected_count", self.language, count, self.n_examples) + "<br>" +
            "<i>" + _t("pca_deep_features_hint", self.language) + "</i><br>" +
            ready_text
        )

        self.done_btn.setEnabled(count >= self.n_examples)

    def _auto_select(self):
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
# REVISED INCEPTIONV3 CLASSIFIER
# =============================================================================

class FastInceptionV3Classifier:
    """
    Revised InceptionV3 classifier with:
    - Proper train/val/test split BEFORE augmentation
    - 2-phase training: 5 epochs frozen + 8 epochs fine-tune
    - Early stopping with patience
    - Only Mixed_7b + Mixed_7c unfrozen for fine-tuning
    """
    def __init__(self, n_classes, device=None):
        self.n_classes = n_classes
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.best_val_acc = 0.0

        # Transforms (no resize - images already 299x299)
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.scaler = GradScaler() if torch.cuda.is_available() else None

    def _build_model(self, freeze_backbone=True):
        """Build model with optional backbone freezing."""
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)

        if freeze_backbone:
            print(_t("freezing_backbone", self.language))
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            print(_t("unfreezing_layers", self.language))
            # Start frozen
            for param in self.model.parameters():
                param.requires_grad = False
            # Only unfreeze the LAST TWO blocks (not 7a)
            for param in self.model.Mixed_7c.parameters():
                param.requires_grad = True
            for param in self.model.Mixed_7b.parameters():
                param.requires_grad = True
            # Keep 7a frozen - too early in network for few data

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.n_classes)
        self.model = self.model.to(self.device)

    def train_phase(self, X_train, y_train, X_val, y_val, 
                    epochs, lr, patience, freeze_backbone=True, progress_callback=None):
        """Train a single phase with early stopping."""
        self._build_model(freeze_backbone=freeze_backbone)

        train_dataset = FastImageDataset(X_train, y_train, self.train_transform)
        val_dataset = FastImageDataset(X_val, y_val, self.val_transform)

        pin_memory = self.device.type == 'cuda'
        train_loader = DataLoader(
            train_dataset, batch_size=64, shuffle=True,
            num_workers=0, pin_memory=pin_memory
        )
        val_loader = DataLoader(
            val_dataset, batch_size=64, shuffle=False,
            num_workers=0, pin_memory=pin_memory
        )

        # Class weights for imbalance
        counts = Counter(y_train)
        total = len(y_train)
        weights = torch.tensor([
            total / (self.n_classes * counts.get(c, 1))
            for c in range(self.n_classes)
        ], dtype=torch.float32).to(self.device)
        print(_t("class_weights", self.language, weights.cpu().numpy()))

        criterion = nn.CrossEntropyLoss(weight=weights)

        # Optimizer: different LR for backbone vs classifier
        params = [{'params': self.model.fc.parameters(), 'lr': lr}]
        if not freeze_backbone:
            params.append({
                'params': list(self.model.Mixed_7b.parameters()) +
                          list(self.model.Mixed_7c.parameters()),
                'lr': lr * 0.1  # 10x lower for fine-tuning layers
            })

        optimizer = torch.optim.Adam(params)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=max(1, patience - 1)
        )

        best_val_acc = 0.0
        epochs_no_improve = 0
        best_state = None

        for epoch in range(epochs):
            if progress_callback:
                progress_callback(epoch + 1, epochs)

            # Training
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

            # Validation
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

            print(_t("epoch_stats", self.language, epoch+1, epochs, train_loss/train_total, train_acc, val_acc))
            print(_t("val_predicted", self.language, dict(val_dist)))
            print(_t("val_true", self.language, dict(true_dist)))

            scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                print(_t("new_best_val_acc", self.language, val_acc))
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(_t("early_stopping_epoch", self.language, epoch+1))
                    break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        print(_t("best_val_accuracy", self.language, best_val_acc))
        return best_val_acc

    def train_two_phase(self, X_train, y_train, X_val, y_val, progress_callback=None):
        """
        Two-phase training:
        Phase 1: Transfer learning (frozen backbone, 5 epochs, lr=0.001)
        Phase 2: Fine-tuning (unfreeze 7b+7c, 8 epochs, lr=0.0001)
        """
        print(f"\n{'='*60}")
        print(_t("phase1_transfer_learning", self.language))
        print(f"{'='*60}")
        print(_t("training_samples", self.language, len(X_train)))
        print(_t("validation_samples", self.language, len(X_val)))

        def _phase1_progress(epoch, total):
            if progress_callback:
                # Phase 1 = 5 epochs, map to 60-75%
                pct = 60 + int((epoch / total) * 15)
                progress_callback(_t("training_phase1_frozen", self.language), pct)

        val_acc_1 = self.train_phase(
            X_train, y_train, X_val, y_val,
            epochs=5, lr=0.001, patience=2, freeze_backbone=True,
            progress_callback=_phase1_progress
        )

        # Save phase 1 weights
        phase1_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        print(f"\n{'='*60}")
        print(_t("phase2_fine_tuning", self.language))
        print(f"{'='*60}")

        # Rebuild with unfrozen layers, load phase 1 weights
        self._build_model(freeze_backbone=False)
        self.model.load_state_dict(phase1_state)

        def _phase2_progress(epoch, total):
            if progress_callback:
                # Phase 2 = 8 epochs, map to 75-90%
                pct = 75 + int((epoch / total) * 15)
                progress_callback(_t("training_phase2_finetune", self.language), pct)

        val_acc_2 = self.train_phase(
            X_train, y_train, X_val, y_val,
            epochs=8, lr=0.0001, patience=3, freeze_backbone=False,
            progress_callback=_phase2_progress
        )

        print(f"\n{'='*60}")
        print(_t("final_results", self.language, val_acc_1, val_acc_2))
        print(f"{'='*60}")

        # Save best model
        best_path = Path("best_model_fast.pth").resolve()
        torch.save(self.model.state_dict(), str(best_path))
        print(_t("model_state_dict_saved", self.language, best_path))

    def predict(self, images, batch_size=64):
        self.model.eval()
        dummy_labels = np.zeros(len(images), dtype=np.int64)
        dataset = FastImageDataset(images, dummy_labels, self.val_transform)
        pin_memory = self.device.type == 'cuda'
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=pin_memory
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

    def extract_features(self, images, batch_size=64):
        """Extract 2048-dim features from avg_pool layer."""
        self.model.eval()

        # Temporarily replace fc with Identity
        original_fc = self.model.fc
        self.model.fc = nn.Identity()

        dummy_labels = np.zeros(len(images), dtype=np.int64)
        dataset = FastImageDataset(images, dummy_labels, self.val_transform)
        pin_memory = self.device.type == 'cuda'
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=pin_memory
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
# ADAPTIVE ENSEMBLE CLASSIFIER (Entropy + CNN Fusion)
# =============================================================================

class AdaptiveEnsembleClassifier:
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
# MAIN SEAFLOOR CLASSIFICATION CLASS (REVISED)
# =============================================================================

class SeafloorClassifier:
    """
    Classificador de fundo marinho com classes dinâmicas - VERSÃO REVISADA.

    CORREÇÕES CRÍTICAS:
    1. PCA para seleção usa DEEP FEATURES (InceptionV3, 2048 dims)
    2. k-NN para expansão de labels usa DEEP FEATURES
    3. Split treino/val/teste ANTES de qualquer augmentation
    4. Treinamento em 2 fases: 5 épocas (frozen) + 8 épocas (fine-tune)
    5. Fine-tuning só descongela Mixed_7b + Mixed_7c

    Classes fixas (atalhos de teclado):
        S → Sedimento
        F → Coral_Fragmento  
        R → Recife_Coral 

    Classes customizadas (atalhos numéricos):
        1, 2, 3... 9, 0 → adicionadas pelo usuário via menu
    """

    FIXED_CLASSES = {
        "S": "Sedimento",
        "F": "Coral_Fragmento", 
        "R": "Recife_Coral"
    }

    FIXED_COLORS = {
        "Sedimento": "#8B4513",
        "Coral_Fragmento": "#FF8C00",
        "Recife_Coral": "#00CED1"
    }

    CUSTOM_SHORTCUTS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]

    def __init__(self, n_clusters=3, max_examples=None,
                 normalize_colors=True, reference_path=None,
                 n_examples_per_class=10, use_augmentation=True,
                 fast_mode=True, output_dir="seafloor_output",
                 crop_to_299=True,
                 crop_ratio=0.85,
                 use_adaptive_ensemble=False,
                 entropy_margin=0.3,
                 custom_classes=None,
                 custom_colors=None,
                 language="pt"):

        # Inicializar classes fixas
        self.fixed_class_names = list(self.FIXED_CLASSES.values())
        self.fixed_colors = dict(self.FIXED_COLORS)

        # Inicializar classes customizadas
        self.custom_classes = {}
        self.custom_colors = {}

        if custom_classes:
            for i, name in enumerate(custom_classes):
                if i < len(self.CUSTOM_SHORTCUTS):
                    shortcut = self.CUSTOM_SHORTCUTS[i]
                    self.custom_classes[shortcut] = name
                    if custom_colors and name in custom_colors:
                        self.custom_colors[name] = custom_colors[name]
                    else:
                        self.custom_colors[name] = self._generate_color(name)

        self.class_names = self.fixed_class_names + list(self.custom_classes.values())
        self.class_colors = {**self.fixed_colors, **self.custom_colors}
        self.n_clusters = len(self.class_names)

        self.max_examples = max_examples
        self.normalize_colors = normalize_colors
        self.n_examples_per_class = n_examples_per_class
        self.use_augmentation = use_augmentation
        self.fast_mode = fast_mode
        self.output_dir = Path(output_dir)
        self.crop_to_299 = crop_to_299
        self.crop_ratio = crop_ratio
        self.cropper = CenterCropTo299(crop_ratio=crop_ratio) if crop_to_299 else None
        self.use_adaptive_ensemble = use_adaptive_ensemble
        self.entropy_margin = entropy_margin
        self.language = language

        self.normalizer = None
        if self.normalize_colors:
            self.normalizer = ColorNormalizer(reference_path=reference_path)

        # Initialize deep feature extractor (CRITICAL: shared across pipeline)
        self.deep_feature_extractor = DeepFeatureExtractor()

        # Labeler now uses deep features
        self.labeler = SemiAutoLabeler(n_examples_per_class=n_examples_per_class)
        self.labeler.set_deep_feature_extractor(self.deep_feature_extractor)

        self.trained_classifier = None

        self.entropy_classifier = EntropyClassifier(
            thresholds=(6.471, 6.980),
            class_names=self.class_names
        )
        self.adaptive_ensemble = None

        # Criar diretórios
        self.output_dir.mkdir(exist_ok=True)
        for sub in ["clusters", "normalized", "supervised", "pca_plots", "cropped", 
                    "train", "val", "test"]:
            (self.output_dir / sub).mkdir(exist_ok=True)
        self._refresh_cluster_dirs()

    def _generate_color(self, name):
        hue = hash(name) % 360 / 360.0
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

    def _refresh_cluster_dirs(self):
        for i in range(self.n_clusters):
            (self.output_dir / "clusters" / f"cluster{i}").mkdir(exist_ok=True)
            (self.output_dir / "supervised" / f"class{i}").mkdir(exist_ok=True)

    def get_class_by_shortcut(self, shortcut):
        shortcut = shortcut.upper() if shortcut in ["s", "f", "r"] else shortcut
        if shortcut in self.FIXED_CLASSES:
            return self.FIXED_CLASSES[shortcut]
        if shortcut in self.custom_classes:
            return self.custom_classes[shortcut]
        return None

    def get_shortcut_for_class(self, class_name):
        for shortcut, name in self.FIXED_CLASSES.items():
            if name == class_name:
                return shortcut
        for shortcut, name in self.custom_classes.items():
            if name == class_name:
                return shortcut
        return None

    def add_custom_class(self, name, color=None):
        if name in self.class_names:
            return False, None, _t("seafloor_class_exists", self.language, name)
        available = [s for s in self.CUSTOM_SHORTCUTS if s not in self.custom_classes]
        if not available:
            return False, None, _t("seafloor_max_custom", self.language)
        shortcut = available[0]
        self.custom_classes[shortcut] = name
        self.custom_colors[name] = color if color else self._generate_color(name)
        self.class_names = self.fixed_class_names + list(self.custom_classes.values())
        self.class_colors[name] = self.custom_colors[name]
        self.n_clusters = len(self.class_names)
        self.entropy_classifier.class_names = self.class_names
        self._refresh_cluster_dirs()
        return True, shortcut, _t("category_added_with_shortcut", self.language, name, shortcut)

    def remove_custom_class(self, name_or_shortcut):
        name = name_or_shortcut
        if name_or_shortcut in self.custom_classes:
            name = self.custom_classes[name_or_shortcut]
        if name not in self.custom_classes.values():
            return False, _t("class_not_found_or_fixed", self.language, name)
        shortcut_to_remove = None
        for s, n in self.custom_classes.items():
            if n == name:
                shortcut_to_remove = s
                break
        if shortcut_to_remove:
            del self.custom_classes[shortcut_to_remove]
        self.custom_colors.pop(name, None)
        self.class_names = self.fixed_class_names + list(self.custom_classes.values())
        self.n_clusters = len(self.class_names)
        self.entropy_classifier.class_names = self.class_names
        self._reorganize_shortcuts()
        return True, _t("category_removed", self.language, name)

    def _reorganize_shortcuts(self):
        classes = list(self.custom_classes.values())
        self.custom_classes = {}
        for i, name in enumerate(classes):
            if i < len(self.CUSTOM_SHORTCUTS):
                self.custom_classes[self.CUSTOM_SHORTCUTS[i]] = name
        self.class_names = self.fixed_class_names + list(self.custom_classes.values())
        self.n_clusters = len(self.class_names)

    def list_all_classes(self):
        result = []
        for shortcut, name in self.FIXED_CLASSES.items():
            result.append({
                "shortcut": shortcut, "name": _tc(name, self.language),
                "internal_name": name,
                "color": self.fixed_colors[name], "type": "fixed"
            })
        for shortcut, name in self.custom_classes.items():
            result.append({
                "shortcut": shortcut, "name": _tc(name, self.language),
                "internal_name": name,
                "color": self.custom_colors.get(name, "#808080"), "type": "custom"
            })
        return result

    def save_config(self, path=None):
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
            "use_adaptive_ensemble": self.use_adaptive_ensemble
        }
        path = path or (self.output_dir / "classifier_config.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return path

    def load_config(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        self.custom_classes = config.get("custom_classes", {})
        self.custom_colors = config.get("custom_colors", {})
        self.class_names = config.get("class_names", self.fixed_class_names)
        self.class_colors = {**self.fixed_colors, **self.custom_colors}
        self.n_clusters = config.get("n_clusters", len(self.class_names))
        self.entropy_classifier.class_names = self.class_names
        self._refresh_cluster_dirs()
        return config

    def load_images_from_folder(self, folder_path):
        folder = Path(folder_path)
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]:
            image_paths.extend(folder.glob(ext))

        # Deduplicar (evita duplicatas em sistemas case-insensitive como Windows)
        image_paths = sorted(set(image_paths))
        if not image_paths:
            raise ValueError(_t("no_images_found_in_folder", self.language, folder_path))

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
            raise ValueError(_t("no_valid_images_loaded", self.language, folder_path))

        print(_t("loaded_images_from", self.language, len(images_raw), folder_path))
        return images_raw, images_rgb, image_names

    def save_model(self, path=None):
        if self.trained_classifier is None:
            raise ValueError(_t("no_trained_model_to_save", self.language))
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
            "use_adaptive_ensemble": self.use_adaptive_ensemble,
            "entropy_margin": self.entropy_margin,
            "ref_mean": self.normalizer.ref_mean.tolist() if (self.normalizer and self.normalizer.ref_mean is not None) else None,
            "ref_std": self.normalizer.ref_std.tolist() if (self.normalizer and self.normalizer.ref_std is not None) else None,
            "reference_path": str(self.normalizer.reference_path) if self.normalizer else None,
        }
        torch.save(checkpoint, str(path))
        print(_t("model_saved_to_path", self.language, path))
        return str(path)

    def load_model(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(_t("model_not_found", self.language, path))
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
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
        self.use_adaptive_ensemble = checkpoint.get("use_adaptive_ensemble", self.use_adaptive_ensemble)
        self.entropy_margin = checkpoint.get("entropy_margin", self.entropy_margin)
        ref_mean = checkpoint.get("ref_mean")
        ref_std = checkpoint.get("ref_std")
        ref_path = checkpoint.get("reference_path")
        if ref_mean is not None and ref_std is not None:
            if self.normalizer is None:
                self.normalizer = ColorNormalizer(reference_path=ref_path)
            self.normalizer.ref_mean = np.array(ref_mean, dtype=np.float32)
            self.normalizer.ref_std = np.array(ref_std, dtype=np.float32)
        n_classes = checkpoint.get("n_classes", len(self.class_names))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trained_classifier = FastInceptionV3Classifier(n_classes=n_classes, device=device)
        self.trained_classifier._build_model(freeze_backbone=False)
        self.trained_classifier.model.load_state_dict(checkpoint["model_state_dict"])
        self.trained_classifier.model.eval()
        self.entropy_classifier.class_names = self.class_names
        print(_t("model_loaded_from", self.language, path))
        print(_t("classes_list", self.language, [_tc(n, self.language) for n in self.class_names]))
        return True

    def has_trained_model(self):
        return self.trained_classifier is not None

    def predict_single_frame(self, frame_bgr):
        if self.trained_classifier is None:
            raise ValueError(_t("classifier_not_trained", self.language))
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
        if class_id >= len(self.class_names):
            class_id = len(self.class_names) - 1
        class_name = self.class_names[class_id]
        return {
            "class_id": class_id,
            "class_name": _tc(class_name, self.language),
            "confidence": confidence,
            "color": self.class_colors.get(class_name, "#FFFFFF"),
            "shortcut": self.get_shortcut_for_class(class_name)
        }


    # =====================================================================
    # MAIN CLASSIFICATION PIPELINE (REVISED)
    # =====================================================================

    def classify_images(self, images_rgb, images_raw, image_names, folder_path=None,
                         parent_widget=None, progress_callback=None, status_callback=None,
                         skip_labeling=False):
        """
        Run full classification workflow WITH CRITICAL FIXES:
        1. PCA uses DEEP FEATURES (InceptionV3, 2048-dim)
        2. k-NN expansion uses DEEP FEATURES
        3. Train/Val/Test split BEFORE augmentation
        4. 2-phase training: 5 epochs (frozen) + 8 epochs (fine-tune)
        """

        def _status(msg):
            ts = datetime.now().strftime("%H:%M:%S")
            full = f"[{ts}] {msg}"
            print(full)
            if status_callback:
                status_callback(full)

        def _progress(msg, pct):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg} ({pct}%)")
            if progress_callback:
                progress_callback(msg, pct)

        if not skip_labeling:
            # ===== Step 1: Extract DEEP FEATURES for PCA (CRITICAL FIX) =====
            _status(_t("step1_extracting_deep_features", self.language))
            print(_t("using_inceptionv3", self.language))
            print(_t("cnn_sees_hint", self.language))

            def feat_progress(pct):
                _progress(_t("deep_features_progress", self.language), pct)

            deep_features = self.deep_feature_extractor.extract(images_rgb, progress_callback=feat_progress)
            print(_t("deep_features_shape", self.language, deep_features.shape))

            # Scale and reduce for PCA visualization
            scaler = StandardScaler()
            deep_scaled = scaler.fit_transform(deep_features)
            pca_2d = PCA(n_components=2).fit_transform(deep_scaled)
            pca_obj = PCA(n_components=2).fit(deep_scaled)
            print(_t("pca_explained_variance", self.language, pca_obj.explained_variance_ratio_))

            # ===== Step 2: INTERACTIVE labeling via Qt Dialog =====
            _status(_t("step2_interactive_labeling", self.language))
            _progress("Interactive labeling", 20)

            self.labeler.interactive_labeling_qt(
                images_rgb, image_names, pca_2d,
                self.n_clusters, self.class_names,
                parent_widget=parent_widget
            )

            # ===== Step 3: EXPAND labels with k-NN in DEEP FEATURE space =====
            _status(_t("step3_expanding_labels", self.language))
            _progress(_t("knn_expansion_progress", self.language), 30)
            self.labeler.expand_labels_nearest_neighbors(images_rgb, n_neighbors=50)

            n_labeled = len(self.labeler.labels)
            print(_t("total_labeled_after_expansion", self.language, n_labeled, len(images_rgb)))
        else:
            _status(_t("step1_3_skipped", self.language))
            _progress(_t("labels_ready_progress", self.language), 30)
            n_labeled = len(self.labeler.labels)
            print(_t("using_preexisting_labels", self.language, n_labeled))

        # ===== Step 4: CROP to 299×299 =====
        _status("STEP 4: Cropping to 299×299")
        _progress("Cropping", 35)

        if self.crop_to_299 and self.cropper is not None:
            cropped_raw, crop_infos = self.cropper.crop_batch(images_raw, image_names)
            cropped_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in cropped_raw]

            methods = Counter([info['method'] for info in crop_infos])
            print(_t("methods", self.language, dict(methods)))

            for img, name in zip(cropped_raw, image_names):
                out_path = self.output_dir / "cropped" / f"crop_{name}"
                cv2.imwrite(str(out_path), img)

            images_cropped_raw = cropped_raw
            images_cropped_rgb = cropped_rgb
        else:
            images_cropped_raw = images_raw
            images_cropped_rgb = images_rgb

        # ===== Step 5: Color normalization =====
        _status(_t("step5_color_normalization", self.language))
        _progress(_t("color_norm_progress", self.language), 40)

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

        # ===== Step 6: Get labeled data (NO augmentation yet!) =====
        _status(_t("step6_preparing_data", self.language))
        _progress(_t("preparing_data_progress", self.language), 45)

        X_labeled, y_labeled, names_labeled = self.labeler.get_training_data(
            images_normalized_rgb, image_names
        )

        print(_t("labeled_samples", self.language, len(X_labeled)))
        print(_t("class_distribution", self.language, dict(Counter(y_labeled))))

        # ===== Step 7: STRATIFIED SPLIT (CRITICAL FIX - before augmentation!) =====
        _status(_t("step7_stratified_split", self.language))
        _progress(_t("split_progress", self.language), 50)
        print(_t("critical_split_before_aug", self.language))

        # First split: separate test set (15%)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_labeled, y_labeled, 
            test_size=0.15, 
            stratify=y_labeled, 
            random_state=42
        )

        # Second split: separate validation from temp (17.6% of temp ≈ 15% of total)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=0.176,
            stratify=y_temp,
            random_state=42
        )

        print(_t("train_samples", self.language, len(X_train)))
        print(_t("val_samples", self.language, len(X_val)))
        print(_t("test_samples", self.language, len(X_test)))
        print(_t("train_distribution", self.language, dict(Counter(y_train))))
        print(_t("val_distribution", self.language, dict(Counter(y_val))))
        print(_t("test_distribution", self.language, dict(Counter(y_test))))

        # ===== Step 8: Balance training set with augmentation (ONLY train!) =====
        _status(_t("step8_balancing", self.language))
        _progress(_t("data_aug_progress", self.language), 55)

        if self.use_augmentation:
            X_train_bal, y_train_bal = balance_classes_by_augmentation(
                X_train, y_train, augment_factor=3, max_oversample_ratio=2.0
            )
        else:
            X_train_bal, y_train_bal = X_train, y_train

        print(_t("after_balancing", self.language, len(X_train_bal)))
        print(_t("balanced_distribution", self.language, dict(Counter(y_train_bal))))

        # ===== Step 9: Train classifier with 2-phase approach =====
        _status(_t("step9_training", self.language))
        _progress(_t("training_progress", self.language), 60)

        n_classes = len(np.unique(y_train_bal))

        classifier = FastInceptionV3Classifier(n_classes=n_classes)
        classifier.train_two_phase(X_train_bal, y_train_bal, X_val, y_val, 
                                    progress_callback=_progress)

        self.trained_classifier = classifier

        # ===== Step 10: Evaluate on TEST set (never seen before!) =====
        _status(_t("step10_evaluating", self.language))
        _progress(_t("evaluation_progress", self.language), 90)

        test_preds, test_confs = classifier.predict(X_test, batch_size=64)
        test_acc = np.mean(test_preds == y_test)
        print(_t("test_accuracy", self.language, test_acc))
        print(_t("test_confusion", self.language))
        print(_t("predicted", self.language, dict(Counter(test_preds))))
        print(_t("true", self.language, dict(Counter(y_test))))

        # ===== Step 11: Predict on ALL images =====
        _status(_t("step11_predicting", self.language))
        _progress(_t("prediction_progress", self.language), 92)

        supervised_preds, confidences = classifier.predict(
            images_normalized_rgb, batch_size=64
        )

        # ===== Step 12: Unsupervised clustering (for comparison) =====
        _status(_t("step12_clustering", self.language))
        _progress(_t("clustering_progress", self.language), 95)

        unsupervised_preds = self._run_unsupervised(images_normalized_rgb)

        # ===== Step 13: Comparison =====
        _status(_t("step13_comparing", self.language))
        _progress(_t("comparison_progress", self.language), 97)

        self._compare_methods(supervised_preds, unsupervised_preds, confidences)

        # ===== Step 14: Save ORIGINAL images to result folders =====
        _status(_t("step14_saving", self.language))
        _progress(_t("saving_progress", self.language), 98)

        for i, name in enumerate(image_names):
            dst = self.output_dir / "supervised" / f"class{supervised_preds[i]}" / name
            cv2.imwrite(str(dst), images_raw[i])

        for i, name in enumerate(image_names):
            dst = self.output_dir / "clusters" / f"cluster{unsupervised_preds[i]}" / name
            cv2.imwrite(str(dst), images_raw[i])

        # ===== Step 15: Visualizations =====
        _status(_t("step15_visualizations", self.language))
        _progress(_t("visualizations_progress", self.language), 99)

        self._visualize_pca_supervised(images_normalized_rgb, supervised_preds, confidences)
        self._visualize_pca_unsupervised(images_normalized_rgb, unsupervised_preds)

        # Salvar modelo treinado com metadados completos
        _status(_t("saving_trained_model", self.language))
        try:
            model_path = self.save_model()
            config_path = self.save_config()
            _status(_t("model_saved_success", self.language, model_path))
            _status(_t("config_saved_success", self.language, config_path))
        except Exception as e:
            _status(_t("error_saving_model", self.language, e))

        _status(_t("classification_complete", self.language))
        _progress(_t("done_progress", self.language), 100)
        return supervised_preds, unsupervised_preds, confidences

    def _run_unsupervised(self, images_normalized_rgb):
        """Unsupervised clustering com InceptionV3 deep features."""
        print(_t("extracting_for_clustering", self.language))
        cnn_features = self.deep_feature_extractor.extract(images_normalized_rgb)

        scaler = StandardScaler()
        cnn_scaled = scaler.fit_transform(cnn_features)

        n_components = min(512, cnn_scaled.shape[0], cnn_scaled.shape[1])
        images_new = PCA(n_components=n_components).fit_transform(cnn_scaled)

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
            print(_t("cohens_kappa", self.language, kappa))

        print(_t("confidence_above_06", self.language, np.mean(confidences > 0.6)))
        print(_t("supervised_distribution", self.language), Counter(supervised))
        print(_t("unsupervised_distribution", self.language), Counter(unsupervised))

    def _visualize_pca_supervised(self, images, predictions, confidences):
        """Visualização PCA supervisionada com deep features."""
        if self.trained_classifier is None:
            return

        print(_t("extracting_for_pca_viz", self.language))
        trained_features = self.trained_classifier.extract_features(images, batch_size=64)
        scaled = StandardScaler().fit_transform(trained_features)
        pca = PCA(n_components=2)
        pca_2d = pca.fit_transform(scaled)
        var = pca.explained_variance_ratio_

        colors_list = ['#2ecc71', '#9b59b6', '#e74c3c', '#3498db', '#f1c40f', '#1abc9c']

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
                cls_name = _tc(self.class_names[i], self.language) if i < len(self.class_names) else f"Class_{i}"
                ax.scatter(pca_2d[mask, 0], pca_2d[mask, 1],
                        c=colors_list[i % len(colors_list)],
                        s=80, alpha=0.6, edgecolors='black', linewidth=0.8,
                        zorder=10, label=f'{cls_name} ({n_points})')

        info_text = ""
        for i in range(self.n_clusters):
            mask = predictions == i
            if np.sum(mask) > 0:
                cls_name = _tc(self.class_names[i], self.language) if i < len(self.class_names) else f"Class_{i}"
                avg_conf = np.mean(confidences[mask])
                info_text += f"{cls_name}: avg conf={avg_conf:.2f}\n"

        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel('PCA 1', fontsize=12)
        ax.set_ylabel('PCA 2', fontsize=12)
        title = 'Supervised Classification - PCA (TRAINED deep features)'
        if self.normalize_colors:
            title += ' (Color Normalized)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        out = self.output_dir / "pca_plots" / "pca_supervised_trained.png"
        plt.savefig(out, dpi=200, bbox_inches='tight')
        plt.close()
        print(_t("saved_to", self.language, out))

    def _visualize_pca_unsupervised(self, images, predictions):
        """Visualização PCA não-supervisionada com deep features."""
        print(_t("extracting_for_unsupervised_pca", self.language))
        cnn_features = self.deep_feature_extractor.extract(images)

        scaler = StandardScaler()
        cnn_scaled = scaler.fit_transform(cnn_features)
        pca = PCA(n_components=2)
        pca_2d = pca.fit_transform(cnn_scaled)
        var = pca.explained_variance_ratio_

        colors_list = ['#2ecc71', '#9b59b6', '#e74c3c', '#3498db', '#f1c40f', '#1abc9c']

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
        title = 'Unsupervised Classification - PCA (Deep Features)'
        if self.normalize_colors:
            title += ' (Color Normalized)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        out = self.output_dir / "pca_plots" / "pca_unsupervised.png"
        plt.savefig(out, dpi=200, bbox_inches='tight')
        plt.close()
        print(_t("saved_to", self.language, out))


    # =====================================================================
    # TRAIN FROM COLLECTED DATA (REVISED)
    # =====================================================================

    def train_from_collected_data(self, data_dir, min_samples_per_class=5,
                                   parent_widget=None):
        """
        Treina o classificador a partir de frames coletados durante anotação.
        REVISADO: Usa split estratificado ANTES de augmentation.
        """
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise ValueError(_t("directory_not_found", self.language, data_dir))

        # Descobrir classes a partir das subpastas
        class_dirs = [d for d in data_dir.iterdir()
                      if d.is_dir() and any(d.glob("*.jpg"))]

        if len(class_dirs) < 2:
            raise ValueError(_t("min_2_classes_required", self.language, len(class_dirs)))

        # Atualizar classes do classificador
        class_names_from_dirs = sorted([d.name for d in class_dirs])
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

        print("\n--- " + _t("loading_training_data", self.language) + " ---")
        for class_id, class_name in enumerate(self.class_names):
            class_dir = data_dir / class_name
            if not class_dir.exists():
                continue

            class_images = []
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                class_images.extend(class_dir.glob(ext))

            if len(class_images) < min_samples_per_class:
                print(_t("warning_class_few_images", self.language, class_name, len(class_images)))

            for img_path in sorted(class_images):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                images_raw.append(img)
                images_rgb.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                image_names.append(img_path.name)
                labels.append(class_id)

        if len(images_raw) < min_samples_per_class * 2:
            raise ValueError(_t("insufficient_data_total", self.language, len(images_raw)))

        print(_t("total_images_classes", self.language, len(images_raw), [_tc(n, self.language) for n in self.class_names]))

        labels = np.array(labels)

        # Executar workflow completo REVISADO
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
        # NÃO usar sinal/slot cross-thread — usamos chamada direta thread-safe via set_labeling_result

    def set_labeling_result(self, labels):
        """Thread-safe: chamado pela UI thread para entregar o resultado do labeling."""
        with QMutexLocker(self.mutex):
            self._interactive_result = labels
            self.condition.wakeOne()

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
            self.progress.emit(0)
            self.status.emit(_t("loading_images", self.language))
            images_raw, images_rgb, image_names = self.classifier.load_images_from_folder(
                self.batch_folder
            )

            self.progress.emit(5)
            self.status.emit(_t("extracting_deep_features_pca", self.language))

            # REVISED: Use deep features for PCA
            def feat_progress(pct):
                self.progress.emit(5 + int(pct * 0.10))

            deep_features = self.classifier.deep_feature_extractor.extract(
                images_rgb, progress_callback=feat_progress
            )
            scaler = StandardScaler()
            deep_scaled = scaler.fit_transform(deep_features)
            pca_2d = PCA(n_components=2).fit_transform(deep_scaled)

            self.progress.emit(15)
            self.status.emit(_t("waiting_user_selection", self.language))
            self._interactive_result = None

            self.request_interactive_labeling.emit(
                images_rgb, image_names, pca_2d, 
                self.classifier.n_clusters, self.classifier.class_names
            )

            # Espera thread-safe pela resposta do labeling (UI thread chama set_labeling_result)
            with QMutexLocker(self.mutex):
                while self._interactive_result is None and self.running:
                    self.condition.wait(self.mutex, 100)  # 100ms timeout para checar self.running

            if not self.running:
                self.finished_signal.emit(False, None, _t("cancelled_by_user", self.language))
                return

            if self._interactive_result is None:
                self.finished_signal.emit(False, None, _t("timeout_waiting_selection", self.language))
                return

            self.classifier.labeler.labels = self._interactive_result

            # REVISED: Expand with deep features
            self.status.emit(_t("expanding_labels_knn", self.language))
            self.progress.emit(25)
            self.classifier.labeler.expand_labels_nearest_neighbors(images_rgb, n_neighbors=50)

            self.status.emit(_t("running_full_classification", self.language))
            self.progress.emit(30)

            def on_progress(msg, pct):
                self.status.emit(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
                self.progress.emit(pct)

            def on_status(msg):
                self.status.emit(msg)

            supervised_preds, unsupervised_preds, confidences = self.classifier.classify_images(
                images_rgb, images_raw, image_names,
                folder_path=self.batch_folder,
                parent_widget=None,
                progress_callback=on_progress,
                status_callback=on_status,
                skip_labeling=True
            )

            results = {
                "labels": self.classifier.labeler.labels,
                "class_names": self.classifier.class_names,
                "supervised_preds": supervised_preds,
                "unsupervised_preds": unsupervised_preds,
                "confidences": confidences
            }
            self.progress.emit(100)
            self.finished_signal.emit(True, results, _t("workflow_complete", self.language))

        except Exception as e:
            import traceback
            self.finished_signal.emit(False, None, _t("error_detail", self.language, str(e), traceback.format_exc()))

    def _run_realtime(self):
        """Classifica frames da fila usando modelo já treinado."""
        if not self.classifier.has_trained_model():
            self.status.emit(_t("error_no_trained_model", self.language))
            self.finished_signal.emit(False, None, _t("model_not_loaded", self.language))
            return

        self.status.emit(_t("realtime_classification_started", self.language))
        self._paused = False
        processed_count = 0

        while self.running:
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

        self.status.emit(_t("realtime_classification_finished", self.language, processed_count))
        self.finished_signal.emit(True, None, _t("realtime_finished", self.language))


# =============================================================================
# DIALOGS
# =============================================================================

class SeafloorClassificationDialog(QDialog):
    def __init__(self, parent=None, language="pt"):
        super().__init__(parent)
        self.language = language
        self.setWindowTitle(_t("seafloor_classification_title", self.language))
        self.resize(500, 450)
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        info = QLabel(
            "<b>" + _t("classes_label", self.language) + "</b> " +
            _t("seafloor_classes_list", self.language) + "<br><br>" +
            "<b>" + _t("instructions", self.language) + "</b> " +
            _t("seafloor_instructions_text", self.language)
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        folder_layout = QHBoxLayout()
        self.folder_label = QLabel(_t("no_folder_selected", self.language))
        self.folder_label.setStyleSheet("color: gray; font-style: italic;")
        browse_btn = QPushButton(_t("select_folder", self.language))
        browse_btn.clicked.connect(self._select_folder)
        folder_layout.addWidget(self.folder_label, 1)
        folder_layout.addWidget(browse_btn)
        layout.addLayout(folder_layout)

        # Nome do experimento
        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel(_t("experiment", self.language)))
        self.exp_name_input = QLineEdit("exp01_baseline")
        self.exp_name_input.setPlaceholderText(_t("experiment_placeholder", self.language))
        exp_layout.addWidget(self.exp_name_input, 1)
        layout.addLayout(exp_layout)

        settings_group = QGroupBox(_t("settings", self.language))
        settings_layout = QFormLayout(settings_group)

        self.examples_spin = QSpinBox()
        self.examples_spin.setRange(5, 50)
        self.examples_spin.setValue(10)
        settings_layout.addRow(_t("examples_per_class", self.language), self.examples_spin)

        self.crop_spin = QDoubleSpinBox()
        self.crop_spin.setRange(0.3, 1.0)
        self.crop_spin.setSingleStep(0.05)
        self.crop_spin.setValue(0.85)
        settings_layout.addRow(_t("crop_ratio_label", self.language), self.crop_spin)

        self.normalize_check = QPushButton(_t("color_normalization_on", self.language))
        self.normalize_check.setCheckable(True)
        self.normalize_check.setChecked(True)
        self.normalize_check.clicked.connect(self._toggle_normalize)
        settings_layout.addRow(_t("normalization_label", self.language), self.normalize_check)

        self.ensemble_check = QPushButton(_t("adaptive_ensemble_off", self.language))
        self.ensemble_check.setCheckable(True)
        self.ensemble_check.setChecked(False)
        self.ensemble_check.clicked.connect(self._toggle_ensemble)
        settings_layout.addRow(_t("ensemble_label", self.language), self.ensemble_check)

        layout.addWidget(settings_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Log ao vivo
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        self.log_text.setPlaceholderText(_t("log_placeholder", self.language))
        self.log_text.setVisible(False)
        layout.addWidget(self.log_text)

        self.status_text = QLabel("")
        self.status_text.setVisible(False)
        layout.addWidget(self.status_text)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton(_t("start_classification", self.language))
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start_classification)
        self.cancel_btn = QPushButton(_t("cancel", self.language))
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

        self.normalize_enabled = True
        self.ensemble_enabled = False
        self.folder_path = None
        self.worker = None

    def _toggle_normalize(self):
        self.normalize_enabled = not self.normalize_enabled
        self.normalize_check.setText(
            _t("color_normalization_on" if self.normalize_enabled else "color_normalization_off", self.language)
        )

    def _toggle_ensemble(self):
        self.ensemble_enabled = not self.ensemble_enabled
        self.ensemble_check.setText(
            _t("adaptive_ensemble_on" if self.ensemble_enabled else "adaptive_ensemble_off", self.language)
        )

    def _select_folder(self):
        path = QFileDialog.getExistingDirectory(self, _t("select_image_folder", self.language))
        if path:
            self.folder_path = path
            self.folder_label.setText(Path(path).name)
            self.folder_label.setStyleSheet("color: black; font-style: normal;")
            self.start_btn.setEnabled(True)

    def _start_classification(self):
        if not self.folder_path:
            return

        # Criar classificador com pasta do experimento
        exp_name = self.exp_name_input.text().strip() or "exp01_baseline"
        exp_name = exp_name.replace(" ", "_").replace("/", "_")
        output_dir = os.path.join(self.folder_path, "seafloor_output", exp_name)

        self.classifier = SeafloorClassifier(
            n_clusters=3,
            normalize_colors=self.normalize_enabled,
            n_examples_per_class=self.examples_spin.value(),
            use_augmentation=True,
            fast_mode=True,
            output_dir=output_dir,
            crop_to_299=True,
            crop_ratio=self.crop_spin.value(),
            use_adaptive_ensemble=self.ensemble_enabled
        )

        self.log_text.append(_t("info_prefix", self.language) + " " + _t("experiment", self.language) + f": {exp_name}")
        self.log_text.append(_t("info_prefix", self.language) + " " + _t("output_dir_label", self.language) + f": {output_dir}")

        # Configurar UI para modo "rodando"
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.log_text.setVisible(True)
        self.log_text.clear()
        self.start_btn.setEnabled(False)
        self.start_btn.setText(_t("processing", self.language))
        self.cancel_btn.setText(_t("cancel", self.language))

        # Criar e configurar worker thread
        self.worker = SeafloorClassificationThread(self.classifier, parent=self)
        self.worker.set_batch_mode(self.folder_path)
        self.worker.progress.connect(self._on_progress)
        self.worker.status.connect(self._on_status)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.request_interactive_labeling.connect(self._on_request_labeling)
        self.worker.start()

    def _on_progress(self, value):
        self.progress_bar.setValue(value)

    def _on_status(self, msg):
        self.log_text.append(msg)
        # Auto-scroll para o final
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_request_labeling(self, images_rgb, image_names, pca_2d, n_clusters, class_names):
        """Mostra o dialog de labeling interativo na thread principal.

        IMPORTANTE: este slot roda na thread principal (UI). O dialog é modal
        e bloqueante. Quando o usuário termina, chamamos set_labeling_result()
        diretamente no worker — isso é thread-safe e acorda o worker imediatamente.
        """
        labels = self.classifier.labeler.interactive_labeling_qt(
            images_rgb, image_names, pca_2d,
            n_clusters, class_names,
            parent_widget=self
        )
        if not labels or len(labels) == 0:
            QMessageBox.warning(self, _t("warning", self.language),
                _t("no_examples_selected_auto", self.language))
        # Chamada direta thread-safe — NÃO usar sinal/slot cross-thread
        self.worker.set_labeling_result(labels)

    def _on_finished(self, success, results, message):
        self.start_btn.setEnabled(True)
        self.start_btn.setText(_t("start_classification", self.language))
        self.progress_bar.setVisible(False)

        # Salvar modelo explicitamente (garantia caso o pipeline tenha falhado)
        if success and hasattr(self, 'classifier') and self.classifier.has_trained_model():
            try:
                model_path = self.classifier.save_model()
                config_path = self.classifier.save_config()
                self.log_text.append(_t("saved_prefix", self.language) + " " + _t("model_saved_log", self.language, model_path))
                self.log_text.append(_t("saved_prefix", self.language) + " " + _t("config_saved_log", self.language, config_path))
            except Exception as e:
                self.log_text.append(_t("warning_prefix", self.language) + " " + _t("error_saving_model", self.language, e))

        if success and results:
            supervised_preds = results.get("supervised_preds", [])
            confidences = results.get("confidences", [])
            class_names = results.get("class_names", [])

            if len(supervised_preds) > 0:
                dist = Counter(supervised_preds)
                msg = _t("classification_complete_msg", self.language, "")
                for cls_id, count in dist.items():
                    if cls_id < len(class_names):
                        name = _tc(class_names[cls_id], self.language)
                        msg += f"  {name}: {count}\n"
                if len(confidences) > 0:
                    msg += "\n" + _t("avg_confidence", self.language, np.mean(confidences))
                QMessageBox.information(self, _t("success", self.language), msg)
            else:
                QMessageBox.information(self, _t("success", self.language), message)
        else:
            QMessageBox.critical(self, _t("error", self.language), message)

    def reject(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
        super().reject()


class SeafloorClassManager(QDialog):
    """Diálogo para gerenciar categorias do classificador de fundo."""

    classes_changed = pyqtSignal()

    def __init__(self, classifier, parent=None, language=None):
        super().__init__(parent)
        self.classifier = classifier
        self.language = language or getattr(classifier, "language", "pt")
        self.setWindowTitle(_t("manage_seafloor_categories", self.language))
        self.resize(450, 550)
        self._build_ui()
        self._refresh_list()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        info = QLabel(
            "<b>" + _t("fixed_classes_label", self.language) + "</b> " +
            _t("fixed_classes_desc", self.language) + "<br>" +
            "<b>" + _t("custom_classes_label", self.language) + "</b> " +
            _t("custom_classes_desc", self.language) + "<br>" +
            _t("max_custom_classes_hint", self.language)
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        layout.addWidget(QLabel("<b>" + _t("categories_label", self.language) + "</b>"))
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

        add_group = QGroupBox(_t("add_new_category", self.language))
        add_layout = QFormLayout(add_group)

        self.new_name = QLineEdit()
        self.new_name.setPlaceholderText(_t("example_placeholder", self.language))
        add_layout.addRow(_t("name_label", self.language), self.new_name)

        color_layout = QHBoxLayout()
        self.color_preview = QLabel("    ")
        self.color_preview.setFixedSize(30, 30)
        self.color_preview.setStyleSheet("background-color: #808080; border: 1px solid #333;")
        self.selected_color = None

        color_btn = QPushButton(_t("choose_color", self.language))
        color_btn.clicked.connect(self._choose_color)
        color_layout.addWidget(self.color_preview)
        color_layout.addWidget(color_btn)
        color_layout.addStretch()
        add_layout.addRow(_t("color_label", self.language), color_layout)

        add_btn = QPushButton(_t("add", self.language))
        add_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        add_btn.clicked.connect(self._add_class)
        add_layout.addRow(add_btn)

        layout.addWidget(add_group)

        remove_btn = QPushButton(_t("remove_selected", self.language))
        remove_btn.setStyleSheet("background-color: #f44336; color: white;")
        remove_btn.clicked.connect(self._remove_class)
        layout.addWidget(remove_btn)

        layout.addStretch()
        btn_layout = QHBoxLayout()
        save_btn = QPushButton(_t("save_configuration", self.language))
        save_btn.clicked.connect(self._save_config)
        close_btn = QPushButton(_t("close", self.language))
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
            QMessageBox.warning(self, _t("error", self.language), _t("enter_category_name", self.language))
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
            QMessageBox.warning(self, _t("error", self.language), msg)

    def _remove_class(self):
        current = self.class_list.currentItem()
        if not current:
            return

        text = current.text()
        if "[FIXO]" in text:
            QMessageBox.warning(self, _t("error", self.language), _t("seafloor_class_fixed", self.language))
            return

        # Find the class in the list to get internal_name
        classes = self.classifier.list_all_classes()
        # Extract display name from text: "[S] Sediment (#8B4513) [FIXO]"
        import re
        match = re.search(r'\] (.+?) \(', text)
        if match:
            display_name = match.group(1)
            # Find internal_name by matching display name
            internal_name = None
            for cls in classes:
                if cls["name"] == display_name:
                    internal_name = cls.get("internal_name", cls["name"])
                    break
            if not internal_name:
                internal_name = display_name
            reply = QMessageBox.question(self, _t("confirm", self.language),
                                       _t("confirm_remove_category", self.language, display_name))
            if reply == QMessageBox.StandardButton.Yes:
                success, msg = self.classifier.remove_custom_class(internal_name)
                if success:
                    self._refresh_list()
                    self.classes_changed.emit()
                else:
                    QMessageBox.warning(self, _t("error", self.language), msg)

    def _refresh_list(self):
        self.class_list.clear()
        classes = self.classifier.list_all_classes()

        for cls in classes:
            shortcut = cls["shortcut"]
            name = cls["name"]
            color = cls["color"]
            type_ = cls["type"]

            item_text = f"[{shortcut}] {name} ({color})"
            if type_ == "fixed":
                item_text += " [" + _t("fixed_marker", self.language) + "]"

            item = QListWidgetItem(item_text)

            pixmap = QPixmap(16, 16)
            pixmap.fill(QColor(color))
            item.setIcon(QIcon(pixmap))

            self.class_list.addItem(item)

    def _save_config(self):
        path = self.classifier.save_config()
        QMessageBox.information(self, _t("success", self.language), _t("config_saved_to", self.language, path))
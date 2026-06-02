"""
Seafloor Classification Module for iSEA
Integrates AI-SCW workflow into the iSEA video annotation platform
Classes: Sedimento, Coral_Fragmento, Coral_Vivo (Recife_de_Coral)
"""

import cv2
import os
import shutil
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
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Qt integration
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition, QMutexLocker
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QProgressBar, QTextEdit, QMessageBox,
                             QComboBox, QSpinBox, QDoubleSpinBox, QFormLayout,
                             QDialogButtonBox, QFileDialog, QGroupBox)


# =============================================================================
# COLOR NORMALIZATION (Reinhard LAB matching)
# =============================================================================

class ColorNormalizer:
    def __init__(self, reference_path=None, auto_select=True):
        self.reference_path = reference_path
        self.auto_select = auto_select
        self.ref_mean = None
        self.ref_std = None

    def select_reference(self, image_paths, folder_path):
        if self.reference_path and os.path.exists(self.reference_path):
            print(f"Using provided reference: {self.reference_path}")
            return cv2.imread(self.reference_path)

        print("\nSelecting reference image automatically...")
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
        print(f"Reference selected: {ref_name} (entropy={entropies[median_idx]:.2f})")
        return cv2.imread(os.path.join(folder_path, ref_name))

    def compute_reference_stats(self, ref_img):
        ref_lab = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        self.ref_mean = np.mean(ref_lab, axis=(0, 1))
        self.ref_std = np.std(ref_lab, axis=(0, 1))
        print(f"Reference stats (LAB): mean={self.ref_mean}, std={self.ref_std}")
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
            if (i + 1) % 50 == 0:
                print(f"  Normalized {i+1}/{len(images)} images")
        return normalized


# =============================================================================
# DATA AUGMENTATION
# =============================================================================

def augment_image(img):
    """Light augmentations for class balancing."""
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
# SEMI-AUTOMATIC LABELER (Non-interactive for iSEA integration)
# =============================================================================

class SemiAutoLabeler:
    def __init__(self, n_examples_per_class=10):
        self.n_examples = n_examples_per_class
        self.labels = {}
        self.class_names = {}

    def extract_domain_features(self, images):
        features = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            global_entropy = shannon_entropy(gray)

            h, w = gray.shape
            block_h, block_w = max(h // 3, 1), max(w // 3, 1)
            local_entropies = []
            for i in range(3):
                for j in range(3):
                    y1, y2 = i * block_h, min((i + 1) * block_h, h)
                    x1, x2 = j * block_w, min((j + 1) * block_w, w)
                    block = gray[y1:y2, x1:x2]
                    if block.size > 0:
                        local_entropies.append(shannon_entropy(block))

            glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, 
                               symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]

            feat = np.concatenate([
                [global_entropy], local_entropies,
                [contrast, dissimilarity, homogeneity, energy, correlation]
            ])
            features.append(feat)
        return np.array(features)

    def non_interactive_labeling(self, images, image_names, pca_2d, n_classes, 
                                class_names_list=None):
        """Non-interactive labeling for batch processing in iSEA."""
        print("\n" + "="*50)
        print("SEMI-AUTOMATIC LABELING (Non-interactive)")
        print("="*50)

        if class_names_list is None:
            class_names_list = [f"Class_{i}" for i in range(n_classes)]

        for cls in range(n_classes):
            cls_name = class_names_list[cls] if cls < len(class_names_list) else f"Class_{cls}"
            self.class_names[cls] = cls_name

            assigned = []
            sector_size = len(pca_2d) // self.n_examples
            for i in range(self.n_examples):
                start_idx = i * sector_size
                end_idx = (i + 1) * sector_size if i < self.n_examples - 1 else len(pca_2d)
                sector_indices = list(range(start_idx, end_idx))
                sector_center = np.mean(pca_2d[sector_indices], axis=0)
                distances = np.linalg.norm(pca_2d[sector_indices] - sector_center, axis=1)
                best_idx = sector_indices[np.argmin(distances)]
                assigned.append(best_idx)

            for idx in assigned:
                self.labels[idx] = cls
            print(f"  '{cls_name}': {len(assigned)} examples")

        return self.labels

    def expand_labels_nearest_neighbors(self, images, n_neighbors=50):
        print(f"\nExpanding labels (k={n_neighbors})...")
        features = self.extract_domain_features(images)
        features_scaled = StandardScaler().fit_transform(features)
        features_pca = PCA(n_components=min(50, features_scaled.shape[1])).fit_transform(features_scaled)

        labeled_indices = list(self.labels.keys())
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
        n_manual = len(labeled_indices)
        n_auto = len(new_labels) - n_manual
        print(f" Labels: {n_manual} manual + {n_auto} auto = {len(new_labels)} total")

        dist = Counter(self.labels.values())
        print(f" Distribution: {dict(dist)}")
        return new_labels

    def get_training_data_balanced(self, images, image_names, augment_factor=3):
        """Returns balanced training data with augmentation."""
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

        print(f" Balanced: {Counter(y_balanced)}")
        return np.array(X_balanced), np.array(y_balanced), names_train


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
        self.model.fc = torch.nn.Linear(num_ftrs, n_classes)
        self.model = self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.scaler = GradScaler() if torch.cuda.is_available() else None

    def train(self, X_train, y_train, epochs=30, batch_size=32, val_split=0.2,
              patience=5, num_workers=0, class_weights=None):
        print(f"\nTraining InceptionV3...")
        print(f" Device: {self.device}")
        print(f" Samples: {len(X_train)} | Epochs: {epochs} | Batch: {batch_size}")
        print(f" Backbone frozen: {self.freeze_backbone}")

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

        criterion = torch.nn.CrossEntropyLoss(weight=weights)

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

            print(f" Epoch {epoch+1}/{epochs} | "
                  f"Loss: {train_loss/train_total:.4f} | "
                  f"Train: {train_acc:.3f} | Val: {val_acc:.3f}")

            scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), "best_model_fast.pth")
                print(f"  -> New best! Saved.")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        print(f"\nBest validation accuracy: {best_val_acc:.3f}")
        self.model.load_state_dict(torch.load("best_model_fast.pth", weights_only=True))

    def predict(self, images, batch_size=32, num_workers=0):
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

    def extract_features(self, images, batch_size=32, num_workers=0):
        """Extract features from penultimate layer (before FC)."""
        self.model.eval()

        original_fc = self.model.fc
        self.model.fc = torch.nn.Identity()

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
# MAIN SEAFLOOR CLASSIFICATION CLASS
# =============================================================================

class SeafloorClassifier:
    """
    Seafloor classification workflow for iSEA.
    Classes: Sedimento, Coral_Fragmento, Coral_Vivo
    """
    CLASS_NAMES = ["Sedimento", "Coral_Fragmento", "Coral_Vivo"]
    COLORS = {
        "Sedimento": "#8B4513",      # Brown
        "Coral_Fragmento": "#FF8C00", # Dark Orange
        "Coral_Vivo": "#00CED1"       # Dark Turquoise
    }

    def __init__(self, n_clusters=3, max_examples=None,
                 normalize_colors=True, reference_path=None,
                 n_examples_per_class=10, use_augmentation=True,
                 fast_mode=True, output_dir="seafloor_output"):

        self.n_clusters = n_clusters
        self.max_examples = max_examples
        self.normalize_colors = normalize_colors
        self.n_examples_per_class = n_examples_per_class
        self.use_augmentation = use_augmentation
        self.fast_mode = fast_mode
        self.output_dir = Path(output_dir)

        self.normalizer = None
        if self.normalize_colors:
            self.normalizer = ColorNormalizer(reference_path=reference_path)
            print("\nColor normalization: ENABLED")
        else:
            print("\nColor normalization: DISABLED")

        self.labeler = SemiAutoLabeler(n_examples_per_class=n_examples_per_class)
        self.trained_classifier = None
        self.class_names = self.CLASS_NAMES

        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        for sub in ["clusters", "normalized", "supervised", "pca_plots"]:
            (self.output_dir / sub).mkdir(exist_ok=True)
        for i in range(n_clusters):
            (self.output_dir / "clusters" / f"cluster{i}").mkdir(exist_ok=True)
            (self.output_dir / "supervised" / f"class{i}").mkdir(exist_ok=True)

    def load_images_from_folder(self, folder_path):
        """Load images from a folder."""
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        paths = [f for f in os.listdir(folder_path)
                 if Path(f).suffix.lower() in valid_extensions]

        if len(paths) == 0:
            raise ValueError(f"No images found in '{folder_path}'")

        if self.max_examples:
            paths = paths[:self.max_examples]

        print(f"\nLoading {len(paths)} images from {folder_path}...")
        images_raw = []
        images_rgb = []
        image_names = []

        for name in paths:
            img = cv2.imread(os.path.join(folder_path, name))
            if img is None:
                print(f"  Skip: {name}")
                continue
            images_raw.append(img.copy())
            images_rgb.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            image_names.append(name)

        print(f" {len(images_raw)} loaded")
        return images_raw, images_rgb, image_names

    def classify_images(self, images_rgb, images_raw, image_names, folder_path=None):
        """Run full classification workflow."""
        print("\n" + "="*60)
        print("SEAFLOOR CLASSIFICATION WORKFLOW")
        print("="*60)

        # Step 1: Domain features + PCA
        print("\n--- Step 1: Domain features + PCA ---")
        features = self.labeler.extract_domain_features(images_rgb)
        features_scaled = StandardScaler().fit_transform(features)
        pca_2d = PCA(n_components=2).fit_transform(features_scaled)

        # Step 2: Semi-automatic labeling (non-interactive for iSEA)
        print("\n--- Step 2: Semi-automatic labeling ---")
        self.labeler.non_interactive_labeling(
            images_rgb, image_names, pca_2d,
            self.n_clusters, self.class_names
        )
        self.labeler.expand_labels_nearest_neighbors(images_rgb, n_neighbors=50)

        # Step 3: Color normalization
        print("\n--- Step 3: Color normalization ---")
        if self.normalize_colors and self.normalizer and folder_path:
            ref_img = self.normalizer.select_reference(image_names, folder_path)
            self.normalizer.compute_reference_stats(ref_img)

            print("Normalizing all images...")
            normalized_bgr = self.normalizer.normalize_dataset(
                images_raw, image_names,
                output_dir=str(self.output_dir / "normalized")
            )
            images_normalized_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                                      for img in normalized_bgr]
        else:
            images_normalized_rgb = images_rgb

        # Step 4: Supervised training
        print("\n--- Step 4: Supervised training ---")
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
            # Phase 1: Transfer learning
            print("\n  Phase 1: Transfer learning...")
            classifier = FastInceptionV3Classifier(
                n_classes=n_classes, freeze_backbone=True
            )
            classifier.train(X_train, y_train, epochs=10, batch_size=32, patience=3)

            # Phase 2: Fine-tuning
            print("\n  Phase 2: Fine-tuning...")
            classifier = FastInceptionV3Classifier(
                n_classes=n_classes, freeze_backbone=False
            )
            classifier.model.load_state_dict(
                torch.load("best_model_fast.pth", weights_only=True)
            )
            classifier.train(X_train, y_train, epochs=30, batch_size=32, patience=7)
            self.trained_classifier = classifier
        else:
            classifier = FastInceptionV3Classifier(
                n_classes=n_classes, freeze_backbone=False
            )
            classifier.train(X_train, y_train, epochs=20, batch_size=32, patience=5)
            self.trained_classifier = classifier

        # Predictions
        supervised_preds, confidences = classifier.predict(
            images_normalized_rgb, batch_size=32
        )

        # Step 5: Unsupervised clustering
        print("\n--- Step 5: Unsupervised clustering ---")
        unsupervised_preds = self._run_unsupervised(images_normalized_rgb)

        # Step 6: Comparison
        print("\n--- Step 6: Comparison ---")
        self._compare_methods(supervised_preds, unsupervised_preds, confidences)

        # Save supervised results
        for i, name in enumerate(image_names):
            dst = self.output_dir / "supervised" / f"class{supervised_preds[i]}" / name
            cv2.imwrite(str(dst), cv2.cvtColor(images_normalized_rgb[i], cv2.COLOR_RGB2BGR))

        # Step 7: Supervised PCA visualization
        print("\n--- Step 7: Supervised PCA visualization ---")
        self._visualize_pca_supervised(images_normalized_rgb, supervised_preds, confidences)

        return supervised_preds, unsupervised_preds, confidences

    def _run_unsupervised(self, images_normalized_rgb):
        """Run unsupervised k-means clustering."""
        print(f"\nK-means (k={self.n_clusters})...")

        # Extract CNN features
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        model.fc = torch.nn.Identity()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        images_resized = np.array([cv2.resize(img, (299, 299)) for img in images_normalized_rgb])

        cnn_features = []
        with torch.no_grad():
            for i in range(0, len(images_resized), 32):
                batch = images_resized[i:i + 32]
                tensors = torch.stack([preprocess(img) for img in batch]).to(device)
                cnn_features.extend(model(tensors).cpu().numpy())

        cnn_features = np.array(cnn_features)

        # Add entropy features
        entropy_features = self.labeler.extract_domain_features(images_normalized_rgb)
        scaler_e = StandardScaler()
        entropy_scaled = scaler_e.fit_transform(entropy_features)
        scaler_c = StandardScaler()
        cnn_scaled = scaler_c.fit_transform(cnn_features)

        combined = np.hstack([cnn_scaled, entropy_scaled])

        # PCA
        n_components = min(512, combined.shape[0], combined.shape[1])
        images_new = PCA(n_components=n_components).fit_transform(
            StandardScaler().fit_transform(combined)
        )

        # K-means
        predictions = KMeans(n_clusters=self.n_clusters, random_state=728, n_init=10).fit_predict(images_new)

        # Save cluster images
        # (Implementation similar to original...)

        return predictions

    def _compare_methods(self, supervised, unsupervised, confidences):
        """Compare supervised and unsupervised results."""
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
        """Create PCA visualization with trained features."""
        if self.trained_classifier is None:
            return

        print("\nExtracting features from trained classifier...")
        trained_features = self.trained_classifier.extract_features(images, batch_size=32)

        scaled = StandardScaler().fit_transform(trained_features)
        pca = PCA(n_components=2)
        pca_2d = pca.fit_transform(scaled)
        var = pca.explained_variance_ratio_

        print(f"  Variance: PC1={var[0]:.1%}, PC2={var[1]:.1%}, Total={var.sum():.1%}")

        colors_list = ['#2ecc71', '#9b59b6', '#e74c3c']

        fig, ax = plt.subplots(figsize=(16, 12))

        # Plot thumbnails
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

        # Scatter points by class
        for i in range(self.n_clusters):
            mask = predictions == i
            n_points = np.sum(mask)
            if n_points > 0:
                cls_name = self.class_names[i] if i < len(self.class_names) else f"Class_{i}"
                ax.scatter(pca_2d[mask, 0], pca_2d[mask, 1],
                        c=colors_list[i % len(colors_list)],
                        s=80, alpha=0.6, edgecolors='black', linewidth=0.8,
                        zorder=10, label=f'{cls_name} ({n_points})')

        # Confidence info
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
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        out = self.output_dir / "pca_plots" / "pca_supervised_trained.png"
        plt.savefig(out, dpi=200, bbox_inches='tight')
        print(f"Saved: {out}")
        plt.close()

    def predict_single_frame(self, frame_bgr):
        """Predict class for a single frame (for real-time use in iSEA)."""
        if self.trained_classifier is None:
            raise ValueError("Classifier not trained yet!")

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Normalize if enabled
        if self.normalize_colors and self.normalizer and self.normalizer.ref_mean is not None:
            # Need to normalize as BGR then convert back
            frame_norm_bgr = self.normalizer.normalize(frame_bgr)
            frame_rgb = cv2.cvtColor(frame_norm_bgr, cv2.COLOR_BGR2RGB)

        # Predict
        preds, confs = self.trained_classifier.predict([frame_rgb], batch_size=1)
        class_id = int(preds[0])
        confidence = float(confs[0])
        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class_{class_id}"

        return {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "color": self.COLORS.get(class_name, "#FFFFFF")
        }


# =============================================================================
# QTHREAD FOR BACKGROUND CLASSIFICATION (iSEA integration)
# =============================================================================

class SeafloorClassificationThread(QThread):
    """Background thread for seafloor classification in iSEA."""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, object, str)  # success, results, message
    frame_classified = pyqtSignal(dict, int)  # result, frame_num

    def __init__(self, classifier: SeafloorClassifier, parent=None):
        super().__init__(parent)
        self.classifier = classifier
        self.frames = []  # List of (frame_bgr, frame_num)
        self.mode = "batch"  # "batch" or "realtime"
        self.running = True
        self.mutex = QMutex()
        self.condition = QWaitCondition()

    def add_frame(self, frame_bgr, frame_num):
        """Add frame for real-time classification."""
        with QMutexLocker(self.mutex):
            self.frames.append((frame_bgr.copy(), frame_num))
            self.condition.wakeOne()

    def set_batch_mode(self, folder_path):
        """Set batch mode with folder of images."""
        self.mode = "batch"
        self.batch_folder = folder_path

    def stop(self):
        with QMutexLocker(self.mutex):
            self.running = False
            self.condition.wakeOne()

    def run(self):
        if self.mode == "batch":
            self._run_batch()
        else:
            self._run_realtime()

    def _run_batch(self):
        try:
            self.status.emit("Loading images...")
            images_raw, images_rgb, image_names = self.classifier.load_images_from_folder(
                self.batch_folder
            )

            self.status.emit("Running classification workflow...")
            supervised, unsupervised, confidences = self.classifier.classify_images(
                images_rgb, images_raw, image_names, self.batch_folder
            )

            # Prepare results
            results = {
                "supervised": supervised,
                "unsupervised": unsupervised,
                "confidences": confidences,
                "image_names": image_names,
                "class_names": self.classifier.class_names
            }

            self.finished_signal.emit(True, results, "Classification complete!")

        except Exception as e:
            import traceback
            self.finished_signal.emit(False, None, f"Error: {str(e)}\n{traceback.format_exc()}")

    def _run_realtime(self):
        """Real-time classification of video frames."""
        while True:
            with QMutexLocker(self.mutex):
                while not self.frames and self.running:
                    self.condition.wait(self.mutex)

                if not self.running:
                    break

                if not self.frames:
                    continue

                frame_bgr, frame_num = self.frames.pop(0)

            try:
                result = self.classifier.predict_single_frame(frame_bgr)
                result["frame_num"] = frame_num
                self.frame_classified.emit(result, frame_num)
            except Exception as e:
                self.status.emit(f"Frame {frame_num} error: {str(e)}")


# =============================================================================
# DIALOG FOR CLASSIFICATION SETTINGS (iSEA UI)
# =============================================================================

class SeafloorClassificationDialog(QDialog):
    """Dialog for configuring and running seafloor classification in iSEA."""

    def __init__(self, parent=None, language="pt"):
        super().__init__(parent)
        self.language = language
        self.setWindowTitle("Classificação de Fundo Marinho (AI-SCW)")
        self.resize(500, 400)
        self.worker = None

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Info
        info = QLabel(
            "Classificação de fundo marinho usando AI-SCW\n"
            "Classes: Sedimento, Coral Fragmento, Coral Vivo\n\n"
            "Selecione uma pasta com imagens de fundo para classificar."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Folder selection
        folder_layout = QHBoxLayout()
        self.folder_label = QLabel("(nenhuma pasta selecionada)")
        self.folder_label.setStyleSheet("color: gray; font-style: italic;")
        browse_btn = QPushButton("Selecionar Pasta...")
        browse_btn.clicked.connect(self._select_folder)
        folder_layout.addWidget(self.folder_label, 1)
        folder_layout.addWidget(browse_btn)
        layout.addLayout(folder_layout)

        # Settings group
        settings_group = QGroupBox("Configurações")
        settings_layout = QFormLayout(settings_group)

        self.examples_spin = QSpinBox()
        self.examples_spin.setRange(5, 50)
        self.examples_spin.setValue(10)
        self.examples_spin.setToolTip("Exemplos por classe para labeling semi-automático")
        settings_layout.addRow("Exemplos/classe:", self.examples_spin)

        self.normalize_check = QPushButton("Normalização de cor: ATIVADA")
        self.normalize_check.setCheckable(True)
        self.normalize_check.setChecked(True)
        self.normalize_check.clicked.connect(self._toggle_normalize)
        settings_layout.addRow("Normalização:", self.normalize_check)

        self.ref_path_label = QLabel("(auto-seleção)")
        self.ref_path_label.setStyleSheet("color: gray; font-size: 10px;")
        ref_btn = QPushButton("Ref. manual...")
        ref_btn.clicked.connect(self._select_reference)
        ref_layout = QHBoxLayout()
        ref_layout.addWidget(self.ref_path_label, 1)
        ref_layout.addWidget(ref_btn)
        settings_layout.addRow("Imagem referência:", ref_layout)

        layout.addWidget(settings_group)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_text = QLabel("")
        self.status_text.setVisible(False)
        layout.addWidget(self.status_text)

        self.log_area = QTextEdit()
        self.log_area.setVisible(False)
        self.log_area.setMaximumHeight(100)
        layout.addWidget(self.log_area)

        # Buttons
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
        self.reference_path = None
        self.folder_path = None

    def _toggle_normalize(self):
        self.normalize_enabled = not self.normalize_enabled
        if self.normalize_enabled:
            self.normalize_check.setText("Normalização de cor: ATIVADA")
        else:
            self.normalize_check.setText("Normalização de cor: DESATIVADA")

    def _select_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Selecionar pasta de imagens")
        if path:
            self.folder_path = path
            self.folder_label.setText(Path(path).name)
            self.folder_label.setStyleSheet("color: black; font-style: normal;")
            self.start_btn.setEnabled(True)

    def _select_reference(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Selecionar imagem de referência", "",
            "Imagens (*.jpg *.jpeg *.png)"
        )
        if path:
            self.reference_path = path
            self.ref_path_label.setText(Path(path).name)

    def _start_classification(self):
        if not self.folder_path:
            return

        # Create classifier
        classifier = SeafloorClassifier(
            n_clusters=3,
            normalize_colors=self.normalize_enabled,
            reference_path=self.reference_path,
            n_examples_per_class=self.examples_spin.value(),
            use_augmentation=True,
            fast_mode=True,
            output_dir=os.path.join(self.folder_path, "seafloor_output")
        )

        # Setup worker
        self.worker = SeafloorClassificationThread(classifier)
        self.worker.set_batch_mode(self.folder_path)

        # UI updates
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.status_text.setVisible(True)
        self.log_area.setVisible(True)
        self.start_btn.setEnabled(False)

        # Connect signals
        self.worker.status.connect(self._update_status)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.start()

    def _update_status(self, message):
        self.status_text.setText(message)
        self.log_area.append(message)

    def _on_finished(self, success, results, message):
        self.progress_bar.setVisible(False)

        if success:
            # Show summary
            summary = self._format_results(results)
            QMessageBox.information(self, "Sucesso", summary)
            self.accept()
        else:
            QMessageBox.critical(self, "Erro", message)
            self.start_btn.setEnabled(True)

    def _format_results(self, results):
        supervised = results["supervised"]
        confidences = results["confidences"]
        class_names = results["class_names"]

        counts = Counter(supervised)
        lines = ["📊 RESULTADOS DA CLASSIFICAÇÃO\n", "=" * 40]

        for i, name in enumerate(class_names):
            count = counts.get(i, 0)
            pct = count / len(supervised) * 100
            avg_conf = np.mean([c for c, p in zip(confidences, supervised) if p == i]) if count > 0 else 0
            lines.append(f"{name}: {count} imagens ({pct:.1f}%) - conf média: {avg_conf:.2f}")

        lines.append(f"\nTotal: {len(supervised)} imagens")
        lines.append(f"\nOutput: seafloor_output/")
        return "\n".join(lines)

    def reject(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1000)
        super().reject()
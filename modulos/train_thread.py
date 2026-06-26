from PyQt6.QtCore import QThread, pyqtSignal
import os
import traceback
from ultralytics import YOLO
from pathlib import Path
from .hierarchical_metrics import HierarchicalValidator, IoUMatcher
from PyQt6.QtWidgets import QMessageBox
import pandas as pd
import numpy as np
import json
from collections import defaultdict, Counter
from datetime import datetime


class TrainThread(QThread):
    finished = pyqtSignal()
    epoch_progress = pyqtSignal(int)
    hier_metrics_ready = pyqtSignal(dict)  # Novo sinal para métricas hierárquicas por época

    def __init__(self, train_config):
        super().__init__()
        self.train_config = train_config
        self.success = False
        self.error = ""
        self.model_path = ""
        self._hierarchical_history = []  # Histórico de métricas por época

    def run(self):
        try:  
            model = YOLO("yolo26n.pt")

            def on_epoch_end(trainer):
                self.epoch_progress.emit(trainer.epoch + 1)

            model.add_callback("on_train_epoch_end", on_epoch_end)

            # Callback para métricas hierárquicas durante validação por época
            def on_val_end(trainer):
                """Executa avaliação hierárquica ao final de cada época de validação."""
                try:
                    self._run_hierarchical_evaluation_per_epoch(trainer)
                except Exception as e:
                    print(f"[Hierarchical] Per-epoch evaluation error: {e}")

            # Registra callback apenas se houver dados taxonômicos disponíveis
            data_path = self.train_config.get('data')
            if data_path and Path(data_path).exists():
                validator = HierarchicalValidator(
                    cache_file="worms_cache.json",
                    dataset_yaml=data_path
                )
                if validator.available:
                    model.add_callback("on_fit_epoch_end", on_val_end)
                    print("[Hierarchical] Per-epoch hierarchical evaluation enabled")

            aug_params = {
                "hsv_h": 0.05,        # matiz
                "hsv_s": 0.9,          # saturação
                "hsv_v": 0.6,          # valor
                "degrees": 15.0,        # rotação
                "translate": 0.2,      # translação
                "scale": 0.7,          # escala
                "shear": 5.0,          # cisalhamento
                "perspective": 0.001,    # perspectiva
                "flipud": 0.3,         # flip vertical
                "fliplr": 0.5,         # flip horizontal
                "mosaic": 1.0,         # mosaic
                "mixup": 0.1,          # mixup
                "copy_paste": 0.1,     # copy-paste
                "auto_augment": "randaugment",
                "erasing": 0.4,        # random erasing
            }

            # Mescla com train_config (train_config tem prioridade)
            final_config = {**aug_params, **self.train_config}

            model.train(**final_config)

            # Avaliação hierárquica final (pós-treino completo)
            self._run_hierarchical_evaluation(model, final_config)

            self.model_path = os.path.join(
                "models", "detect",
                final_config["name"],
                "weights", "best.pt"
            )
            self.success = True

        except Exception as e:
            self.success = False
            self.error = str(e)
            print(f"Training error: {traceback.format_exc()}")
        finally:
            self.finished.emit()

    def _run_hierarchical_evaluation_per_epoch(self, trainer):
        """
        Avaliação hierárquica rápida por época (amostra pequena para não travar treino).
        """
        try:
            # Usa o modelo atual do trainer (não recarrega)
            model = trainer.model
            if model is None:
                return

            data_path = self.train_config.get('data')
            if not data_path or not Path(data_path).exists():
                return

            validator = HierarchicalValidator(
                cache_file="worms_cache.json",
                dataset_yaml=data_path
            )

            if not validator.available:
                return

            # Amostra pequena (50 imgs) para não travar treino
            val_path = Path(data_path).parent if Path(data_path).is_file() else Path(data_path)
            val_images = self._find_val_images(val_path)

            if not val_images:
                return

            import random
            random.seed(42)
            sample = random.sample(val_images, min(50, len(val_images)))

            # Processa amostra
            metrics = self._evaluate_sample(model, sample, validator, is_epoch_eval=True)

            if metrics:
                metrics['epoch'] = trainer.epoch
                metrics['timestamp'] = datetime.now().isoformat()
                self._hierarchical_history.append(metrics)
                self.hier_metrics_ready.emit(metrics)

        except Exception as e:
            print(f"[Hierarchical] Epoch eval error: {e}")

    def _run_hierarchical_evaluation(self, model, config):
        """
        Avaliação hierárquica completa pós-treino.
        Processa TODO o val set (ou amostra estratificada de até 500 imgs).
        """
        try:
            data_path = config.get('data')
            if not data_path or not Path(data_path).exists():
                return

            validator = HierarchicalValidator(
                cache_file="worms_cache.json",
                dataset_yaml=data_path
            )

            if not validator.available:
                print("[Hierarchical] No taxonomy data available (cache or dataset)")
                return

            print("\n" + "="*60)
            print("HIERARCHICAL EVALUATION (FULL)")
            print("="*60)

            # Validação padrão do YOLO
            val_results = model.val(verbose=False)

            box_map50 = 0
            if hasattr(val_results, 'results_dict'):
                box_map50 = val_results.results_dict.get('metrics/mAP50(B)', 
                            val_results.results_dict.get('metrics/mAP50', 0))
                print(f"Standard mAP@0.5: {box_map50:.4f}")

            # === AVALIAÇÃO HIERÁRQUICA APRIMORADA ===
            val_path = Path(data_path).parent if Path(data_path).is_file() else Path(data_path)

            if val_path.exists():
                import random
                random.seed(42)

                # Encontra TODAS as imagens de validação
                val_images = self._find_val_images(val_path)

                if not val_images:
                    print("[Hierarchical] No validation images found")
                    return

                # Amostra estratificada: até 500 imagens, garantindo diversidade
                max_sample = 500
                if len(val_images) > max_sample:
                    # Estratificação simples: amostrar proporcionalmente
                    sample = self._stratified_sample(val_images, max_sample)
                else:
                    sample = val_images

                print(f"Evaluating {len(sample)} validation images (from {len(val_images)} total)...")

                # Processa amostra completa
                metrics = self._evaluate_sample(model, sample, validator, is_epoch_eval=False)

                if metrics:
                    h_map = metrics['h_mAP']
                    traditional_mAP = metrics['traditional_mAP']

                    print(f"\nh-mAP (hierarchical):   {h_map:.4f}")
                    print(f"mAP (traditional):      {traditional_mAP:.4f}")
                    print(f"Standard mAP@0.5:       {box_map50:.4f}")

                    # Estatísticas por tipo de erro/acerto
                    print("\nBreakdown:")
                    for match_type, count in metrics['breakdown'].items():
                        pct = count / metrics['total_samples'] * 100
                        print(f"  {match_type}: {count} ({pct:.1f}%)")

                    # Métricas por rank
                    if metrics.get('h_mAP_per_rank'):
                        print("\nh-mAP per rank:")
                        for rank, score in sorted(metrics['h_mAP_per_rank'].items(), 
                                                   key=lambda x: -x[1]):
                            print(f"  {rank}: {score:.4f}")

                    # Taxonomia coverage
                    if metrics.get('taxonomy_coverage'):
                        cov = metrics['taxonomy_coverage']
                        print(f"\nTaxonomy coverage: {cov['coverage_pct']:.1f}% "
                              f"({cov['with_hierarchy']}/{cov['total']} with hierarchy)")

                    # Salva resultados enriquecidos
                    self._save_hierarchical_results(config, metrics, box_map50, len(sample))

                    # Salva histórico completo
                    self._save_hierarchical_history(config)

                print("="*60)

        except Exception as e:
            print(f"[Hierarchical] Evaluation error: {e}")
            import traceback
            traceback.print_exc()

    def _find_val_images(self, val_path: Path) -> list:
        """Encontra todas as imagens de validação no diretório."""
        val_images = []

        # Padrões comuns de diretório YOLO
        patterns = [
            'images/val/*',
            'images/val/**/*',
            'val/images/*',
            'val/images/**/*',
            'val/*',
            'val/**/*',
        ]

        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']

        for pattern in patterns:
            for ext in extensions:
                found = list(val_path.rglob(f'{pattern}/{ext}'))
                val_images.extend(found)
                # Também tenta maiúsculas
                found_upper = list(val_path.rglob(f'{pattern}/{ext.upper()}'))
                val_images.extend(found_upper)

        # Remove duplicatas mantendo ordem
        seen = set()
        unique = []
        for p in val_images:
            if p not in seen:
                seen.add(p)
                unique.append(p)

        return unique

    def _stratified_sample(self, images: list, max_size: int) -> list:
        """
        Amostra estratificada garantindo diversidade de vídeos/diretórios.
        """
        import random

        # Agrupa por diretório pai (presumivelmente vídeo/fonte diferente)
        groups = defaultdict(list)
        for img in images:
            parent = img.parent.name
            groups[parent].append(img)

        # Calcula quantas imagens por grupo (proporcional)
        total = len(images)
        sample = []

        for group, imgs in groups.items():
            n = max(1, int(len(imgs) / total * max_size))
            sample.extend(random.sample(imgs, min(n, len(imgs))))

        # Se ainda não atingiu max_size, preenche aleatoriamente
        if len(sample) < max_size:
            remaining = [img for img in images if img not in sample]
            need = max_size - len(sample)
            sample.extend(random.sample(remaining, min(need, len(remaining))))

        return sample[:max_size]

    def _evaluate_sample(self, model, sample: list, validator: HierarchicalValidator, 
                         is_epoch_eval: bool = False) -> dict:
        """
        Avalia uma amostra de imagens com matching IoU-based.

        Args:
            model: Modelo YOLO
            sample: Lista de Path das imagens
            validator: HierarchicalValidator instanciado
            is_epoch_eval: Se True, avaliação rápida por época (menos verbosa)

        Returns:
            Dict com métricas hierárquicas
        """
        matcher = IoUMatcher(iou_threshold=0.5)

        all_scores = []
        all_ious = []
        total_with_hierarchy = 0
        total_without_hierarchy = 0

        # Batch predictions para eficiência
        batch_size = 8 if is_epoch_eval else 16

        for batch_start in range(0, len(sample), batch_size):
            batch = sample[batch_start:batch_start + batch_size]

            # Predições em batch
            results = model.predict(source=batch, verbose=False, conf=0.25)

            for img_path, r in zip(batch, results):
                # Extrai predições
                pred_boxes = []
                pred_classes = []
                pred_ids = []

                if r.boxes:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                        cls_id = int(box.cls)
                        cls_name = model.names[cls_id]
                        pred_boxes.append((x1, y1, x2, y2))
                        pred_classes.append(cls_name)
                        pred_ids.append(cls_id)

                # Ground truths
                gt_boxes, gt_classes, gt_ids = self._read_ground_truths_with_ids(
                    img_path, model.names
                )

                # Matching IoU-based
                matched_preds, matched_gts, matched_pred_ids, matched_gt_ids, matched_ious =                     matcher.match(pred_boxes, gt_boxes, pred_classes, gt_classes, 
                                  pred_ids, gt_ids)

                # Calcula scores hierárquicos para matches
                for pred, gt, pred_id, gt_id, iou in zip(
                    matched_preds, matched_gts, matched_pred_ids, matched_gt_ids, matched_ious
                ):
                    score, match_type = validator.calculate_score(pred, gt, pred_id, gt_id)

                    # Verifica se há hierarquia disponível
                    gt_hier = validator.get_hierarchy_by_id(gt_id, gt)
                    if len(gt_hier) > 1 or gt_hier[0].get('rank') != 'unknown':
                        total_with_hierarchy += 1
                    else:
                        total_without_hierarchy += 1

                    all_scores.append({
                        "pred": pred, "gt": gt,
                        "pred_id": pred_id, "gt_id": gt_id,
                        "score": score, "type": match_type,
                        "iou": iou,
                        "image": str(img_path.name)
                    })
                    all_ious.append(iou)

        if not all_scores:
            return None

        # Métricas agregadas
        h_map = sum(s["score"] for s in all_scores) / len(all_scores)
        exact = sum(1 for s in all_scores if s["type"] == "exact")
        mean_iou = sum(all_ious) / len(all_ious) if all_ious else 0

        # Breakdown por tipo
        type_counts = Counter(s["type"] for s in all_scores)

        # Breakdown por rank do GT
        rank_scores = defaultdict(list)
        for s in all_scores:
            gt_hier = validator.get_hierarchy_by_id(s["gt_id"], s["gt"])
            gt_rank = gt_hier[0].get("rank", "unknown") if gt_hier else "unknown"
            rank_scores[gt_rank].append(s["score"])

        h_mAP_per_rank = {
            rank: sum(scores_list) / len(scores_list) 
            for rank, scores_list in rank_scores.items() if scores_list
        }

        total_samples = len(all_scores)

        return {
            'h_mAP': h_map,
            'traditional_mAP': exact / total_samples,
            'mean_iou': mean_iou,
            'scores': all_scores if not is_epoch_eval else [],  # Não guarda detalhes em epoch eval
            'breakdown': dict(type_counts.most_common()),
            'h_mAP_per_rank': h_mAP_per_rank,
            'total_samples': total_samples,
            'taxonomy_coverage': {
                'with_hierarchy': total_with_hierarchy,
                'without_hierarchy': total_without_hierarchy,
                'total': total_with_hierarchy + total_without_hierarchy,
                'coverage_pct': (total_with_hierarchy / (total_with_hierarchy + total_without_hierarchy) * 100) 
                                if (total_with_hierarchy + total_without_hierarchy) > 0 else 0
            }
        }

    def _read_ground_truths_with_ids(self, img_path, class_names):
        """Lê ground truths com IDs de classe e bounding boxes."""
        gt_boxes = []
        gt_classes = []
        gt_ids = []

        # Possíveis caminhos para labels
        possible_paths = [
            img_path.parent.parent / 'labels' / 'val' / f"{img_path.stem}.txt",
            img_path.parent / 'labels' / f"{img_path.stem}.txt",
            Path(str(img_path.parent).replace('images', 'labels')) / f"{img_path.stem}.txt",
            # Padrão YOLO: labels/val/imagem.txt
            img_path.parent.parent.parent / 'labels' / 'val' / f"{img_path.stem}.txt",
            img_path.parent.parent.parent / 'labels' / img_path.parent.name / f"{img_path.stem}.txt",
        ]

        label_path = None
        for p in possible_paths:
            if p.exists():
                label_path = p
                break

        if label_path and label_path.exists():
            # Lê dimensões da imagem para converter normalized -> pixel
            import cv2
            img = cv2.imread(str(img_path))
            if img is not None:
                img_h, img_w = img.shape[:2]
            else:
                img_h, img_w = 1, 1

            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        # Converte normalized -> pixel
                        x1 = (x_center - width/2) * img_w
                        y1 = (y_center - height/2) * img_h
                        x2 = (x_center + width/2) * img_w
                        y2 = (y_center + height/2) * img_h

                        cls_name = class_names.get(cls_id, str(cls_id))
                        gt_ids.append(cls_id)
                        gt_classes.append(cls_name)
                        gt_boxes.append((x1, y1, x2, y2))

        return gt_boxes, gt_classes, gt_ids

    def _save_hierarchical_results(self, config, metrics, box_map50, sample_size):
        """Salva resultados da avaliação hierárquica em JSON."""
        result_file = Path(config.get("project", "runs")) / "hierarchical_eval.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)

        output = {
            'h_mAP': metrics['h_mAP'],
            'traditional_mAP': metrics['traditional_mAP'],
            'mean_iou': metrics.get('mean_iou', 0),
            'standard_mAP50': box_map50,
            'sample_size': sample_size,
            'total_evaluated': metrics['total_samples'],
            'breakdown': metrics['breakdown'],
            'h_mAP_per_rank': metrics.get('h_mAP_per_rank', {}),
            'taxonomy_coverage': metrics.get('taxonomy_coverage', {}),
            'timestamp': datetime.now().isoformat(),
            'config': {
                'data': config.get('data'),
                'epochs': config.get('epochs'),
                'imgsz': config.get('imgsz'),
                'model': 'yolo26n.pt'
            }
        }

        with open(result_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved hierarchical eval to: {result_file}")

    def _save_hierarchical_history(self, config):
        """Salva histórico completo de métricas hierárquicas por época."""
        if not self._hierarchical_history:
            return

        history_file = Path(config.get("project", "runs")) / "hierarchical_history.json"
        history_file.parent.mkdir(parents=True, exist_ok=True)

        with open(history_file, 'w') as f:
            json.dump({
                'history': self._hierarchical_history,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

        print(f"Saved hierarchical history ({len(self._hierarchical_history)} epochs) to: {history_file}")


class TrainSegmentationThread(QThread):
    finished = pyqtSignal()
    epoch_progress = pyqtSignal(int)

    def __init__(self, train_config):
        super().__init__()
        self.train_config = train_config
        self.success = False
        self.error = ""
        self.model_path = ""

    def run(self):
        try:
            model = YOLO("yolo26n-seg.pt")

            def on_epoch_end(trainer):
                self.epoch_progress.emit(trainer.epoch + 1)

            model.add_callback("on_train_epoch_end", on_epoch_end)

            # Mesmos defaults de augmentation
            aug_params = {
                "hsv_h": 0.05,        # matiz
                "hsv_s": 0.9,          # saturação
                "hsv_v": 0.6,          # valor
                "degrees": 15.0,        # rotação
                "translate": 0.2,      # translação
                "scale": 0.7,          # escala
                "shear": 5.0,          # cisalhamento
                "perspective": 0.001,  # perspectiva
                "flipud": 0.3,         # flip vertical
                "fliplr": 0.5,         # flip horizontal
                "mosaic": 1.0,         # mosaic
                "mixup": 0.1,          # mixup
                "copy_paste": 0.1,     # copy-paste
                "auto_augment": "randaugment",
                "erasing": 0.4,        # random erasing
            }

            final_config = {**aug_params, **self.train_config}
            final_config["task"] = "segment"

            model.train(**final_config)

            self.model_path = os.path.join(
                "models", "segment",
                final_config["name"],
                "weights", "best.pt"
            )
            self.success = True

        except Exception as e:
            self.success = False
            self.error = str(e)
            print(f"Segmentation training error: {traceback.format_exc()}")
        finally:
            self.finished.emit()
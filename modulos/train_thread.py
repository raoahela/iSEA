from PyQt6.QtCore import QThread, pyqtSignal
import os
import traceback
from ultralytics import YOLO
from pathlib import Path
from .hierarchical_metrics import HierarchicalValidator
from PyQt6.QtWidgets import QMessageBox
import pandas as pd

class TrainThread(QThread):
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
            model = YOLO("yolo26n.pt")

            def on_epoch_end(trainer):
                self.epoch_progress.emit(trainer.epoch + 1)

            model.add_callback("on_train_epoch_end", on_epoch_end)
            
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

    def _run_hierarchical_evaluation(self, model, config):
        try:
            # Detecta se há dataset.yaml no config
            data_path = config.get('data')
            if not data_path or not Path(data_path).exists():
                return
                
            # Inicializa validator com o dataset específico
            validator = HierarchicalValidator(
                cache_file="worms_cache.json",
                dataset_yaml=data_path
            )
            
            if not validator.available:
                print("[Hierarchical] No taxonomy data available (cache or dataset)")
                return
            
            print("\n" + "="*50)
            print("HIERARCHICAL EVALUATION")
            print("="*50)
            
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
                
                # Encontra imagens de validação
                val_images = list(val_path.rglob('images/val/*.jpg'))
                val_images += list(val_path.rglob('images/val/*.png'))
                if not val_images:
                    val_images = list(val_path.rglob('val/**/*.jpg'))
                    val_images += list(val_path.rglob('val/**/*.png'))
                
                if val_images:
                    sample = random.sample(val_images, min(100, len(val_images)))
                    print(f"Evaluating {len(sample)} validation images...")
                    
                    # Predições
                    results = model.predict(source=sample, verbose=False)
                    
                    all_pred_classes = []
                    all_gt_classes = []
                    all_pred_ids = []  # IDs numéricos
                    all_gt_ids = []    # IDs numéricos
                    
                    for img_path, r in zip(sample, results):
                        # Predições desta imagem
                        img_preds = []
                        img_pred_ids = []
                        for box in r.boxes:
                            cls_id = int(box.cls)
                            cls_name = model.names[cls_id]
                            img_preds.append(cls_name)
                            img_pred_ids.append(cls_id)
                        
                        # Ground truths desta imagem
                        img_gts, img_gt_ids = self._read_ground_truths_with_ids(img_path, model.names)
                        
                        # Matching
                        matched_preds, matched_gts, matched_pred_ids, matched_gt_ids = self._match_predictions(
                            img_preds, img_gts, img_pred_ids, img_gt_ids, img_path.stem
                        )
                        
                        all_pred_classes.extend(matched_preds)
                        all_gt_classes.extend(matched_gts)
                        all_pred_ids.extend(matched_pred_ids)
                        all_gt_ids.extend(matched_gt_ids)
                    
                    if all_pred_classes and all_gt_classes:
                        # Calcula métricas hierárquicas
                        scores = []
                        for pred, gt, pred_id, gt_id in zip(all_pred_classes, all_gt_classes, all_pred_ids, all_gt_ids):
                            score, match_type = validator.calculate_score(pred, gt, pred_id, gt_id)
                            scores.append({
                                "pred": pred, "gt": gt,
                                "pred_id": pred_id, "gt_id": gt_id,
                                "score": score, "type": match_type
                            })
                        
                        h_map = sum(s["score"] for s in scores) / len(scores)
                        exact = sum(1 for s in scores if s["type"] == "exact")
                        
                        print(f"\nh-mAP (hierarchical):   {h_map:.4f}")
                        print(f"mAP (traditional):      {exact / len(scores):.4f}")
                        print(f"Standard mAP@0.5:       {box_map50:.4f}")
                        
                        # Estatísticas por tipo de erro/acerto
                        from collections import Counter
                        type_counts = Counter(s["type"] for s in scores)
                        print("\nBreakdown:")
                        for match_type, count in type_counts.most_common():
                            pct = count / len(scores) * 100
                            print(f"  {match_type}: {count} ({pct:.1f}%)")
                        
                        # Salva resultados
                        self._save_hierarchical_results(config, {
                            'h-mAP': h_map,
                            'traditional_mAP': exact / len(scores),
                            'scores': scores
                        }, box_map50, len(sample))
                    
                    print("="*50)
                    
        except Exception as e:
            print(f"[Hierarchical] Evaluation error: {e}")
            import traceback
            traceback.print_exc()

    def _read_ground_truths_with_ids(self, img_path, class_names):
        """Lê ground truths com IDs de classe"""
        gt_classes = []
        gt_ids = []
        
        # Possíveis caminhos para labels
        possible_paths = [
            img_path.parent.parent / 'labels' / 'val' / f"{img_path.stem}.txt",
            img_path.parent / 'labels' / f"{img_path.stem}.txt",
            Path(str(img_path.parent).replace('images', 'labels')) / f"{img_path.stem}.txt"
        ]
        
        label_path = None
        for p in possible_paths:
            if p.exists():
                label_path = p
                break
        
        if label_path and label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        cls_name = class_names.get(cls_id, str(cls_id))
                        gt_ids.append(cls_id)
                        gt_classes.append(cls_name)
        
        return gt_classes, gt_ids

    def _match_predictions(self, pred_classes, gt_classes, pred_ids, gt_ids, img_name):
        """Matching aprimorado considerando IDs"""
        if not pred_classes and not gt_classes:
            return [], [], [], []
        
        if not pred_classes:
            return ["background"] * len(gt_classes), gt_classes, [-1] * len(gt_ids), gt_ids
        
        if not gt_classes:
            return pred_classes, ["background"] * len(pred_classes), pred_ids, [-1] * len(pred_ids)
        
        if len(pred_classes) == len(gt_classes):
            return pred_classes, gt_classes, pred_ids, gt_ids
        
        # Se quantidades diferentes, usa estratégia de matching por IoU ou trunca
        min_len = min(len(pred_classes), len(gt_classes))
        return (pred_classes[:min_len], gt_classes[:min_len], 
                pred_ids[:min_len], gt_ids[:min_len])

    def _read_ground_truths_single(self, img_path, class_names):
        """Lê ground truths de UMA imagem específica"""
        gt_classes = []
        
        # Possíveis caminhos para labels
        possible_paths = [
            img_path.parent.parent / 'labels' / 'val' / f"{img_path.stem}.txt",
            img_path.parent / 'labels' / f"{img_path.stem}.txt",
            img_path.parent / f"{img_path.stem}.txt",
            Path(str(img_path.parent).replace('images', 'labels')) / f"{img_path.stem}.txt"
        ]
        
        label_path = None
        for p in possible_paths:
            if p.exists():
                label_path = p
                break
        
        if label_path and label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        cls_name = class_names.get(cls_id, str(cls_id))
                        gt_classes.append(cls_name)
        
        return gt_classes

    def _match_predictions_to_gt(self, pred_classes, gt_classes, img_name):
        if not pred_classes and not gt_classes:
            return [], []
        
        if not pred_classes:
            # Modelo não detectou nada, mas há GT
            return ["background"] * len(gt_classes), gt_classes
        
        if not gt_classes:
            # Falsos positivos
            return pred_classes, ["background"] * len(pred_classes)
        
        # Se quantidades iguais, assume correspondência direta
        if len(pred_classes) == len(gt_classes):
            return pred_classes, gt_classes
        
        # Se quantidades diferentes, usa estratégia de matching
        # Opção 1: Truncar na menor lista (conservador)
        min_len = min(len(pred_classes), len(gt_classes))
        return pred_classes[:min_len], gt_classes[:min_len]
        
        # Opção 2 (alternativa): Expandir com "unmatched"
        # max_len = max(len(pred_classes), len(gt_classes))
        # pred_extended = pred_classes + ["unmatched_pred"] * (max_len - len(pred_classes))
        # gt_extended = gt_classes + ["unmatched_gt"] * (max_len - len(gt_classes))
        # return pred_extended, gt_extended

    def _save_hierarchical_results(self, config, metrics, box_map50, sample_size):
        """Salva resultados da avaliação hierárquica"""
        import json
        from pathlib import Path
        
        result_file = Path(config.get("project", "runs")) / "hierarchical_eval.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(result_file, 'w') as f:
            json.dump({
                'h_mAP': metrics['h-mAP'],
                'traditional_mAP': metrics['traditional_mAP'],
                'standard_mAP50': box_map50,
                'sample_size': sample_size,
                'timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else None
            }, f, indent=2)
        print(f"\nSaved to: {result_file}")


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
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


# === MAPA DE CATEGORIAS TAXONOMICAS PARA RESUMO ===
# Agora normalizado ao nivel do GT:
# - correto: exato OU predicao mais especifica que o GT (descendente correto)
# - genero, familia, ordem, classe, filo: predicao ancestral no respectivo nivel
# - superior: kingdom, phylum como ancestral comum ou ancestral direto muito alto
# - sem_relacao: unrelated, unmatched, name_similarity
CATEGORY_MAP = {
    'exact': 'correto',
    'descendant_correct': 'correto',
    'ancestor_species': 'correto',
    'ancestor_genus': 'genero',
    'ancestor_family': 'familia',
    'ancestor_superfamily': 'familia',
    'ancestor_order': 'ordem',
    'ancestor_class': 'classe',
    'ancestor_gigaclass': 'superior',
    'ancestor_infraclass': 'superior',
    'ancestor_phylum': 'filo',
    'ancestor_kingdom': 'superior',
    'common_species': 'correto',
    'common_genus': 'genero',
    'common_family': 'familia',
    'common_superfamily': 'familia',
    'common_order': 'ordem',
    'common_class': 'classe',
    'common_gigaclass': 'superior',
    'common_infraclass': 'superior',
    'common_phylum': 'filo',
    'common_kingdom': 'superior',
    'unrelated': 'sem_relacao',
    'unmatched': 'sem_relacao',
    'name_similarity': 'sem_relacao',
}


class TrainThread(QThread):
    finished = pyqtSignal()
    epoch_progress = pyqtSignal(int)
    hier_metrics_ready = pyqtSignal(dict)

    def __init__(self, train_config):
        super().__init__()
        self.train_config = train_config
        self.success = False
        self.error = ''
        self.model_path = ''
        self._hierarchical_history = []
        self._yolo_model = None

    def run(self):
        try:
            model = YOLO('yolo26n.pt')
            self._yolo_model = model

            def on_epoch_end(trainer):
                self.epoch_progress.emit(trainer.epoch + 1)

            model.add_callback('on_train_epoch_end', on_epoch_end)

            aug_params = {
                'hsv_h': 0.05,
                'hsv_s': 0.9,
                'hsv_v': 0.6,
                'degrees': 15.0,
                'translate': 0.2,
                'scale': 0.7,
                'shear': 5.0,
                'perspective': 0.001,
                'flipud': 0.3,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.1,
                'copy_paste': 0.1,
                'auto_augment': 'randaugment',
                'erasing': 0.4,
            }

            final_config = {**aug_params, **self.train_config}
            model.train(**final_config)

            self._run_hierarchical_evaluation(model, final_config)

            self.model_path = os.path.join(
                'models', 'detect',
                final_config['name'],
                'weights', 'best.pt'
            )
            self.success = True

        except Exception as e:
            self.success = False
            self.error = str(e)
            print(f'Training error: {traceback.format_exc()}')
        finally:
            self.finished.emit()

    def _run_hierarchical_evaluation(self, model, config):
        try:
            data_path = config.get('data')
            if not data_path or not Path(data_path).exists():
                return

            validator = HierarchicalValidator(
                cache_file='worms_cache.json',
                dataset_yaml=data_path
            )

            if not validator.available:
                print('[Hierarchical] No taxonomy data available (cache or dataset)')
                return

            print('\n' + '='*60)
            print('HIERARCHICAL EVALUATION (FULL)')
            print('='*60)

            val_results = model.val(verbose=False)

            box_map50 = 0
            if hasattr(val_results, 'results_dict'):
                box_map50 = val_results.results_dict.get('metrics/mAP50(B)',
                            val_results.results_dict.get('metrics/mAP50', 0))
                print(f'Standard mAP@0.5: {box_map50:.4f}')

            val_path = Path(data_path).parent if Path(data_path).is_file() else Path(data_path)

            if val_path.exists():
                import random
                random.seed(42)

                val_images = self._find_val_images(val_path)

                if not val_images:
                    print('[Hierarchical] No validation images found')
                    return

                max_sample = 500
                if len(val_images) > max_sample:
                    sample = self._stratified_sample(val_images, max_sample)
                else:
                    sample = val_images

                print(f'Evaluating {len(sample)} validation images (from {len(val_images)} total)...')

                metrics = self._evaluate_sample(model, sample, validator, is_epoch_eval=False)

                if metrics:
                    h_map = metrics['h_mAP']
                    traditional_mAP = metrics['traditional_mAP']
                    calib = metrics['calibration_factor']
                    h_map_calib = metrics['h_mAP_calib']
                    trad_calib = metrics['traditional_mAP_calib']

                    print(f'\nh-mAP50 (hierarchical):   {h_map:.4f} (raw) / {h_map_calib:.4f} (calib)')
                    print(f'mAP50 (traditional):      {traditional_mAP:.4f} (raw) / {trad_calib:.4f} (calib)')
                    print(f'Standard mAP@0.5:         {box_map50:.4f}')
                    print(f'Calib factor:             {calib:.4f}')

                    print('\nBreakdown:')
                    for match_type, count in metrics['breakdown'].items():
                        pct = count / metrics['total_samples'] * 100
                        print(f'  {match_type}: {count} ({pct:.1f}%)')

                    if metrics.get('h_mAP_per_rank'):
                        print('\nh-mAP per rank:')
                        for rank, score in sorted(metrics['h_mAP_per_rank'].items(),
                                                   key=lambda x: -x[1]):
                            print(f'  {rank}: {score:.4f}')

                    if metrics.get('taxonomy_coverage'):
                        cov = metrics['taxonomy_coverage']
                        print(f'\nTaxonomy coverage: {cov["coverage_pct"]:.1f}% '
                              f'({cov["with_hierarchy"]}/{cov["total"]} with hierarchy)')

                    self._print_hierarchical_summary(metrics['breakdown'], metrics['total_samples'])

                    self._save_hierarchical_results(config, metrics, box_map50, len(sample))
                    self._save_hierarchical_history(config)

                print('='*60)

        except Exception as e:
            print(f'[Hierarchical] Evaluation error: {e}')
            import traceback
            traceback.print_exc()

    def _find_val_images(self, val_path: Path) -> list:
        val_images = []
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
        possible_val_dirs = [
            val_path / 'images' / 'val',
            val_path / 'val' / 'images',
            val_path / 'val',
        ]
        for val_dir in possible_val_dirs:
            if val_dir.exists() and val_dir.is_dir():
                for f in val_dir.rglob('*'):
                    if f.is_file() and f.suffix.lower() in extensions:
                        val_images.append(f)
        if not val_images and val_path.parent.exists():
            for val_dir in [
                val_path.parent / 'images' / 'val',
                val_path.parent / 'val' / 'images',
                val_path.parent / 'val',
            ]:
                if val_dir.exists() and val_dir.is_dir():
                    for f in val_dir.rglob('*'):
                        if f.is_file() and f.suffix.lower() in extensions:
                            val_images.append(f)
        seen = set()
        unique = []
        for p in val_images:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique

    def _stratified_sample(self, images: list, max_size: int) -> list:
        import random
        groups = defaultdict(list)
        for img in images:
            parent = img.parent.name
            groups[parent].append(img)
        total = len(images)
        sample = []
        for group, imgs in groups.items():
            n = max(1, int(len(imgs) / total * max_size))
            sample.extend(random.sample(imgs, min(n, len(imgs))))
        if len(sample) < max_size:
            remaining = [img for img in images if img not in sample]
            need = max_size - len(sample)
            sample.extend(random.sample(remaining, min(need, len(remaining))))
        return sample[:max_size]

    def _iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def _compute_ap(self, recall, precision):
        if not recall or not precision:
            return 0.0
        recall = [0.0] + list(recall) + [1.0]
        precision = [0.0] + list(precision) + [0.0]
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])
        ap = 0.0
        prev_r = 0.0
        for r, p in zip(recall, precision):
            if r > prev_r:
                ap += p * (r - prev_r)
                prev_r = r
        return ap

    def _evaluate_sample(self, model, sample: list, validator: HierarchicalValidator,
                         is_epoch_eval: bool = False) -> dict:
        import numpy as np
        all_class_names = list(model.names.values())

        # --- Passo 1: Coleta predicoes e GTs por imagem ---
        img_data = []
        for img_path in sample:
            r = model.predict(source=[img_path], verbose=False, conf=0.25)[0]
            preds = []
            if r.boxes:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                    preds.append({
                        'box': (x1, y1, x2, y2),
                        'cls': model.names[int(box.cls)],
                        'cls_id': int(box.cls),
                        'conf': float(box.conf)
                    })
            gt_boxes, gt_classes, gt_ids = self._read_ground_truths_with_ids(
                img_path, model.names
            )
            gts = [{'box': b, 'cls': c, 'cls_id': cid}
                   for b, c, cid in zip(gt_boxes, gt_classes, gt_ids)]
            img_data.append({'img_path': img_path, 'preds': preds, 'gts': gts})

        # --- Passo 2: MATCHING EXATO GLOBAL ---
        all_preds_exact = []
        for img_idx, img in enumerate(img_data):
            for pred_idx, p in enumerate(img['preds']):
                all_preds_exact.append({
                    'conf': p['conf'], 'img_idx': img_idx, 'pred_idx': pred_idx,
                    'cls': p['cls'], 'cls_id': p['cls_id'], 'box': p['box']
                })
        all_preds_exact.sort(key=lambda x: -x['conf'])

        for img in img_data:
            for g in img['gts']:
                g['matched_exact'] = False
                g['matched_hier'] = False

        exact_matches = []
        for p in all_preds_exact:
            img = img_data[p['img_idx']]
            best_iou = 0.0
            best_gt = None
            for g in img['gts']:
                if g['cls'] != p['cls'] or g.get('matched_exact', False):
                    continue
                iou = self._iou(p['box'], g['box'])
                if iou >= 0.5 and iou > best_iou:
                    best_iou = iou
                    best_gt = g
            if best_gt:
                best_gt['matched_exact'] = True
                exact_matches.append({
                    'pred_cls': p['cls'], 'pred_cls_id': p['cls_id'],
                    'gt_cls': best_gt['cls'], 'gt_cls_id': best_gt['cls_id'],
                    'conf': p['conf'], 'img_idx': p['img_idx'], 'score': 1.0, 'type': 'exact'
                })

        # --- Passo 3: MATCHING HIERARQUICO GLOBAL ---
        pred_matched_exact = set()
        for m in exact_matches:
            img = img_data[m['img_idx']]
            for pi, p in enumerate(img['preds']):
                if abs(p['conf'] - m['conf']) < 1e-9 and p['cls'] == m['pred_cls']:
                    key = (m['img_idx'], pi)
                    if key not in pred_matched_exact:
                        pred_matched_exact.add(key)
                        break

        all_preds_hier = []
        for img_idx, img in enumerate(img_data):
            for pred_idx, p in enumerate(img['preds']):
                if (img_idx, pred_idx) not in pred_matched_exact:
                    all_preds_hier.append({
                        'conf': p['conf'], 'img_idx': img_idx, 'pred_idx': pred_idx,
                        'cls': p['cls'], 'cls_id': p['cls_id'], 'box': p['box']
                    })
        all_preds_hier.sort(key=lambda x: -x['conf'])

        hier_matches = []
        for p in all_preds_hier:
            img = img_data[p['img_idx']]
            best_iou = 0.0
            best_gt = None
            for g in img['gts']:
                if g.get('matched_exact', False) or g.get('matched_hier', False):
                    continue
                iou = self._iou(p['box'], g['box'])
                if iou >= 0.5 and iou > best_iou:
                    best_iou = iou
                    best_gt = g
            if best_gt:
                best_gt['matched_hier'] = True
                if validator.available:
                    score, mtype = validator.calculate_score(
                        p['cls'], best_gt['cls'], p['cls_id'], best_gt['cls_id']
                    )
                else:
                    score = 0.0
                    mtype = 'unrelated'
                hier_matches.append({
                    'pred_cls': p['cls'], 'pred_cls_id': p['cls_id'],
                    'gt_cls': best_gt['cls'], 'gt_cls_id': best_gt['cls_id'],
                    'conf': p['conf'], 'img_idx': p['img_idx'], 'score': score, 'type': mtype
                })

        # --- Passo 4: mAP TRADICIONAL (por classe) ---
        traditional_ap_per_class = {}
        for cls_name in all_class_names:
            cls_preds = [m for m in exact_matches if m['pred_cls'] == cls_name]
            num_gt = sum(1 for img in img_data for g in img['gts'] if g['cls'] == cls_name)
            if not cls_preds:
                traditional_ap_per_class[cls_name] = 0.0
                continue
            cls_preds.sort(key=lambda x: -x['conf'])
            scores = [m['score'] for m in cls_preds]
            cumsum = 0
            precisions = []
            recalls = []
            for i, s in enumerate(scores, 1):
                cumsum += s
                precisions.append(cumsum / i)
                recalls.append(cumsum / num_gt if num_gt > 0 else 0.0)
            traditional_ap_per_class[cls_name] = self._compute_ap(recalls, precisions)
        traditional_mAP = sum(traditional_ap_per_class.values()) / len(all_class_names)

        # --- Passo 5: h-mAP (por classe, com matches globais) ---
        h_ap_per_class = {}
        for cls_name in all_class_names:
            cls_matches = ([m for m in exact_matches if m['pred_cls'] == cls_name] +
                           [m for m in hier_matches if m['pred_cls'] == cls_name])
            num_gt = sum(1 for img in img_data for g in img['gts'] if g['cls'] == cls_name)
            if not cls_matches:
                h_ap_per_class[cls_name] = 0.0
                continue
            cls_matches.sort(key=lambda x: -x['conf'])
            scores = [m['score'] for m in cls_matches]
            cumsum = 0
            precisions = []
            recalls = []
            for i, s in enumerate(scores, 1):
                cumsum += s
                precisions.append(cumsum / i)
                recalls.append(cumsum / num_gt if num_gt > 0 else 0.0)
            h_ap_per_class[cls_name] = self._compute_ap(recalls, precisions)
        h_map = sum(h_ap_per_class.values()) / len(all_class_names)

        # --- Passo 6: Calibracao ---
        trad_raw = metrics['traditional_mAP'] if False else traditional_mAP  # placeholder para definicao abaixo
        # Calibracao sera aplicada no metodo chamador

        # --- Estatisticas extras ---
        all_matches = exact_matches + hier_matches
        type_counts = Counter(m['type'] for m in all_matches)
        all_ious = []
        total_with_h = 0
        total_without_h = 0
        for idx, img in enumerate(img_data):
            for g in img['gts']:
                g['matched_stat'] = False
            for p in img['preds']:
                best_iou = 0.0
                best_gt = None
                for g in img['gts']:
                    if g.get('matched_stat', False):
                        continue
                    iou = self._iou(p['box'], g['box'])
                    if iou >= 0.5 and iou > best_iou:
                        best_iou = iou
                        best_gt = g
                if best_gt:
                    best_gt['matched_stat'] = True
                    all_ious.append(best_iou)
                    gt_hier = validator.get_hierarchy_by_id(best_gt['cls_id'], best_gt['cls'])
                    if len(gt_hier) > 1 or gt_hier[0].get('rank') != 'unknown':
                        total_with_h += 1
                    else:
                        total_without_h += 1
        mean_iou = sum(all_ious) / len(all_ious) if all_ious else 0
        total_samples = len(all_matches)

        return {
            'h_mAP': h_map,
            'traditional_mAP': traditional_mAP,
            'mean_iou': mean_iou,
            'scores': all_matches if not is_epoch_eval else [],
            'breakdown': dict(type_counts.most_common()),
            'total_samples': total_samples,
            'taxonomy_coverage': {
                'with_hierarchy': total_with_h,
                'without_hierarchy': total_without_h,
                'total': total_with_h + total_without_h,
                'coverage_pct': (total_with_h / (total_with_h + total_without_h) * 100)
                                if (total_with_h + total_without_h) > 0 else 0
            },
            'h_ap50_per_class': h_ap_per_class,
            'traditional_ap50_per_class': traditional_ap_per_class,
            'num_exact_matches': len(exact_matches),
            'num_hier_matches': len(hier_matches),
        }

    def _print_hierarchical_summary(self, breakdown, total_samples):
        "Imprime resumo hierarquico por nivel taxonomico, normalizado ao GT."
        cats = {
            'correto': 0,
            'genero': 0,
            'familia': 0,
            'ordem': 0,
            'classe': 0,
            'filo': 0,
            'superior': 0,
            'sem_relacao': 0
        }
        for mtype, count in breakdown.items():
            cat = CATEGORY_MAP.get(mtype, 'sem_relacao')
            cats[cat] += count

        def pct(n):
            return (n / total_samples * 100) if total_samples > 0 else 0

        print('\n' + '='*60)
        print('RESUMO HIERARQUICO POR NIVEL TAXONOMICO (NORMALIZADO AO GT)')
        print('='*60)
        print(f'  Correto (exato/descendente): {cats["correto"]:>5}  ({pct(cats["correto"]):.1f}%)')
        print(f'  Genero (ancestral comum):    {cats["genero"]:>5}  ({pct(cats["genero"]):.1f}%)')
        print(f'  Familia (ancestral comum):   {cats["familia"]:>5}  ({pct(cats["familia"]):.1f}%)')
        print(f'  Ordem (ancestral comum):     {cats["ordem"]:>5}  ({pct(cats["ordem"]):.1f}%)')
        print(f'  Classe (ancestral comum):    {cats["classe"]:>5}  ({pct(cats["classe"]):.1f}%)')
        print(f'  Filo (ancestral comum):      {cats["filo"]:>5}  ({pct(cats["filo"]):.1f}%)')
        print(f'  Superior (reino/etc):        {cats["superior"]:>5}  ({pct(cats["superior"]):.1f}%)')
        print(f'  Sem relacao:                 {cats["sem_relacao"]:>5}  ({pct(cats["sem_relacao"]):.1f}%)')

        print('\n--- TEXTO PARA ARTIGO ---')
        cor = pct(cats['correto'])
        gen = pct(cats['genero'])
        fam = pct(cats['familia'])
        sup = pct(cats['ordem']) + pct(cats['classe']) + pct(cats['filo']) + pct(cats['superior'])
        sem = pct(cats['sem_relacao'])

        print(f'A analise hierarquica mostrou que {cor:.1f}% das deteccoes')
        print(f'foram corretas no nivel do GT ou mais especificas.')
        if gen > 0:
            print(f'Adicionalmente, {gen:.1f}% atingiram o nivel de genero.')
        if fam > 0:
            print(f'{fam:.1f}% atingiram o nivel de familia.')
        if sup > 0:
            print(f'O restante {sup:.1f}% correspondeu a niveis superiores ou distantes.')
        if sem > 0:
            print(f'{sem:.1f}% nao apresentaram relacao taxonomica.')
        print('='*60)

    def _read_ground_truths_with_ids(self, img_path, class_names):
        gt_boxes = []
        gt_classes = []
        gt_ids = []
        img_path = Path(img_path)
        possible_label_paths = []
        parts = img_path.parts
        if 'images' in parts:
            img_idx = parts.index('images')
            base_path = Path(*parts[:img_idx])
            rel_path = Path(*parts[img_idx + 1:])
            label_path = base_path / 'labels' / rel_path.parent / f"{img_path.stem}.txt"
            possible_label_paths.append(label_path)
        if 'val' in parts or 'train' in parts:
            for split in ['val', 'train']:
                if split in parts:
                    split_idx = parts.index(split)
                    base_path = Path(*parts[:split_idx])
                    rel_after_split = Path(*parts[split_idx + 1:])
                    if rel_after_split.parts[0] == 'images':
                        label_path = base_path / split / 'labels' / f"{img_path.stem}.txt"
                        possible_label_paths.append(label_path)
        possible_label_paths.extend([
            img_path.parent.parent / 'labels' / 'val' / f"{img_path.stem}.txt",
            img_path.parent.parent / 'labels' / img_path.parent.name / f"{img_path.stem}.txt",
            img_path.parent / 'labels' / f"{img_path.stem}.txt",
            Path(str(img_path.parent).replace('images', 'labels')) / f"{img_path.stem}.txt",
            img_path.parent.parent.parent / 'labels' / 'val' / f"{img_path.stem}.txt",
            img_path.parent.parent.parent / 'labels' / img_path.parent.name / f"{img_path.stem}.txt",
        ])
        label_path = None
        for p in possible_label_paths:
            if p.exists():
                label_path = p
                break
        if label_path and label_path.exists():
            import cv2
            img = cv2.imread(str(img_path))
            if img is not None:
                img_h, img_w = img.shape[:2]
            else:
                img_h, img_w = 1, 1
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts_line = line.split()
                    if len(parts_line) >= 5:
                        try:
                            cls_id = int(parts_line[0])
                            x_center = float(parts_line[1])
                            y_center = float(parts_line[2])
                            width = float(parts_line[3])
                            height = float(parts_line[4])
                            x1 = (x_center - width/2) * img_w
                            y1 = (y_center - height/2) * img_h
                            x2 = (x_center + width/2) * img_w
                            y2 = (y_center + height/2) * img_h
                            cls_name = class_names.get(cls_id, str(cls_id))
                            gt_ids.append(cls_id)
                            gt_classes.append(cls_name)
                            gt_boxes.append((x1, y1, x2, y2))
                        except (ValueError, IndexError):
                            continue
        return gt_boxes, gt_classes, gt_ids

    def _save_hierarchical_results(self, config, metrics, box_map50, sample_size):
        result_file = Path(config.get('project', 'runs')) / 'hierarchical_eval.json'
        result_file.parent.mkdir(parents=True, exist_ok=True)
        trad_raw = metrics['traditional_mAP']
        h_raw = metrics['h_mAP']
        calib = box_map50 / trad_raw if trad_raw > 0 else 1.0
        output = {
            'h_mAP': h_raw,
            'h_mAP_calib': h_raw * calib,
            'traditional_mAP': trad_raw,
            'traditional_mAP_calib': trad_raw * calib,
            'standard_mAP50': box_map50,
            'calibration_factor': calib,
            'gap_h_vs_traditional': (h_raw * calib) - (trad_raw * calib),
            'mean_iou': metrics.get('mean_iou', 0),
            'sample_size': sample_size,
            'total_evaluated': metrics['total_samples'],
            'breakdown': metrics['breakdown'],
            'taxonomy_coverage': metrics.get('taxonomy_coverage', {}),
            'h_ap50_per_class': metrics.get('h_ap50_per_class', {}),
            'traditional_ap50_per_class': metrics.get('traditional_ap50_per_class', {}),
            'num_exact_matches': metrics.get('num_exact_matches', 0),
            'num_hier_matches': metrics.get('num_hier_matches', 0),
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
        print(f'\nSaved hierarchical eval to: {result_file}')

    def _save_hierarchical_history(self, config):
        if not self._hierarchical_history:
            return
        history_file = Path(config.get('project', 'runs')) / 'hierarchical_history.json'
        history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(history_file, 'w') as f:
            json.dump({
                'history': self._hierarchical_history,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        print(f'Saved hierarchical history ({len(self._hierarchical_history)} epochs) to: {history_file}')


class TrainSegmentationThread(QThread):
    finished = pyqtSignal()
    epoch_progress = pyqtSignal(int)

    def __init__(self, train_config):
        super().__init__()
        self.train_config = train_config
        self.success = False
        self.error = ''
        self.model_path = ''

    def run(self):
        try:
            model = YOLO('yolo26n-seg.pt')

            def on_epoch_end(trainer):
                self.epoch_progress.emit(trainer.epoch + 1)

            model.add_callback('on_train_epoch_end', on_epoch_end)

            aug_params = {
                'hsv_h': 0.05,
                'hsv_s': 0.9,
                'hsv_v': 0.6,
                'degrees': 15.0,
                'translate': 0.2,
                'scale': 0.7,
                'shear': 5.0,
                'perspective': 0.001,
                'flipud': 0.3,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.1,
                'copy_paste': 0.1,
                'auto_augment': 'randaugment',
                'erasing': 0.4,
            }

            final_config = {**aug_params, **self.train_config}
            final_config['task'] = 'segment'

            model.train(**final_config)

            self.model_path = os.path.join(
                'models', 'segment',
                final_config['name'],
                'weights', 'best.pt'
            )
            self.success = True

        except Exception as e:
            self.success = False
            self.error = str(e)
            print(f'Segmentation training error: {traceback.format_exc()}')
        finally:
            self.finished.emit()
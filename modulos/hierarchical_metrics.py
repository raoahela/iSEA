import yaml
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher
import numpy as np


class HierarchicalValidator:

    # Scores por rank (alinhado com sua descrição intuitiva)
    RANK_SCORES = {
        'species': 1.0,
        'genus': 0.9,
        'family': 0.75,
        'superfamily': 0.65,
        'order': 0.50,
        'class': 0.25,
        'phylum': 0.10,
        'kingdom': 0.05,
        'gigaclass': 0.20,
        'infraclass': 0.22,
        'unknown': 0.0
    }

    # Ordem taxonômica do mais geral para o mais específico (para cálculo de distância)
    RANK_ORDER = [
        'kingdom', 'phylum', 'class', 'gigaclass', 'infraclass',
        'superfamily', 'order', 'family', 'genus', 'species'
    ]

    def __init__(self, cache_file: str = "worms_cache.json", dataset_yaml: str = None):
        cache_path = Path(cache_file)
        self.cache = {}
        self.dataset_taxonomy = {}
        self.available = False
        self._name_index = {}

        if cache_path.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                self.cache = json.load(f)

        if dataset_yaml and Path(dataset_yaml).exists():
            self._load_dataset_taxonomy(dataset_yaml)

        self.available = len(self.cache) > 0 or len(self.dataset_taxonomy) > 0

    def _load_dataset_taxonomy(self, yaml_path: str):
        yaml_file = Path(yaml_path)
        base_dir = yaml_file.parent

        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)

        if data and 'taxonomy' in data and 'classes' in data['taxonomy']:
            for class_id, tax_info in data['taxonomy']['classes'].items():
                class_id = int(class_id)
                hierarchy = self._parse_yaml_taxonomy(tax_info)
                self.dataset_taxonomy[class_id] = hierarchy
                if hierarchy:
                    self._name_index[hierarchy[0]["name"].lower()] = class_id
        else:
            csv_path = base_dir / "taxonomy_metadata.csv"
            if csv_path.exists():
                self._load_from_csv(csv_path)

    def _parse_yaml_taxonomy(self, tax_info: dict) -> List[Dict]:
        hierarchy = []
        lineage = tax_info.get('lineage', '')
        name = tax_info.get('name', 'unknown')
        rank = tax_info.get('rank', 'unknown')

        if lineage and ' (' in lineage:
            main_name = lineage.split(' (')[0]
            parents = lineage.split(' (')[1].rstrip(')').split(', ')

            hierarchy.append({
                "name": name,
                "rank": rank,
                "aphia_id": tax_info.get('aphia_id')
            })

            rank_order = ['genus', 'family', 'order', 'class', 'phylum', 'kingdom']
            parent_rank_idx = 0
            for parent in parents[:3]:
                estimated_rank = rank_order[min(parent_rank_idx, len(rank_order)-1)]
                hierarchy.append({
                    "name": parent.strip(),
                    "rank": estimated_rank,
                    "aphia_id": None
                })
                parent_rank_idx += 1
        else:
            hierarchy.append({
                "name": name,
                "rank": rank,
                "aphia_id": tax_info.get('aphia_id')
            })

        return hierarchy

    def _load_from_csv(self, csv_path: Path):
        import csv
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_id = int(row['class_id'])
                hierarchy = []
                ranks = ['species', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom']
                for rank in ranks:
                    if row.get(rank):
                        hierarchy.append({
                            "name": row[rank],
                            "rank": rank,
                            "aphia_id": row.get('aphia_id') if rank == row.get('rank') else None
                        })
                if not hierarchy:
                    hierarchy.append({
                        "name": row['original_name'],
                        "rank": row.get('rank', 'unknown'),
                        "aphia_id": row.get('aphia_id')
                    })
                self.dataset_taxonomy[class_id] = hierarchy
                if hierarchy:
                    self._name_index[hierarchy[0]["name"].lower()] = class_id

    def get_hierarchy(self, taxon: str) -> List[Dict]:
        if isinstance(taxon, str) and taxon.isdigit():
            taxon_id = int(taxon)
            if taxon_id in self.dataset_taxonomy:
                return self.dataset_taxonomy[taxon_id]
        return self.cache.get(taxon, [{"name": taxon, "rank": "unknown"}])

    def get_hierarchy_by_id(self, class_id: int, class_name: str = None) -> List[Dict]:
        if class_id in self.dataset_taxonomy:
            return self.dataset_taxonomy[class_id]
        if class_name:
            return self.get_hierarchy(class_name)
        return [{"name": str(class_id), "rank": "unknown"}]

    def _get_rank_of(self, name: str, hierarchy: List[Dict]) -> str:
        for node in hierarchy:
            if node["name"].lower() == name.lower():
                return node.get("rank", "unknown")
        return "unknown"

    def _find_deepest_common(self, pred_hier: List[Dict], gt_hier: List[Dict]) -> Optional[Dict]:
        gt_names_lower = {n["name"].lower() for n in gt_hier}
        best_node = None
        best_depth = -1
        for idx, node in enumerate(pred_hier):
            if node["name"].lower() in gt_names_lower:
                rank = node.get("rank", "unknown")
                depth = self.RANK_ORDER.index(rank) if rank in self.RANK_ORDER else -1
                if depth > best_depth:
                    best_depth = depth
                    best_node = node
        return best_node

    def calculate_score(self, pred: str, gt: str, pred_id: int = None, gt_id: int = None) -> Tuple[float, str]:
        if pred_id is not None and gt_id is not None:
            pred_hier = self.get_hierarchy_by_id(pred_id, pred)
            gt_hier = self.get_hierarchy_by_id(gt_id, gt)
        else:
            pred_hier = self.get_hierarchy(pred)
            gt_hier = self.get_hierarchy(gt)

        pred_names = [n["name"] for n in pred_hier]
        gt_names = [n["name"] for n in gt_hier]
        pred_names_lower = [n.lower() for n in pred_names]
        gt_names_lower = [n.lower() for n in gt_names]

        # 1) EXATO (case-insensitive)
        if pred.lower() == gt.lower() or (pred_names_lower and gt_names_lower and pred_names_lower[0] == gt_names_lower[0]):
            return 1.0, "exact"

        # 2) PREDIÇÃO É DESCENDENTE DO GT (mais específica, mas correta)
        #    Ex: GT="Ophiuridae" (família), Pred="Ophiura ophiura" (espécie)
        #    O GT aparece na hierarquia da predição.
        if gt.lower() in pred_names_lower:
            return 1.0, "descendant_correct"

        # 3) PREDIÇÃO É ANCESTRAL DO GT (menos específica)
        #    Ex: GT="Ophiura ophiura" (espécie), Pred="Ophiuridae" (família)
        #    Penalização proporcional ao rank da predição (quanto mais alto, menor o score).
        if pred.lower() in gt_names_lower:
            pred_rank = self._get_rank_of(pred, pred_hier)
            score = self.RANK_SCORES.get(pred_rank, 0.0)
            return score, f"ancestor_{pred_rank}"

        # 4) ANCESTRAL COMUM (ramos divergentes)
        common_node = self._find_deepest_common(pred_hier, gt_hier)
        if common_node:
            rank = common_node.get("rank", "unknown")
            base_score = self.RANK_SCORES.get(rank, 0.0)
            return base_score * 0.5, f"common_{rank}"

        # 5) FALLBACK: similaridade de nome
        sim = SequenceMatcher(None, pred.lower(), gt.lower()).ratio()
        if sim > 0.8:
            return sim * 0.3, "name_similarity"

        return 0.0, "unrelated"

    def calculate_scores_batch(self, predictions: List[Tuple], ground_truths: List[Tuple]) -> Dict:
        scores = []
        for pred, gt in zip(predictions, ground_truths):
            pred_name, pred_id = pred[0], pred[1]
            gt_name, gt_id = gt[0], gt[1]
            score, match_type = self.calculate_score(pred_name, gt_name, pred_id, gt_id)
            scores.append({
                "pred": pred_name, "gt": gt_name,
                "pred_id": pred_id, "gt_id": gt_id,
                "score": score, "type": match_type
            })

        if not scores:
            return {"h_mAP": 0.0, "traditional_mAP": 0.0, "scores": []}

        h_map = sum(s["score"] for s in scores) / len(scores)
        exact = sum(1 for s in scores if s["type"] == "exact")

        from collections import Counter
        type_counts = Counter(s["type"] for s in scores)

        rank_scores = {}
        for s in scores:
            gt_hier = self.get_hierarchy_by_id(s["gt_id"], s["gt"])
            gt_rank = gt_hier[0].get("rank", "unknown") if gt_hier else "unknown"
            if gt_rank not in rank_scores:
                rank_scores[gt_rank] = []
            rank_scores[gt_rank].append(s["score"])

        h_mAP_per_rank = {
            rank: sum(scores_list) / len(scores_list)
            for rank, scores_list in rank_scores.items()
        }

        return {
            "h_mAP": h_map,
            "traditional_mAP": exact / len(scores),
            "scores": scores,
            "breakdown": dict(type_counts.most_common()),
            "h_mAP_per_rank": h_mAP_per_rank,
            "total_samples": len(scores)
        }


class IoUMatcher:

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    @staticmethod
    def compute_iou(box1: Tuple[float, float, float, float],
                    box2: Tuple[float, float, float, float]) -> float:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

        union_area = box1_area + box2_area - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def match(self,
              pred_boxes: List[Tuple],
              gt_boxes: List[Tuple],
              pred_classes: List[str],
              gt_classes: List[str],
              pred_ids: List[int],
              gt_ids: List[int]) -> Tuple[List, List, List, List, List]:
        n_pred = len(pred_boxes)
        n_gt = len(gt_boxes)

        if n_pred == 0 or n_gt == 0:
            return [], [], [], [], []

        iou_matrix = np.zeros((n_pred, n_gt))
        for i in range(n_pred):
            for j in range(n_gt):
                iou_matrix[i, j] = self.compute_iou(pred_boxes[i], gt_boxes[j])

        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        except ImportError:
            row_ind, col_ind = self._greedy_match(iou_matrix)

        matched_preds = []
        matched_gts = []
        matched_pred_ids = []
        matched_gt_ids = []
        matched_ious = []

        used_pred = set()
        used_gt = set()

        for i, j in zip(row_ind, col_ind):
            if iou_matrix[i, j] >= self.iou_threshold:
                matched_preds.append(pred_classes[i])
                matched_gts.append(gt_classes[j])
                matched_pred_ids.append(pred_ids[i])
                matched_gt_ids.append(gt_ids[j])
                matched_ious.append(float(iou_matrix[i, j]))
                used_pred.add(i)
                used_gt.add(j)

        for i in range(n_pred):
            if i not in used_pred:
                matched_preds.append(pred_classes[i])
                matched_gts.append("background")
                matched_pred_ids.append(pred_ids[i])
                matched_gt_ids.append(-1)
                matched_ious.append(0.0)

        for j in range(n_gt):
            if j not in used_gt:
                matched_preds.append("background")
                matched_gts.append(gt_classes[j])
                matched_pred_ids.append(-1)
                matched_gt_ids.append(gt_ids[j])
                matched_ious.append(0.0)

        return matched_preds, matched_gts, matched_pred_ids, matched_gt_ids, matched_ious

    def _greedy_match(self, iou_matrix: np.ndarray) -> Tuple[List[int], List[int]]:
        n_pred, n_gt = iou_matrix.shape
        matched_pred = []
        matched_gt = []
        used_pred = set()
        used_gt = set()

        pairs = []
        for i in range(n_pred):
            for j in range(n_gt):
                pairs.append((iou_matrix[i, j], i, j))
        pairs.sort(reverse=True)

        for iou, i, j in pairs:
            if i not in used_pred and j not in used_gt:
                matched_pred.append(i)
                matched_gt.append(j)
                used_pred.add(i)
                used_gt.add(j)

        return matched_pred, matched_gt
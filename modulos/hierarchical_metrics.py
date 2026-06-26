import yaml
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher
import numpy as np


class HierarchicalValidator:

    RANK_SCORES = {
        'species': 1.0, 'genus': 0.85, 'family': 0.7,
        'order': 0.55, 'class': 0.4, 'phylum': 0.25, 'kingdom': 0.1
    }

    # Pesos ajustáveis para diferentes tipos de match
    MATCH_WEIGHTS = {
        'exact': 1.0,
        'ancestor': 0.85,      # Predizer ancestral é quase tão bom quanto exato
        'descendant': 0.75,    # Predizer descendente (excesso de especificidade)
        'common': 0.5,         # Ancestral comum
        'name_similarity': 0.3, # Similaridade de nome (fallback)
        'unrelated': 0.0
    }

    def __init__(self, cache_file: str = "worms_cache.json", dataset_yaml: str = None):
        """
        Args:
            cache_file: Cache global do WoRMS
            dataset_yaml: Path para dataset.yaml (opcional, para datasets enriquecidos)
        """
        cache_path = Path(cache_file)
        self.cache = {}
        self.dataset_taxonomy = {}  # id -> hierarquia
        self.available = False
        self._name_index = {}  # Índice invertido: nome -> class_id (para fallback)

        # Carrega cache global
        if cache_path.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                self.cache = json.load(f)

        # Carrega metadados do dataset se fornecido
        if dataset_yaml and Path(dataset_yaml).exists():
            self._load_dataset_taxonomy(dataset_yaml)

        self.available = len(self.cache) > 0 or len(self.dataset_taxonomy) > 0

    def _load_dataset_taxonomy(self, yaml_path: str):
        """Carrega hierarquia do dataset.yaml (seção taxonomy) ou taxonomy_metadata.csv"""
        yaml_file = Path(yaml_path)
        base_dir = yaml_file.parent

        # Tenta carregar do YAML primeiro (se foi enriquecido)
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)

        if data and 'taxonomy' in data and 'classes' in data['taxonomy']:
            # Formato enriquecido pelo YOLODatasetEnricher
            for class_id, tax_info in data['taxonomy']['classes'].items():
                class_id = int(class_id)
                hierarchy = self._parse_yaml_taxonomy(tax_info)
                self.dataset_taxonomy[class_id] = hierarchy
                # Indexar nomes para fallback
                if hierarchy:
                    self._name_index[hierarchy[0]["name"].lower()] = class_id
        else:
            # Fallback: carrega do CSV de metadados se existir
            csv_path = base_dir / "taxonomy_metadata.csv"
            if csv_path.exists():
                self._load_from_csv(csv_path)

    def _parse_yaml_taxonomy(self, tax_info: dict) -> List[Dict]:
        """Converte a seção taxonomy do YAML para formato de hierarquia"""
        hierarchy = []
        lineage = tax_info.get('lineage', '')
        name = tax_info.get('name', 'unknown')
        rank = tax_info.get('rank', 'unknown')

        # Se tiver lineage completo, parseia
        if lineage and ' (' in lineage:
            main_name = lineage.split(' (')[0]
            parents = lineage.split(' (')[1].rstrip(')').split(', ')

            hierarchy.append({
                "name": name,
                "rank": rank,
                "aphia_id": tax_info.get('aphia_id')
            })

            # Adiciona pais se disponíveis (com ranks estimados)
            rank_order = ['genus', 'family', 'order', 'class', 'phylum', 'kingdom']
            parent_rank_idx = 0
            for parent in parents[:3]:  # Limita a 3 níveis superiores
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
        """Carrega metadados do CSV gerado pelo YOLODatasetEnricher"""
        import csv
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_id = int(row['class_id'])
                hierarchy = []

                # Reconstroi hierarquia do CSV
                ranks = ['species', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom']
                for rank in ranks:
                    if row.get(rank):
                        hierarchy.append({
                            "name": row[rank],
                            "rank": rank,
                            "aphia_id": row.get('aphia_id') if rank == row.get('rank') else None
                        })

                # Se não achou hierarquia detalhada, usa pelo menos o nome original
                if not hierarchy:
                    hierarchy.append({
                        "name": row['original_name'],
                        "rank": row.get('rank', 'unknown'),
                        "aphia_id": row.get('aphia_id')
                    })

                self.dataset_taxonomy[class_id] = hierarchy
                # Indexar nome
                if hierarchy:
                    self._name_index[hierarchy[0]["name"].lower()] = class_id

    def get_hierarchy(self, taxon: str) -> List[Dict]:
        """Busca hierarquia: primeiro no dataset local, depois no cache global"""
        if isinstance(taxon, str) and taxon.isdigit():
            taxon_id = int(taxon)
            if taxon_id in self.dataset_taxonomy:
                return self.dataset_taxonomy[taxon_id]

        # Se é nome, busca no cache
        return self.cache.get(taxon, [{"name": taxon, "rank": "unknown"}])

    def get_hierarchy_by_id(self, class_id: int, class_name: str = None) -> List[Dict]:
        """Obtém hierarquia por ID de classe YOLO"""
        if class_id in self.dataset_taxonomy:
            return self.dataset_taxonomy[class_id]

        if class_name:
            return self.get_hierarchy(class_name)

        return [{"name": str(class_id), "rank": "unknown"}]

    def calculate_score(self, pred: str, gt: str, pred_id: int = None, gt_id: int = None) -> Tuple[float, str]:
        """
        Calcula score hierárquico.
        Pode receber nomes (str) ou IDs (int) das classes.
        Retorna: (score, match_type)
        """
        # Obtém hierarquias
        if pred_id is not None and gt_id is not None:
            pred_hier = self.get_hierarchy_by_id(pred_id, pred)
            gt_hier = self.get_hierarchy_by_id(gt_id, gt)
        else:
            pred_hier = self.get_hierarchy(pred)
            gt_hier = self.get_hierarchy(gt)

        pred_names = [n["name"] for n in pred_hier]
        gt_names = [n["name"] for n in gt_hier]

        # Match exato (case-insensitive para robustez)
        if pred.lower() == gt.lower() or (pred_names and gt_names and pred_names[0].lower() == gt_names[0].lower()):
            return 1.0, "exact"

        # Predição é ancestral do GT
        if pred in gt_names:
            for node in gt_hier:
                if node["name"] == pred:
                    score = self.RANK_SCORES.get(node["rank"], 0.0) * self.MATCH_WEIGHTS['ancestor']
                    return score, f"ancestor_{node['rank']}"

        # GT é ancestral da predição (erro por excesso de especificidade)
        if gt in pred_names:
            for node in pred_hier:
                if node["name"] == gt:
                    score = self.RANK_SCORES.get(node["rank"], 0.0) * self.MATCH_WEIGHTS['descendant']
                    return score, f"descendant_{node['rank']}"

        # Ancestral comum
        common = set(pred_names) & set(gt_names)
        if common:
            best_score = 0
            best_rank = "unknown"
            for node in pred_hier:
                if node["name"] in common:
                    score = self.RANK_SCORES.get(node["rank"], 0.0)
                    if score > best_score:
                        best_score = score
                        best_rank = node["rank"]

            return best_score * self.MATCH_WEIGHTS['common'], f"common_{best_rank}"

        # Fallback: similaridade de nome (quando não há taxonomia disponível)
        sim = SequenceMatcher(None, pred.lower(), gt.lower()).ratio()
        if sim > 0.8:
            return sim * self.MATCH_WEIGHTS['name_similarity'], "name_similarity"

        return 0.0, "unrelated"

    def calculate_scores_batch(self, predictions: List[Tuple], ground_truths: List[Tuple]) -> Dict:
        """
        Calcula scores hierárquicos para um batch de predições.

        Args:
            predictions: Lista de (class_name, class_id, bbox)
            ground_truths: Lista de (class_name, class_id, bbox)

        Returns:
            Dict com métricas agregadas
        """
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

        # Breakdown por tipo
        from collections import Counter
        type_counts = Counter(s["type"] for s in scores)

        # Breakdown por rank do GT
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
    """
    Matching de predições com ground-truths baseado em IoU (Intersection over Union).
    Usa algoritmo Hungarian (Kuhn-Munkres) para matching ótimo.
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    @staticmethod
    def compute_iou(box1: Tuple[float, float, float, float], 
                    box2: Tuple[float, float, float, float]) -> float:
        """
        Calcula IoU entre duas bounding boxes.
        Formato: (x1, y1, x2, y2)
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Coordenadas da interseção
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        # Áreas
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
        """
        Faz matching ótimo entre predições e GTs baseado em IoU.

        Args:
            pred_boxes: Lista de (x1, y1, x2, y2) das predições
            gt_boxes: Lista de (x1, y1, x2, y2) dos GTs
            pred_classes: Nomes das classes preditas
            gt_classes: Nomes das classes GT
            pred_ids: IDs das classes preditas
            gt_ids: IDs das classes GT

        Returns:
            (matched_preds, matched_gts, matched_pred_ids, matched_gt_ids, matched_ious)
            Listas apenas com os pares que satisfazem iou_threshold
        """
        n_pred = len(pred_boxes)
        n_gt = len(gt_boxes)

        if n_pred == 0 or n_gt == 0:
            return [], [], [], [], []

        # Calcular matriz de IoU
        iou_matrix = np.zeros((n_pred, n_gt))
        for i in range(n_pred):
            for j in range(n_gt):
                iou_matrix[i, j] = self.compute_iou(pred_boxes[i], gt_boxes[j])

        # Matching ótimo via Hungarian algorithm (maximiza IoU total)
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        except ImportError:
            # Fallback: greedy matching se scipy não disponível
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

        # Adicionar falsos positivos (predições sem match)
        for i in range(n_pred):
            if i not in used_pred:
                matched_preds.append(pred_classes[i])
                matched_gts.append("background")
                matched_pred_ids.append(pred_ids[i])
                matched_gt_ids.append(-1)
                matched_ious.append(0.0)

        # Adicionar falsos negativos (GTs sem match)
        for j in range(n_gt):
            if j not in used_gt:
                matched_preds.append("background")
                matched_gts.append(gt_classes[j])
                matched_pred_ids.append(-1)
                matched_gt_ids.append(gt_ids[j])
                matched_ious.append(0.0)

        return matched_preds, matched_gts, matched_pred_ids, matched_gt_ids, matched_ious

    def _greedy_match(self, iou_matrix: np.ndarray) -> Tuple[List[int], List[int]]:
        """Matching greedy como fallback quando scipy não está disponível."""
        n_pred, n_gt = iou_matrix.shape
        matched_pred = []
        matched_gt = []
        used_pred = set()
        used_gt = set()

        # Ordenar todos os pares por IoU decrescente
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
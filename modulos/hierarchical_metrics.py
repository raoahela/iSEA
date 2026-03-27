import yaml
from pathlib import Path
import json
from typing import List, Dict, Tuple

class HierarchicalValidator:
    
    RANK_SCORES = {
        'species': 1.0, 'genus': 0.85, 'family': 0.7,
        'order': 0.55, 'class': 0.4, 'phylum': 0.25, 'kingdom': 0.1
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
            
            # Reconstroi hierarquia aproximada
            # Nota: isso é uma simplificação, o CSV tem dados mais completos
            hierarchy.append({
                "name": name,
                "rank": rank,
                "aphia_id": tax_info.get('aphia_id')
            })
            
            # Adiciona pais se disponíveis
            for parent in parents[:2]:  # Limita a 2 níveis superiores
                # Estima o rank (simplificado)
                hierarchy.append({
                    "name": parent.strip(),
                    "rank": "unknown", 
                    "aphia_id": None
                })
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
    
    def get_hierarchy(self, taxon: str) -> List[Dict]:
        """Busca hierarquia: primeiro no dataset local, depois no cache global"""
        # Se taxon é um ID numérico (string), converte
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
        
        # Fallback: busca pelo nome no cache
        if class_name:
            return self.get_hierarchy(class_name)
        
        return [{"name": str(class_id), "rank": "unknown"}]
    
    def calculate_score(self, pred: str, gt: str, pred_id: int = None, gt_id: int = None) -> Tuple[float, str]:
        """
        Calcula score hierárquico.
        Pode receber nomes (str) ou IDs (int) das classes.
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
        
        # Match exato
        if pred == gt or (pred_names and gt_names and pred_names[0] == gt_names[0]):
            return 1.0, "exact"
        
        # Predição é ancestral do GT
        if pred in gt_names:
            for node in gt_hier:
                if node["name"] == pred:
                    score = self.RANK_SCORES.get(node["rank"], 0.0)
                    return score, f"ancestor_{node['rank']}"
        
        # GT é ancestral da predição (erro por excesso de especificidade)
        if gt in pred_names:
            for node in pred_hier:
                if node["name"] == gt:
                    score = self.RANK_SCORES.get(node["rank"], 0.0) * 0.9  # Penalidade leve
                    return score, f"descendant_{node['rank']}"
        
        # Ancestral comum
        common = set(pred_names) & set(gt_names)
        if common:
            # Pega o ancestral comum mais próximo (maior score)
            best_score = 0
            best_rank = "unknown"
            for node in pred_hier:
                if node["name"] in common:
                    score = self.RANK_SCORES.get(node["rank"], 0.0)
                    if score > best_score:
                        best_score = score
                        best_rank = node["rank"]
            
            return best_score * 0.5, f"common_{best_rank}"  # 0.5 peso para ancestral comum
        
        return 0.0, "unrelated"
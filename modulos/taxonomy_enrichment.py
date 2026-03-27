from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QProgressBar, QTextEdit, QMessageBox
)
from PyQt6.QtCore import QThread, pyqtSignal
import csv
import json
import requests
import time
from pathlib import Path
from dataclasses import dataclass
import yaml  
from collections import defaultdict 
import shutil 
from datetime import datetime  


@dataclass
class TaxonNode:
    """Single node in taxonomic hierarchy"""
    name: str
    rank: str
    aphia_id: int = None


class WoRMSCache:
    """Manages local cache of WoRMS taxonomic data"""
    
    def __init__(self, cache_file="worms_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache = {}
        self.api_calls = 0
        self.cache_hits = 0
        self._load_cache()
    
    def _load_cache(self):
        """Load existing cache from disk"""
        if self.cache_file.exists():
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                self.cache = json.load(f)
    
    def save(self):
        """Save cache to disk for future offline use"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)
    
    def get_hierarchy(self, taxon_name):
        """Get hierarchy for taxon (cache or API)"""
        if taxon_name in self.cache:
            self.cache_hits += 1
            return self.cache[taxon_name]
        
        hierarchy = self._fetch_from_worms(taxon_name)
        self.cache[taxon_name] = hierarchy
        self.api_calls += 1
        return hierarchy
    
    def _fetch_from_worms(self, taxon_name):
       # Estratégia 1: Busca exata primeiro
        try:
            import urllib.parse
            encoded_name = urllib.parse.quote(taxon_name.strip())
            
            # Tenta AphiaRecordsByMatchNames primeiro (mais preciso)
            response = requests.get(
                f"https://www.marinespecies.org/rest/AphiaRecordsByMatchNames",
                params={'scientificnames[]': taxon_name},
                timeout=15
            )
            
            if response.status_code == 200:
                records = response.json()
                if records and records[0] and records[0][0]:
                    aphia_id = records[0][0].get('AphiaID')
                else:
                    # Fallback para AphiaRecordsByName
                    response = requests.get(
                        f"https://www.marinespecies.org/rest/AphiaRecordsByName/{encoded_name}",
                        params={'marine_only': 'false', 'like': 'false'},  # like=false para match exato
                        timeout=15
                    )
                    if response.status_code == 200:
                        records = response.json()
                        if records and isinstance(records, list):
                            aphia_id = records[0].get('AphiaID')
                        else:
                            return self._create_unknown(taxon_name, "No match found")
                    else:
                        return self._create_unknown(taxon_name, f"API error: {response.status_code}")
            else:
                return self._create_unknown(taxon_name, f"API error: {response.status_code}")
            
            if not aphia_id:
                return self._create_unknown(taxon_name, "No AphiaID found")
            
            # Busca classificação
            class_response = requests.get(
                f"https://www.marinespecies.org/rest/AphiaClassificationByAphiaID/{aphia_id}",
                timeout=15
            )
            
            if class_response.status_code == 200:
                data = class_response.json()
                if data:
                    return self._parse_classification(data, aphia_id)
            
            # Se não achou classificação, retorna pelo menos o básico
            return [{
                "name": taxon_name,
                "rank": records[0].get('rank', 'unknown').lower(),
                "aphia_id": aphia_id
            }]
            
        except requests.exceptions.Timeout:
            return self._create_unknown(taxon_name, "Timeout")
        except Exception as e:
            return self._create_unknown(taxon_name, f"Error: {str(e)}")

    def _create_unknown(self, taxon_name, reason):
        """Cria entrada unknown com log de motivo"""
        print(f"⚠️ WoRMS lookup failed for '{taxon_name}': {reason}")
        return [{"name": taxon_name, "rank": "unknown", "aphia_id": None, "reason": reason}]
    
    def _parse_classification(self, data, aphia_id):
        """Parse WoRMS response into hierarchy list"""
        hierarchy = []
        
        def traverse(node, path=None):
            if path is None:
                path = []
            
            current = {
                'name': node.get('scientificname', 'unknown'),
                'rank': node.get('rank', 'unknown').lower(),
                'aphia_id': node.get('AphiaID')
            }
            current_path = path + [current]
            
            child = node.get('child')
            if child:
                traverse(child, current_path)
            else:
                hierarchy.extend(reversed(current_path))
        
        traverse(data)
        return hierarchy
    
    def get_lineage(self, taxon_name):
        """Get display lineage string"""
        hier = self.get_hierarchy(taxon_name)
        if len(hier) > 1:
            main = hier[0]['name']
            parents = [n['name'] for n in hier[1:3]]
            return f"{main} ({', '.join(parents)})"
        return taxon_name

class YOLODatasetEnricher: #for yolo datasets  
    def __init__(self, dataset_dir: str, cache: 'WoRMSCache'):
        self.dataset_dir = Path(dataset_dir)
        self.cache = cache
        self.yaml_path = self.dataset_dir / "dataset.yaml"
        self.classes = {}  # id -> name
        self.taxons_found = defaultdict(int)
        
    def load_dataset(self):
        """Load dataset.yaml and extract class names"""
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"dataset.yaml não encontrado em {self.dataset_dir}")
            
        with open(self.yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        names = data.get('names', {})
        if isinstance(names, dict):
            self.classes = {int(k): v for k, v in names.items()}
        elif isinstance(names, list):
            self.classes = {i: name for i, name in enumerate(names)}
            
        # Scan label files
        labels_dir = self.dataset_dir / "labels"
        if labels_dir.exists():
            self._scan_labels(labels_dir)
            
        return list(self.taxons_found.keys())
    
    def _scan_labels(self, labels_dir: Path):
        """Scan .txt files to count taxon occurrences"""
        for txt_file in labels_dir.rglob("*.txt"):
            with open(txt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        if class_id in self.classes:
                            taxon = self.classes[class_id]
                            self.taxons_found[taxon] += 1
    
    def enrich_and_save(self, output_file: str = None) -> dict:
        """Enrich classes with WoRMS data"""
        if output_file is None:
            output_file = self.dataset_dir / "taxonomy_metadata.csv"
            
        unique_taxons = sorted(set(self.classes.values()))
        
        # Query WoRMS
        enrichment_data = {}
        for taxon in unique_taxons:
            hierarchy = self.cache.get_hierarchy(taxon)
            lineage = self.cache.get_lineage(taxon)
            rank = hierarchy[0]['rank'] if hierarchy else 'unknown'
            
            enrichment_data[taxon] = {
                'original_name': taxon,
                'lineage_display': lineage,
                'rank': rank,
                'aphia_id': hierarchy[0].get('aphia_id') if hierarchy else None,
                'full_hierarchy': hierarchy,
                'usage_count': self.taxons_found.get(taxon, 0)
            }
        
        # Save CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'class_id', 'original_name', 'rank', 'aphia_id',
                'lineage_display', 'kingdom', 'phylum', 'class', 'order',
                'family', 'genus', 'species', 'usage_count'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for class_id, taxon in sorted(self.classes.items()):
                data = enrichment_data.get(taxon, {})
                hier = data.get('full_hierarchy', [])
                rank_data = {item['rank']: item['name'] for item in hier}
                
                writer.writerow({
                    'class_id': class_id,
                    'original_name': taxon,
                    'rank': data.get('rank', 'unknown'),
                    'aphia_id': data.get('aphia_id'),
                    'lineage_display': data.get('lineage_display', taxon),
                    'kingdom': rank_data.get('kingdom', ''),
                    'phylum': rank_data.get('phylum', ''),
                    'class': rank_data.get('class', ''),
                    'order': rank_data.get('order', ''),
                    'family': rank_data.get('family', ''),
                    'genus': rank_data.get('genus', ''),
                    'species': rank_data.get('species', ''),
                    'usage_count': data.get('usage_count', 0)
                })
        
        # Update dataset.yaml with taxonomy section
        self._update_yaml_with_taxonomy(enrichment_data)
        
        return {
            'total_classes': len(unique_taxons),
            'enriched_file': str(output_file)
        }
    
    def _update_yaml_with_taxonomy(self, enrichment_data: dict):
        """Add taxonomy section to dataset.yaml"""
        with open(self.yaml_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f) or {}
        
        yaml_data['taxonomy'] = {
            'source': 'WoRMS (World Register of Marine Species)',
            'enrichment_date': datetime.now().isoformat(),
            'classes': {}
        }
        
        for class_id, taxon in self.classes.items():
            data = enrichment_data.get(taxon, {})
            yaml_data['taxonomy']['classes'][int(class_id)] = {
                'name': taxon,
                'rank': data.get('rank', 'unknown'),
                'lineage': data.get('lineage_display', taxon),
                'aphia_id': data.get('aphia_id')
            }
        
        # Backup original
        backup_path = self.yaml_path.with_suffix('.yaml.backup')
        shutil.copy2(self.yaml_path, backup_path)
        
        with open(self.yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, allow_unicode=True, sort_keys=False)

class EnrichmentWorker(QThread):
    """Background worker for enrichment (non-blocking GUI)"""
    progress = pyqtSignal(int)      # 0-100
    status = pyqtSignal(str)        # Status message
    finished_signal = pyqtSignal(bool, str)  # Success, message
    
    def __init__(self, input_file, output_file, cache_file="worms_cache.json"):
        super().__init__()
        self.input_file = input_file
        self.output_file = output_file
        self.cache = WoRMSCache(cache_file)
        self.should_stop = False
    
    def run(self):
        try:
            # Read CSV
            self.status.emit("Reading annotations...")
            rows = self._read_csv()
            
            # Get unique taxa
            unique_taxa = set()
            for row in rows:
                # Tenta 'Taxon' primeiro (seu CSV), depois outras variantes
                taxon = (row.get('Taxon') or 
                        row.get('taxon') or 
                        row.get('class') or 
                        row.get('Class') or 
                        row.get('label') or 
                        'unknown')
                
                unique_taxa.add(taxon)
                
            total = len(unique_taxa)
            
            # Fetch from WoRMS
            for i, taxon in enumerate(sorted(unique_taxa)):
                if self.should_stop:
                    self.finished_signal.emit(False, "Cancelled")
                    return
                
                self.status.emit(f"Fetching '{taxon}'...")
                self.cache.get_hierarchy(taxon)
                
                progress = int((i + 1) / total * 50)  # 0-50% for fetching
                self.progress.emit(progress)
                
                time.sleep(0.5)  # Rate limiting
            
            # Save enriched data
            self.status.emit("Saving enriched dataset...")
            self._save_enriched(rows)
            self.progress.emit(100)
            
            self.cache.save()
            
            self.finished_signal.emit(
                True, 
                f"Enriched {len(rows)} annotations with {len(unique_taxa)} taxa"
            )
            
        except Exception as e:
            self.finished_signal.emit(False, str(e))
    
    def _read_csv(self):
        with open(self.input_file, 'r', encoding='utf-8') as f:
            return list(csv.DictReader(f))
    
    def _save_enriched(self, rows):
        # Add new columns
        fieldnames = list(rows[0].keys()) + ['lineage_display', 'taxon_rank']
        
        with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in rows:
                taxon = (row.get('Taxon') or 
                    row.get('taxon') or 
                    row.get('class') or 
                    row.get('Class') or 
                    row.get('label') or 
                    'unknown')
                
                row['lineage_display'] = self.cache.get_lineage(taxon)
                row['taxon_rank'] = self.cache.get_hierarchy(taxon)[0]['rank']
                
                writer.writerow(row)
    
    def stop(self):
        self.should_stop = True

class YOLOEnrichmentWorker(QThread): #for yolo datasets
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self, dataset_dir: str, output_file: str, cache_file="worms_cache.json"):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.output_file = output_file
        self.cache = WoRMSCache(cache_file)
        self.should_stop = False
        
    def run(self):
        try:
            self.status.emit("Carregando dataset YOLO...")
            enricher = YOLODatasetEnricher(self.dataset_dir, self.cache)
            unique_taxons = enricher.load_dataset()
            
            total = len(unique_taxons)
            if total == 0:
                self.finished_signal.emit(False, "Nenhum taxon encontrado")
                return
            
            # Fetch from WoRMS
            for i, taxon in enumerate(sorted(unique_taxons)):
                if self.should_stop:
                    self.finished_signal.emit(False, "Cancelado")
                    return
                
                self.status.emit(f"Consultando WoRMS: '{taxon}'...")
                self.cache.get_hierarchy(taxon)
                
                progress = int((i + 1) / total * 80)
                self.progress.emit(progress)
                time.sleep(0.5)
            
            self.status.emit("Gerando metadados...")
            self.progress.emit(90)
            
            stats = enricher.enrich_and_save(self.output_file)
            self.cache.save()
            
            self.progress.emit(100)
            
            msg = (f"Dataset enriquecido!\n"
                   f"• {stats['total_classes']} classes processadas\n"
                   f"• Metadados: {stats['enriched_file']}\n"
                   f"• dataset.yaml atualizado com seção 'taxonomy'")
            
            self.finished_signal.emit(True, msg)
            
        except Exception as e:
            self.finished_signal.emit(False, str(e))
            
    def stop(self):
        self.should_stop = True

class TaxonomyEnrichmentDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enriquecer com Taxonomia WoRMS")
        self.resize(600, 450)
        self.worker = None
        
        self._build_ui()
    
    def _build_ui(self):
        layout = QVBoxLayout(self)
        
        # Info text
        info = QLabel(
            "Enriquece anotações com hierarquia taxonômica do WoRMS "
            "(World Register of Marine Species).\n\n"
            "Requer conexão com internet. O cache local será atualizado "
            "para uso futuro offline no ROV."
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_label = QLabel("Arquivo:")
        self.file_path = QLabel("(nenhum selecionado)")
        self.file_path.setStyleSheet("color: gray; font-style: italic;")
        
        browse_btn = QPushButton("Selecionar CSV...")
        browse_btn.clicked.connect(self._select_file)

        dataset_btn = QPushButton("Selecionar Dataset YOLO...") 
        dataset_btn.clicked.connect(self._select_dataset)

        test_btn = QPushButton("Testar")
        test_btn.clicked.connect(self._test_source)
        
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_path, 1)
        file_layout.addWidget(browse_btn)
        file_layout.addWidget(dataset_btn) 
        file_layout.addWidget(test_btn)
        layout.addLayout(file_layout)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.status_text = QLabel("")
        self.status_text.setVisible(False)
        layout.addWidget(self.status_text)
        
        # Log area
        self.log_area = QTextEdit()
        self.log_area.setVisible(False)
        self.log_area.setMaximumHeight(100)
        layout.addWidget(self.log_area)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Iniciar Enriquecimento")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start)
        
        cancel_btn = QPushButton("Cancelar")
        cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def _find_taxon_column(self, fieldnames):
        possible_names = [
            'Taxon', 'taxon', 'class', 'Class', 
            'label', 'species', 'Species', 'name',
            'nome', 'cientifico', 'scientific_name'
        ]
        
        for name in possible_names:
            if name in fieldnames:
                return name
        
        # Se não encontrar, usa a primeira coluna
        return fieldnames[0] if fieldnames else None
    
    def _select_dataset(self): #for yolo datasets
        path = QFileDialog.getExistingDirectory(
            self, "Selecionar pasta do Dataset YOLO", ""
        )
        
        if path:
            yaml_path = Path(path) / "dataset.yaml"
            if not yaml_path.exists():
                QMessageBox.warning(
                    self, "Dataset Inválido", 
                    "A pasta não contém dataset.yaml"
                )
                return
                
            self.input_file = path
            self.input_type = 'yolo'  # Flag para diferenciar
            self.file_path.setText(f"[YOLO] {Path(path).name}")
            self.file_path.setStyleSheet("color: black; font-style: normal;")
            self.start_btn.setEnabled(True)
    
    def _select_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Selecionar anotações CSV", "", "CSV Files (*.csv)"
        )
        
        if path:
            self.input_file = path
            self.input_type = 'csv'  # NOVO: Flag
            self.file_path.setText(Path(path).name)
            self.file_path.setStyleSheet("color: black; font-style: normal;")
            self.start_btn.setEnabled(True)

    def _test_source(self):
        if not hasattr(self, 'input_type'):
            QMessageBox.warning(self, "Aviso", "Selecione uma fonte primeiro.")
            return
            
        if self.input_type == 'csv':
            self._test_csv()
        else:
            self._test_dataset()
    
    def _test_csv(self):  
        # Verifica se um arquivo foi selecionado
        if not hasattr(self, 'input_file') or not self.input_file:
            QMessageBox.warning(
                self, 
                "Aviso", 
                "Por favor, selecione um arquivo CSV primeiro."
            )
            return
        
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                
                # Tenta ler a primeira linha
                try:
                    first_row = next(reader)
                except StopIteration:
                    # Arquivo vazio
                    QMessageBox.warning(
                        self,
                        "Arquivo vazio",
                        "O arquivo CSV está vazio."
                    )
                    return
                
                # Mostra a área de log
                self.log_area.setVisible(True)
                self.log_area.clear()
                
                # Formata a saída
                self.log_area.append("=" * 50)
                self.log_area.append("📋 ESTRUTURA DO CSV")
                self.log_area.append("=" * 50)
                self.log_area.append("")
                
                # Headers
                self.log_area.append("📌 CABEÇALHOS ENCONTRADOS:")
                for i, header in enumerate(headers, 1):
                    self.log_area.append(f"  {i}. '{header}'")
                
                self.log_area.append("")
                self.log_area.append("-" * 30)
                self.log_area.append("")
                
                # Primeira linha
                self.log_area.append("📄 PRIMEIRA LINHA (AMOSTRA):")
                for key, value in first_row.items():
                    # Trunca valores muito longos
                    if len(str(value)) > 50:
                        display_value = str(value)[:47] + "..."
                    else:
                        display_value = value if value else "(vazio)"
                    
                    self.log_area.append(f"  📍 {key}: {display_value}")
                
                self.log_area.append("")
                self.log_area.append("-" * 30)
                self.log_area.append("")
                
                # Identifica coluna de taxonomia
                taxon_col = self._find_taxon_column(headers)
                
                self.log_area.append("🔍 ANÁLISE DE TAXONOMIA:")
                self.log_area.append(f"  Coluna identificada: '{taxon_col}'")
                
                if taxon_col in first_row:
                    taxon_value = first_row[taxon_col]
                    self.log_area.append(f"  Valor encontrado: '{taxon_value}'")
                else:
                    self.log_area.append("  ⚠️ Coluna de taxonomia não encontrada nos dados")
                
                self.log_area.append("")
                self.log_area.append("=" * 50)
                self.log_area.append("✅ Teste concluído!")
                
                # Atualiza status
                self.status_text.setText(f"Teste concluído. {len(headers)} colunas encontradas.")
                self.status_text.setVisible(True)
                
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Erro", 
                f"Falha ao ler CSV:\n\n{str(e)}"
            )

    def _test_dataset(self): #for yolo datasets
        try:
            enricher = YOLODatasetEnricher(self.input_file, WoRMSCache())
            classes = enricher.load_dataset()
            
            self.log_area.setVisible(True)
            self.log_area.clear()
            self.log_area.append("=" * 50)
            self.log_area.append("📋 DATASET YOLO DETECTADO")
            self.log_area.append("=" * 50)
            self.log_area.append(f"\n📁 Pasta: {self.input_file}")
            self.log_area.append(f"📊 Classes: {len(enricher.classes)}")
            self.log_area.append(f"🐛 Taxons em labels: {len(classes)}")
            self.log_area.append("\n📌 MAPEAMENTO:")
            
            for cid, name in sorted(enricher.classes.items()):
                count = enricher.taxons_found.get(name, 0)
                self.log_area.append(f"  ID {cid}: {name} ({count} amostras)")
            
            self.status_text.setText(f"Dataset válido: {len(enricher.classes)} classes")
            self.status_text.setVisible(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Falha ao ler dataset:\n{str(e)}")
            
    def _start(self):
        # Define output e worker baseado no tipo
        if not hasattr(self, 'input_type') or self.input_type == 'csv':
            # Comportamento CSV original
            base = Path(self.input_file).stem
            self.output_file = f"{base}_enriched.csv"
            self.worker = EnrichmentWorker(self.input_file, self.output_file)
        else:
            # Modo YOLO
            self.output_file = str(Path(self.input_file) / "taxonomy_metadata.csv")
            self.worker = YOLOEnrichmentWorker(self.input_file, self.output_file)
        
        # UI progress (código existente continua igual...)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_text.setVisible(True)
        self.log_area.setVisible(True)
        self.start_btn.setEnabled(False)
        
        # Conectar sinais (código existente)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self._update_status)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.start()
    
    def _update_status(self, message):
        self.status_text.setText(message)
        self.log_area.append(message)
    
    def _on_finished(self, success, message):
        self.progress_bar.setVisible(False)
        
        if success:
            QMessageBox.information(
                self,
                "Sucesso",
                f"Enriquecimento completo!\n\n"
                f"Arquivo salvo: {self.output_file}\n"
                f"Cache atualizado: worms_cache.json\n\n"
                f"{message}"
            )
            self.accept()
        else:
            QMessageBox.critical(self, "Erro", f"Falha: {message}")
            self.start_btn.setEnabled(True)
    
    def reject(self):
        """Cancel and cleanup"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1000)
        super().reject()



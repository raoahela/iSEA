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
        """Fetch from WoRMS REST API"""
        try:
            # Get AphiaID
            response = requests.get(
                f"https://www.marinespecies.org/rest/AphiaIDByName/{taxon_name}",
                params={'marine_only': 'false'},
                timeout=10
            )
            
            if response.status_code != 200:
                return [{"name": taxon_name, "rank": "unknown", "aphia_id": None}]
            
            aphia_id = response.json()
            if not isinstance(aphia_id, int):
                return [{"name": taxon_name, "rank": "unknown", "aphia_id": None}]
            
            # Get classification
            class_response = requests.get(
                f"https://www.marinespecies.org/rest/AphiaClassificationByAphiaID/{aphia_id}",
                timeout=10
            )
            
            if class_response.status_code != 200:
                return [{"name": taxon_name, "rank": "unknown", "aphia_id": None}]
            
            return self._parse_classification(class_response.json(), aphia_id)
            
        except Exception as e:
            return [{"name": taxon_name, "rank": "unknown", "aphia_id": None}]
    
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
        
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_path, 1)
        file_layout.addWidget(browse_btn)
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
    
    def _select_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar anotações CSV",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if path:
            self.input_file = path
            self.file_path.setText(Path(path).name)
            self.file_path.setStyleSheet("color: black; font-style: normal;")
            self.start_btn.setEnabled(True)
    
    def _start(self):
        # Generate output filename
        base = Path(self.input_file).stem
        self.output_file = f"{base}_enriched.csv"
        
        # Show progress UI
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_text.setVisible(True)
        self.log_area.setVisible(True)
        self.start_btn.setEnabled(False)
        
        # Start worker thread
        self.worker = EnrichmentWorker(self.input_file, self.output_file)
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



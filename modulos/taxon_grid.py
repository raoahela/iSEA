from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (QPushButton, QVBoxLayout, QScrollArea, 
                            QHBoxLayout, QWidget,  QMessageBox, 
                            QInputDialog,  QGridLayout, QToolButton)
from .translations import TEXTS

class TaxonGrid(QWidget):
    taxon_changed = pyqtSignal(str)  
    title_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_layout = QVBoxLayout(self)
        self.grid_widget = QWidget()
        self.grid = QGridLayout(self.grid_widget)
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.grid_widget)
        self.scroll.setWidgetResizable(True)
        self.main_layout.addWidget(self.scroll)
        self.language = "pt" 
        self.texts = TEXTS[self.language]
        self.dark_mode = False
        self.removal_mode = False

        self.add_btn = QPushButton(self.texts["add_taxon"])
        self.add_btn.clicked.connect(self._add_new_taxon)
        self.main_layout.addWidget(self.add_btn)

        self._buttons = {}   

        top_bar = QHBoxLayout()
        top_bar.addStretch()
        self.trash_btn = QToolButton()
        self.trash_btn.setText("🗑️")
        self.trash_btn.setStyleSheet("""
                    border: 1px solid #555;
                    border-radius: 2px;
                    padding: 4px;
            """)
        self.trash_btn.setFixedSize(28, 28)
        self.trash_btn.setToolTip(self.texts["remove_selected_taxons"])
        self.trash_btn.clicked.connect(self.remove_selected)
        top_bar.addWidget(self.trash_btn)

        self.main_layout.insertLayout(0, top_bar)


    def update_language(self, lang):
        self.language = lang
        self.texts = TEXTS[lang]  
        self.add_btn.setText(self.texts["add_taxon"]) 
        self.title_changed.emit(self.texts["taxons"])

    def clear(self):
        """Remove all buttons from the grid"""
        for btn in list(self._buttons.values()):
            self.grid.removeWidget(btn)
        self._buttons.clear()

    def populate(self, classes):
        """Receives list of strings and make the buttons of the taxon grid"""
        for name in sorted(classes, key=str.lower):
            self.insert_button(name)

    def add_taxon(self, name):
        """Inserts new taxon (if it does not already exist)"""
        if name and name not in self._buttons:
            self.insert_button(name)

    def insert_button(self, name):
        btn = QPushButton(name)
        btn.setCheckable(True) 
        btn.setChecked(False)    
        if self.dark_mode:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #2e3136;
                    color: white;
                    border: 1px solid #555;
                    border-radius: 2px;
                    padding: 4px;
                }
                QPushButton:checked {
                    background-color: #4a8ad4;
                }
            """)
        else:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #f0f0f0;
                    border: 1px solid #555;
                    border-radius: 2px;
                    padding: 4px;
                }
                QPushButton:checked {
                    background-color: #ccc;
                }
            """)
        
        btn.clicked.connect(lambda _, n=name: self.select(n))
        row = len(self._buttons) // 3
        col = len(self._buttons) % 3
        self.grid.addWidget(btn, row, col)
        self._buttons[name] = btn

    def set_dark_mode(self, enable=True):
        self.dark_mode = enable
        
        for btn in self._buttons.values():
            if enable:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #2e3136;
                        color: white;
                        border: 1px solid #555;
                        border-radius: 2px;
                        padding: 4px;
                    }
                    QPushButton:checked {
                        background-color: #4a8ad4;
                    }
                """)

                self.add_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #2e3136;
                        color: white;
                        border: 1px solid #555;
                        border-radius: 2px;
                        padding: 4px;
                    }
                    QPushButton:hover {
                        background-color: #4a8ad4;
                    }
                """)
            else:
                btn.setStyleSheet("""
                QPushButton {
                    background-color: #f0f0f0;
                    border: 1px solid #555;
                    border-radius: 2px;
                    padding: 4px;
                }
                QPushButton:checked {
                    background-color: #ccc;
                }
            """)
                
                self.add_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #f0f0f0;
                        border: 1px solid #555;
                        border-radius: 2px;
                        padding: 4px;
                    }
                    QPushButton:hover {
                        background-color: #e0e0e0;
                    }
                """)
                
    def select(self, name):
        if self.removal_mode:
            return  
        for n, btn in self._buttons.items():
            btn.setChecked(n == name)
        self.taxon_changed.emit(name) 

    def _add_new_taxon(self):
        name, ok = QInputDialog.getText(self, self.texts["new_taxon"], self.texts["taxon_name"])
        if ok and name:
            self.add_taxon(name)
            self.select(name)   

    def remove_selected(self):
        if not self.removal_mode:
            self.enter_removal_mode()
            return

        selected = [name for name, btn in self._buttons.items() if btn.isChecked()]

        if not selected:
            return

        reply = QMessageBox.question(
            self,
            self.texts["confirm_remove_title"],
            self.texts["confirm_remove_multiple"].format(len(selected)),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            for name in selected:
                self.remove_taxon(name)
        self.exit_removal_mode()
    
    def remove_taxon(self, name):
        btn = self._buttons.pop(name, None)
        if btn:
            self.grid.removeWidget(btn)
        self.rebuild_grid()

    def rebuild_grid(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                pass

        names = list(self._buttons.keys())
        for index, name in enumerate(names):
            row = index // 3
            col = index % 3
            self.grid.addWidget(self._buttons[name], row, col)

    def enter_removal_mode(self):
        self.removal_mode = True
        self.trash_btn.setStyleSheet("""background-color: #ff5c5c; border: 1px solid #555;
                        border-radius: 2px;
                        padding: 4px;""") 
        for btn in self._buttons.values():
            btn.setChecked(False)
            btn.setCheckable(True)          

    def exit_removal_mode(self):
        self.removal_mode = False
        self.trash_btn.setStyleSheet("""border: 1px solid #555;
                        border-radius: 2px;
                        padding: 4px;""")   
        

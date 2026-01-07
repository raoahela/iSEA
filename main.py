import sys
import os
from PyQt6.QtWidgets import QApplication
from modulos.video_annotator import VideoAnnotator

def resource_path(relative_path):
    """Obtém caminho correto para arquivos no executável"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path).replace("/", os.sep)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoAnnotator()
    window.show()
    sys.exit(app.exec())
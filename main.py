# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026  R. N. Lopes

import sys
import os
from PyQt6.QtWidgets import QApplication
from modulos.video_annotator import VideoAnnotator
from PyQt6.QtGui import QIcon
from ctypes import windll

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path).replace("/", os.sep)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(resource_path("icons/iSEA_icon.png")))
    try:
        myappid = 'isea.annotator.v1.0' 
        windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except ImportError:
        pass 
    window = VideoAnnotator()
    window.show()
    sys.exit(app.exec())
    

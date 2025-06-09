from pathlib import Path
import os
import sys
from PyQt5.QtWidgets import QApplication
from gui.components import pyqt_main

log_path = Path('gui_validation.log')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
try:
    app = QApplication(sys.argv)
    win = pyqt_main.VoxSigilMainWindow()
    status = 'success'
    app.quit()
except Exception as e:
    status = f'error: {e}'

log_path.write_text(f'GUI load status: {status}\n')
print(f'Logged GUI validation: {status}')

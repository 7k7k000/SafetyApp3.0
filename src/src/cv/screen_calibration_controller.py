from typing import Optional
from PySide6.QtCore import Signal, Qt, QObject

class ScreenCalibrationController(QObject):
    def __init__(self) -> None:
        super().__init__()
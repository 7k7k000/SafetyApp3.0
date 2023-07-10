import time
import numpy as np

from src.src.utils.Async import QAsyncWorkerThread



def print_debug(e):
    print(e)


if __name__ == "__main__":
    from src.src.sensors.SensorHub import SensorHub
    from PySide6.QtWidgets import (
        QWidget,
        QPushButton,
        QApplication,
        QVBoxLayout,
        QMainWindow,
    )
    import sys

    app = QApplication(sys.argv)
    hub = SensorHub()
    thread = QAsyncWorkerThread()
    hub.moveToThread(thread)
    hub.sigSensorMsg.connect(print_debug)
    thread.start()
    window = QMainWindow()
    window.setCentralWidget(QWidget())
    window.centralWidget().setLayout(QVBoxLayout())

    window.centralWidget().layout().addWidget(QPushButton(text="Start", clicked=hub.start_scanning))  # type: ignore
    window.centralWidget().layout().addWidget(QPushButton(text="Stop", clicked=hub.stop_scanning))  # type: ignore
    window.centralWidget().layout().addWidget(QPushButton(text="Test", clicked=hub.test))  # type: ignore
    # hub.start_scanning()
    window.show()
    app.exec()



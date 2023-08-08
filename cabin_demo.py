from typing import Optional
from src.ui.views.faceview import FaceSettingView
from PySide6.QtWidgets import (
    QWidget,
    QPushButton,
    QApplication,
    QVBoxLayout,
    QMainWindow,
)
from PySide6.QtCore import Qt, QThread, QObject
from src.src.cv.camera_manager import QtDualCamManager
import sys
import qdarktheme


class MainWindow(QObject):
    def __init__(self,) -> None:
        super().__init__()
        self.caman = QtDualCamManager()  
        self.view = FaceSettingView()
        self.link()
        self.t = QThread()
        self.caman.moveToThread(self.t)
        self.t.start()
        self.view.show()
        
    def link(self):
        self.caman.set_destine(self.view.faceLabelUpdate, 0)
        self.caman.set_destine(self.view.cabinLabelUpdate, 1)

        self.view.facecamselect.videoInputsChanged(self.caman.refresh_cam_info(), 0)
        self.view.facecamselect.currentTextChanged.connect(lambda cam: self.caman.start_cam(cam, 0))
        self.view.facecamselect.currentTextChanged.connect(lambda _: self.view.enable_gaze_tracking.setStatus(False))
        self.caman.sigVideoInputsChanged.connect(lambda e: self.view.facecamselect.videoInputsChanged(e, 0))     

        self.view.cabincamselect.videoInputsChanged(self.caman.refresh_cam_info(), 1)
        self.view.cabincamselect.currentTextChanged.connect(lambda cam: self.caman.start_cam(cam, 1))
        self.caman.sigVideoInputsChanged.connect(lambda e: self.view.cabincamselect.videoInputsChanged(e, 1))  
        
        self.view.enable_gaze_tracking.stateChanged.connect(lambda e: self.caman.switch_worker(e, 0))
        self.view.enable_finger_tracking.stateChanged.connect(self.caman.enable_finger_tracking)
        
        self.view.screen_cali_widget.sigScreenRect.connect(self.caman.finger_screen_data)
        self.view.screen_cali_widget.sigWidgetShow.connect(self.caman.finger_screen_show)

        self.caman.sigFingerTrackingPos.connect(self.view.preview_finger_status)


app = QApplication(sys.argv)
qdarktheme.setup_theme()
view = MainWindow()
app.exec()
from typing import Optional
import pyqtgraph as pg
from PySide6.QtCore import Signal, Qt, QTimer, QThread
import pyqtgraph.functions as fn
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtWidgets import (
    QWidget,  QLabel,
    QCheckBox, QComboBox,  QDoubleSpinBox,
    QLCDNumber, QPushButton, QSpinBox, 
    QToolBar, QGroupBox, QSlider,QLineEdit,
    QVBoxLayout, QHBoxLayout, QMessageBox, QSizePolicy
)
from araviq6 import NDArrayLabel
import numpy as np
from datetime import datetime
import time


class VideoLabel(NDArrayLabel):
    pause = False
    def __init__(self):
        super().__init__()
        self.setMinimumSize(500, 300)
        self.pixmap_rect = [0,0, self.width(), self.height()]
        self._referrence_pixmap = QtGui.QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setArray(np.zeros((480,853,3), np.uint8))
    
    def setArray(self, array):
        if not self.pause:
            super().setArray(array)
        
    def setPixmap(self, pixmap: QtGui.QPixmap):
        if pixmap.width() == 0:
            return
        self._referrence_pixmap = pixmap
        self._setPixmap()
        
    def _setPixmap(self):
        """Scale the pixmap and display."""
        
        pixmap = self._referrence_pixmap.copy()
        if pixmap.width() == 0:
            return
        new_w = self.width()
        new_h = self.height()

        x1, x2, y1, y2 = 0, 0, new_w, new_h
        w, h = pixmap.width(), pixmap.height()
        if w == 0 or h == 0:
            pass
        else:
            if h * new_w / w > new_h:
                x1, y1 = (new_w - w * new_h / h) / 2, 0
                x2, y2 = w * new_h / h, new_h
            elif w * new_h / h > new_w:
                x1, y1 = 0, (new_h- h * new_w / w) / 2
                x2, y2 = new_w, h * new_w / w
            
            self.pixmap_rect = [int(i) for i in (x1, y1, x2, y2)]

        pixmap = pixmap.scaled(
            new_w, new_h, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation
        )
        self._original_pixmap = pixmap
        super().setPixmap(pixmap)

class CabinLabel(VideoLabel):
    mode = 0
    def __init__(self):
        super().__init__()
        self.rect_path = []
        
    def setMode(self, mode:int):
        self.mode = mode
        if mode == 1:
            self.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
            
    def mousePressEvent(self, event):
        if self.mode:
            if event.button() == Qt.LeftButton:
                self.pressPos = event.pos()

    def mouseReleaseEvent(self, event):
        # ensure that the left button was pressed *and* released within the
        # geometry of the widget; if so, emit the signal;
        if self.mode:
            if (self.pressPos is not None and 
                event.button() == Qt.LeftButton and len(self.rect_path) < 5):
                    self.get_img_crd(self.pressPos)
            self.pressPos = None

    def reset_selected_pos(self, e):
        self.rect_path = []

    def get_img_crd(self, pos):
        x, y = pos.x(), pos.y()
        x1, y1, x2, y2 = self.pixmap_rect
        w, h = self._original_pixmap.width(), self._original_pixmap.height()
        x0, y0 = x-x1, y-y1
        x, y = x0 / w, y0/ h
        if x > 1 or y > 1 or x < 0 or y < 0:
            return
        
        self.rect_path.append((x, y))
        if len(self.rect_path) == 4:
            print(self.rect_path)
            self.rect_path.append(self.rect_path[0])

class PedalCalibrator(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.brake0 = QPushButton("标定零点", clicked=self.cal0)
        self.brake1 = QPushButton("标定行程", clicked=self.cal1)

class Dial(pg.GraphicsLayoutWidget):
    pos = (0, 0)
    def __init__(self):
        self.r = 120
        super().__init__()
        self.setBackground(QtGui.QColor(32,33,36))
        pg.setConfigOptions(antialias=True)
        p1 = self.addPlot()
        p1.setMenuEnabled(False)
        p1.setMouseEnabled(x=False, y=False)
        p1.hideAxis('bottom')
        p1.hideAxis('left')
        p1.setAspectLocked()
        p_ellipse = QtWidgets.QGraphicsEllipseItem(-self.r,-self.r,2*self.r,2*self.r)
        p_ellipse.setPen(pg.mkPen((231,246,242), width=3))
        p_ellipse.setBrush(pg.mkBrush((27,36,48)))
        p1.addItem(p_ellipse)
        p1.plot([-self.r * 1.3, self.r * 1.3],[0,0], pen=pg.mkPen((231,246,242), width=3))
        p1.plot([0,0],[-self.r * 1.3, self.r * 1.3], pen=pg.mkPen((231,246,242), width=3))
        # self.target_line = pg.PlotDataItem([0,0],[500,500], pen=pg.mkPen((231,246,242), width=3, connect='all'))
        # p1.addItem(self.target_line)
        self.rr = 20
        self.target = QtWidgets.QGraphicsEllipseItem(self.pos[0]-self.rr, self.pos[1]-self.rr, 2*self.rr, 2*self.rr)
        self.target.setPen(pg.mkPen((230,210,170), width=6))
        self.target.setBrush(pg.mkBrush((255,75,75)))
        p1.addItem(self.target)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
            # QtWidgets.QSizePolicy.Fixed
        )

    def sizeHint(self):
        return QtCore.QSize(200,200)

    def rotate(self, deg):
        deg = np.radians(float(deg))
        x = np.sin(deg) * self.r
        y = np.cos(deg) * self.r
        self.target.setPos(x, y)

    def move(self, vec):
        #vec shall be betwenn (-1, 1)
        for i in vec:
            if i > 1:
                i = 1
        self.target.setPos(vec[0]*self.r, vec[1]*self.r)

        # self.target.setPos(-float(vec[0])*10, -float(vec[1])*10)
        
class LineWidget(QWidget):
    def __init__(self, text, widget, text1="", width = 250):
        super().__init__()
        self.setContentsMargins(0,0,0,0)
        self.multiline = True
        self.text = text
        if text1 == "":
            self.multiline = False
        self.label = QLabel(f"<strong>{text}</strong>{text1}")

        self.widget = widget
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.label)
        layout.addWidget(self.widget)
        self.setLayout(layout)
        self.setMaximumWidth(width)

    def setText(self, s):
        if self.multiline:
            self.label.setText(f"<strong>{self.text}</strong>{s}")
        else:
            self.label.setText(f"<strong>{s}</strong>")
        pass

class CamSelectWidget(QtCore.QObject):
    currentTextChanged = Signal(object)
    def __init__(self):
        super().__init__()
        self.camselect = QComboBox()
        self.camselect.setMinimumWidth(140)
        self.img = LineWidget("选择实时视频来源", self.camselect)
        
        self.inputs = []
        self.freeze = False
        self.camselect.currentTextChanged.connect(self.handle_text)

    def getWidget(self):
        return self.img
    
    def videoInputsChanged(self, e, destine_index):
        cam_list, current_cams = e
        currentText = self.camselect.currentText()
        other_cam_index = abs(destine_index-1)
        self.all_cams = {f"{i+1}: {cam.description()}":cam for i, cam in enumerate(cam_list)}
        if current_cams[other_cam_index] is not None:
            inputs = [key for key, value in self.all_cams.items() if value.id() != current_cams[other_cam_index].id()]
        else:
            inputs = [key for key, value in self.all_cams.items()]
            
        if self.inputs != inputs:
            self.freeze = True
            self.inputs = inputs
            self.camselect.clear()
            self.camselect.addItem('无')
            self.camselect.addItems(inputs)
            self.camselect.setCurrentText(currentText)
            self.freeze = False
            
    def handle_text(self, t):
        if not self.freeze:
            if t != "无":
                cam = self.all_cams[t]
            else:
                cam = None
            self.currentTextChanged.emit(cam)

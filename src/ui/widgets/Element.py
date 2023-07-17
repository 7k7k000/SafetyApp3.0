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
import numpy as np
from datetime import datetime
import time

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
    def __init__(self, text, widget, text1=""):
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

    def setText(self, s):
        if self.multiline:
            self.label.setText(f"<strong>{self.text}</strong>{s}")
        else:
            self.label.setText(f"<strong>{s}</strong>")
        pass
    

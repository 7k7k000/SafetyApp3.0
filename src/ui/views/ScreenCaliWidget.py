from src.ui.widgets.Element import XYInputWidget
from src.ui.widgets.Element import LineWidget


import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QPushButton, QWidget


class ScreenCaliWidget(QWidget):
    showing = False
    sigScreenRect = Signal(tuple)
    sigScreenReset = Signal()
    sigCloseEvent = Signal()
    manual_enable = False
    SCREEN_RECT = (148, 236)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self._resetAreaSelectButton = QPushButton("重置屏幕区域", clicked=self.on_reset_screen)
        self._screen_area_input = XYInputWidget(prompt=("宽:", "高:"))
        self._pt0_input = XYInputWidget()
        self._pt1_input = XYInputWidget()
        self._pt2_input = XYInputWidget()
        self._pt3_input = XYInputWidget()

        for i in (self._pt0_input,
                self._pt1_input,
                self._pt2_input,
                self._pt3_input):
            i.sigPosChanged.connect(self._on_pos_maunally_changed)
            i.setDisabled(True)
        self._screen_area_input.sigPosChanged.connect(self._on_pos_maunally_changed)

        # self._screen_area_input.sigPosChanged.connect(self._on_screen_res_manually_changed)
        self._screen_area_input.set_xy_bound(1000,1000)
        self._screen_area_input.update_crd(self.SCREEN_RECT)


        ctrl_layout = QGridLayout()
        llll = QHBoxLayout()
        llll.addStretch()
        llll.addWidget(self._resetAreaSelectButton)
        llll.addStretch()

        ctrl_layout.addLayout(llll,3,0,1,2)
        linew = LineWidget("屏幕尺寸(mm):", self._screen_area_input, width=350)
        lll = QHBoxLayout()
        lll.addStretch()
        lll.addWidget(linew)
        lll.addStretch()
        ctrl_layout.addLayout(lll,0,0,1,2)
        ctrl_layout.addWidget(LineWidget("左上角(px):", self._pt0_input, width=300),1,0)
        ctrl_layout.addWidget(LineWidget("左下角(px):", self._pt3_input, width=300),2,0)
        ctrl_layout.addWidget(LineWidget("右上角(px):", self._pt1_input, width=300),1,1)
        ctrl_layout.addWidget(LineWidget("右下角(px):", self._pt2_input, width=300),2,1)
        self.setLayout(ctrl_layout)

        self.setWindowTitle("中控屏位置标定")

    def closeEvent(self, event) -> None:
        self.showing = False
        self.sigCloseEvent.emit()
        self.enable_manual_editing(False)
        return super().closeEvent(event)

    def showEvent(self, event) -> None:
        self.showing = True
        return super().showEvent(event)

    def enable_manual_editing(self, i:bool):
        self.manual_enable = i
        for j in (self._pt0_input,
                self._pt1_input,
                self._pt2_input,
                self._pt3_input):
            j.setDisabled(not i)
            
    def on_reset_screen(self):
        self.enable_manual_editing(False)
        for i in (self._pt0_input,
                self._pt1_input,
                self._pt2_input,
                self._pt3_input):
            i.update_crd((0,0))
        self.sigScreenReset.emit()
        
    def _on_pos_maunally_changed(self, e):
        if self.manual_enable:
            pts = np.array([i.get_pos() for i in (self._pt0_input,
                                                    self._pt1_input,
                                                    self._pt2_input,
                                                    self._pt3_input)]).astype('float32')
            screen_dimension = self._screen_area_input.get_pos()
            # print(pts, screen_dimension)
            self.sigScreenRect.emit((pts, screen_dimension))
            

    def on_viewwidget_rect_selected(self, e):
        rect, shape = e
        h, w, _ = shape
        h*=4
        w*=4
        for cord, widget in zip(rect, (self._pt0_input,
                self._pt1_input,
                self._pt2_input,
                self._pt3_input)):
            widget.set_xy_bound(w, h)
            widget.update_crd((int(cord[0]*w), int(cord[1]*h)))
        self.enable_manual_editing(True)
        self._on_pos_maunally_changed(0)
        
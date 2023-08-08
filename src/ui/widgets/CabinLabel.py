import numpy as np
from PySide6 import QtCore, QtGui
from PySide6.QtCore import Qt, Signal
from araviq6 import NDArrayLabel


class CabinLabel(NDArrayLabel):
    sigRectPath = Signal(tuple)
    mode = 0
    rect_path = []
    def __init__(self):
        super().__init__()
        self.setMinimumSize(500, 300)
        self.pixmap_rect = [0,0, self.width(), self.height()]
        self._referrence_pixmap = QtGui.QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setArray(np.zeros((480,853,3), np.uint8))

    def setMode(self, mode:int):
        self.mode = mode
        self.rect_path = []
        self.setArray(self._array)
        if mode == 1:
            self.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        else:
            self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))

    def setArray(self, array):
        self._array = array
        super().setArray(array)

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

    def reset_selected_pos(self):
        self.setMode(1)
        

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
            self.sigRectPath.emit((self.rect_path, self._array.shape))
            self.rect_path.append(self.rect_path[0])
            # self.rect_path = []

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


        if len(self.rect_path) > 1 and len(self.rect_path) < 5:
            painter = QtGui.QPainter(pixmap)
            pen = QtGui.QPen()
            pen.setWidth(2)
            pen.setColor(QtGui.QColor('red'))
            painter.setPen(pen)
            rect_path = [(i[0] * w, i[1] * h) for i in self.rect_path]
            for i in range(len(rect_path) - 1):
                if i == len(rect_path) - 1:
                    break
                painter.drawLine(rect_path[i][0], rect_path[i][1], rect_path[i+1][0], rect_path[i+1][1])
            painter.end()
        pixmap = pixmap.scaled(
            new_w, new_h, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation
        )
        # self._original_pixmap = pixmap
        super().setPixmap(pixmap)
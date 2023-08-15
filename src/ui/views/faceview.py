from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import Signal, Qt, QEvent
import PySide6.QtGui
from PySide6.QtWidgets import (
    QWidget,  QCheckBox, QComboBox,  QDoubleSpinBox,
    QLCDNumber, QPushButton, QToolBar, QGroupBox, QSlider,
    QVBoxLayout, QHBoxLayout, QGroupBox
)
from src.ui.views.ScreenCaliWidget import ScreenCaliWidget
from src.ui.widgets.CabinLabel import CabinLabel
from src.ui.widgets.Element import VideoLabel, LineWidget, CamSelectWidget
from src.ui.widgets.Toggle import SettingWidget
import os
import time
import numpy as np

class CabinVideoLabel(VideoLabel):
    default_color = ('#CC97DECE', '#439A97')
    pressed_color = ('#CCFF9E9E', '#FF5858')

    gaze_calibrate_signal = Signal(object)

    def __init__(self, img_path):
        super().__init__()
        self.background = QtGui.QPixmap(img_path)
        self.setPixmap(self.background)
        # self.gazebrush = QtGui.QBrush(QtGui.QColor("##006EE6FF"))
        # self.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.setMouseTracking(True)
        self.current_gaze = [-1, -1]
        self.is_calibrating = False
        self.is_painting = True

    def test(self, e):
        self.paintGaze([0.5,0.5])

    def paintGaze(self, pos):
        if not self.is_painting:
            return
        radius = int(self.background.size().width()*0.2)
        # radius = 1800
        pixmap = self.background.copy()
        qp = QtGui.QPainter(pixmap)
        # qp.device()
        qp.setPen(QtGui.Qt.NoPen)
        radialGradient = QtGui.QRadialGradient(
            QtCore.QPoint(pos[1] * self.background.size().width(),
                          pos[0] * self.background.size().height()), int(radius/2))
        radialGradient.setColorAt(0.2, QtGui.QColor("#d62827"))
        # radialGradient.setColorAt(0.8, Qt.green)
        radialGradient.setColorAt(0.9, QtGui.QColor(255,255,255,0))
        qp.setBrush(QtGui.QBrush(radialGradient))
        # qp.setBrush(self.gazebrush)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        qp.drawEllipse(pos[1] * self.background.size().width() -radius/2, 
        pos[0] * self.background.size().height()-radius/2, radius, radius)
        qp.end()
        # return
        self.setPixmap(pixmap)
        self.adapt_size()
        self.update()

    def setImg(self, img_path):
        self.background = QtGui.QPixmap(img_path)
        self.setPixmap(self.background)
        self.adapt_size()

    def mouseReleaseEvent(self, e):
        if self.is_calibrating:
            p = self.mapFromGlobal(QtGui.QCursor.pos())
            nomalized_pos = (p.x()/self.size().width(), p.y()/self.size().height())
            print(nomalized_pos)
            self.gaze_calibrate_signal.emit(nomalized_pos)

    def leaveEvent(self, event) -> None:
        # 清除光标
        self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        # self.mouse.setPixmap(QtGui.QPixmap())
        return super().leaveEvent(event)
    
    def enterEvent(self, event) -> None:
        if self.is_calibrating:
            self.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        return super().enterEvent(event)
    
    def adapt_size(self, frame_size=None):
        try:
            return super().adapt_size(frame_size)
        except:
            pass

class FaceSettingView(QWidget):
    face_cam_changed_signal = Signal(int)
    def __init__(self):
        super().__init__()

        dir = './resources/cabin_imgs'
        self.all_avaliable_cabin_imgs_path = [dir+i for i in os.listdir(dir) 
            if i.endswith(".jpg") or i.endswith('.png') or i.endswith('.jpeg')]
        
        self.screen_cali_widget = ScreenCaliWidget()
        self.screen_cali_widget.sigWidgetShow.connect(self.set_cabin_cali)
        self.facebox = QGroupBox("驾驶员画面")
        self.cabinbox = QGroupBox("座舱画面")
        gaze_settingbox = QGroupBox("眼动追踪设置")
        fingersetting_box = QGroupBox("手指追踪设置")
        self.cabincalibox = QGroupBox("存储设置")
        gaze_settingbox.setMaximumWidth(250)
        fingersetting_box.setMaximumWidth(250)

        self.facebox.setLayout(self.facebox1())

        gaze_settingbox.setLayout(self.settingbox1())
        # self.facebox.setMaximumSize(640,480)

        self.cabinbox.setLayout(self.cabinbox1())
        # self.cabinbox.setMaximumSize(640,480)

        self.cabincalibox.setLayout(self.cabin_cali_box())

        fingersetting_box.setLayout(self.fingersetting())

        l1 = QHBoxLayout()
        l2 = QHBoxLayout()
        layout = QVBoxLayout()
        
        l1.addWidget(self.facebox, 5)
        l1.addWidget(self.cabinbox, 5)
        l2.addWidget(gaze_settingbox)
        l2.addWidget(fingersetting_box)
        l2.addWidget(self.cabincalibox)

        layout.addLayout(l1, 6.5)
        layout.addLayout(l2, 3.5)
        
        self.setLayout(layout)

        #手指移动计算初始化变量
        # 初始化变量
        self.num_frames = 5  # 使用多少帧来计算平均速度
        self.frame_buffer = []
        self.prev_time = time.time()
        self.prev_velocity = None
        self.prev_acceleration = 0.0

        #Dynamic Resizing
        # self.facebox.resizeEvent
        # self.toggle_cali_enable(False)

    def faceLabelUpdate(self, array):
        self.facelabel.setArray(array)
        
    def cabinLabelUpdate(self, array):
        self.cabinlabel.setArray(array)
        
    def show_cabin_cali(self):
        self.screen_cali_widget.show()
        
    def set_cabin_cali(self, e):
        self.cabinlabel.setMode(int(e))

    def fingersetting(self):
        l = QVBoxLayout()
        self.enable_finger_tracking = SettingWidget("手指实时追踪")
        self.enable_recording_overlay = SettingWidget("实时显示手指叠加")
        
        # self.open_screen_cali = QPushButton("标定中控屏区域", clicked = self.show_cabin_cali)
        # self.open_screen_cali.setFixedSize(100,30)
        # l.setAlignment(Qt.AlignmentFlag.AlignCenter)
        l.addWidget(self.enable_finger_tracking)
        l.addWidget(self.enable_recording_overlay)
        
        # l.addWidget(self.open_screen_cali)
        l.addStretch()
        return l
    

    

    def toggle_cali(self, e):
        self.cabinimg.is_calibrating = bool(e)

    def toggle_cali_enable(self, e):
        b = bool(e)
        self.cabinimg.is_painting = b
        for i in (self.img, self.btn, self.show_heatmap_btn, self.btn2):
            i.setEnabled(b)

    def cabin_cali_box(self):
        l = QVBoxLayout()
        self.finger_preview_text = QtWidgets.QLabel("")
        l.addWidget(self.finger_preview_text)
        # enable = SettingWidget("开启座舱POV映射")
        # enable.stateChanged.connect(self.toggle_cali_enable)
        # self.all_cabin_imgs = [i.split('/')[-1].split('.')[0] for i in self.all_avaliable_cabin_imgs_path]
        # self.imgselect = QComboBox()
        # self.imgselect.addItems(self.all_cabin_imgs)
        # self.imgselect.currentIndexChanged.connect(self.set_cabin_img)
        # self.img = LineWidget("座舱示意图", self.imgselect)
        # self.btn = SettingWidget("开启互动式校准")
        # self.btn.stateChanged.connect(self.toggle_cali)

        # for i in (self.img, self.btn):
        #     l.addWidget(i)
        
        l.addStretch()
        return l

    def preview_finger_status(self, res):
        if len(res) == 0:
            # 初始化变量
            self.prev_time = time.time()
            self.frame_buffer = []
            self.prev_position = None
            self.prev_velocity = None
            self.prev_acceleration = 0.0

            self.finger_preview_text.setText("未检测到食指位置")
        else:
            s = ""
            vel_acc_msg = ''
            for i in res:
                status, x, y = i
                s += f"x={round(x, 2)}mm, y={round(y,2)}mm{' 在屏幕外' if not status else ''}\n"

                current_time = time.time()
                dt = current_time - self.prev_time
                # 添加当前帧的手指位置到缓冲区
                self.frame_buffer.append((x, y))
                if len(self.frame_buffer) >= self.num_frames:
                    # 计算平均速度
                    total_displacement = 0
                    for i in range(1, len(self.frame_buffer)):
                        displacement = np.sqrt((self.frame_buffer[i][0] - self.frame_buffer[i - 1][0]) ** 2 +
                                               (self.frame_buffer[i][1] - self.frame_buffer[i - 1][1]) ** 2)
                        total_displacement += displacement
                    average_velocity = total_displacement / (self.num_frames - 1) / dt

                # 计算加速度（加速度 = 速度变化 / 时间）
                if self.prev_velocity != None:
                    acceleration = (average_velocity - self.prev_velocity) / dt

                # 更新变量
                self.prev_time = current_time
                self.prev_velocity = average_velocity
                self.prev_acceleration = acceleration

                # 移除最旧的帧
                self.frame_buffer.pop(0)

                vel_acc_msg += f"速度={average_velocity}, 加速度={acceleration}\n"


            self.finger_preview_text.setText(s)
            self.finger_preview_text.setText(vel_acc_msg)

    def settingbox1(self):
        l = QVBoxLayout()
        self.enable_gaze_tracking = SettingWidget("视线实时追踪")
        self.enable_gaze_preview = SettingWidget("视频预览视线叠加")

        l.addWidget(self.enable_gaze_tracking)
        # l.addWidget(self.enable_gaze_preview)

        l.addStretch()
        return l
    
    current_calibration_img = Signal(str)

    def set_cabin_img(self, e):
        current_img_path = self.all_avaliable_cabin_imgs_path[e]
        self.cabinimg.setImg(current_img_path)
        self.current_calibration_img.emit(current_img_path)

    def get_current_cabin(self):
        return self.all_avaliable_cabin_imgs_path[self.imgselect.currentIndex()]


    def cabinbox1(self):
        self.cabinlabel = CabinLabel()
        self.cabinlabel.sigRectPath.connect(self.screen_cali_widget.on_viewwidget_rect_selected)
        self.screen_cali_widget.sigScreenReset.connect(self.cabinlabel.reset_selected_pos)
        l = QVBoxLayout()
        self.cabincamselect = CamSelectWidget()
        l.addWidget(self.cabinlabel)
        ll = QHBoxLayout()
        self.open_screen_cali = QPushButton("标定中控屏区域", clicked = self.show_cabin_cali)
        ll.addWidget(self.open_screen_cali)
        ll.addStretch()
        ll.addWidget(self.cabincamselect.getWidget())
        l.addLayout(ll)
        return l


    def facebox1(self):
        self.facelabel = VideoLabel()
        self.facelayout = QVBoxLayout()
        self.facecamselect = CamSelectWidget()
        self.facelayout.addWidget(self.facelabel)
        ll = QHBoxLayout()
        ll.addStretch()
        ll.addWidget(self.facecamselect.getWidget())
        self.facelayout.addLayout(ll)
        return self.facelayout


    def showEvent(self, e):
        print("切换到驾驶员设置界面")
        pass
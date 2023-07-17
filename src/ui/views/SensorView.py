from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget,  QLabel,
    QCheckBox, QComboBox,  QDoubleSpinBox,
    QLCDNumber, QPushButton, QSpinBox, 
    QToolBar, QGroupBox, QSlider,
    QVBoxLayout, QHBoxLayout, QProgressBar, QInputDialog
)
import pickle
from src.ui.widgets.Element import LineWidget, Dial
from src.ui.widgets.Toggle import SettingWidget
import numpy as np

class IMUSensorGUI(QtCore.QObject):
    #屎山代码
    imus = []
    sensor_type_changed_signal = Signal(object)
    start_process_signal = Signal(bool)
    sensor_alias = {}

    sensor_type = ('无', '方向盘', '制动踏板', '加速踏板')
    avaliable_sensor_type = ('无', '方向盘', '制动踏板', '加速踏板')


    def __init__(self):
        super().__init__()
        self.sensorlayout = IMUSensorLayout()
        try:
            with open('./resources/imusensors.config', 'rb') as handle:
                self.sensor_alias = pickle.load(handle)
        except:
            self.sensor_alias = {}

    def enable_IMU(self, i):
        i = bool(i)
        if i == False:
            self.device_removal(None, all=True)
            self.start_process_signal.emit(i)
            self.ble_state_signal.emit(False)
        self.wheel_box.setEnabled(i)
        self.imu_box.setEnabled(i)
        self.pedal_box.setEnabled(i)
        self.startscanIMU.setEnabled(i)
        self.stopscanIMU.setEnabled(i)
        self.resetIMU.setEnabled(i)
        self.blestate.setEnabled(i)

    def device_data_update(self, data):
        if data['address'] == self.current_displaying_imu:
            y = np.sin(np.radians((float(data['pitch']))))
            x = np.sin(np.radians((float(data['roll']))))
            self.preview_dial.move([y, x])
            
    def device_status_update(self, info):
        exists = False
        for i in self.imus:
            if i.mac == info['address']:
                exists = True
        if info['status'] is True and exists is False:
            alias = self.sensor_alias[info['address']]  if info['address'] in self.sensor_alias.keys() else "未命名"
            imu = IMUSensor(info['address'], alias)
            imu.selected_signal.connect(self.handle_imu_select)
            imu.type_changed_signal.connect(self.handle_type_changed)
            imu.rename_signal.connect(self.handle_rename)
            imu.rate_changed_signal.connect(lambda e: self.sensor_rate_changed_signal.emit(e))
            imu.delete_signal.connect(self.delete_imu)
            self.imus.append(imu)
            self.sensorlayout.addWidget(imu)
        elif info['status'] is False and exists is True:
            self.delete_imu(info['address'])
            
            
    
    def device_battery_update(self, info):
        for i in self.imus:
            if i.mac == info['address']:
                i.battery = info['voltage']
                i.update_gui()
        pass

    imu_device_removed_signal = Signal(object)
    
    def delete_imu(self, mac):
        #从bleman中删除
        self.imu_device_removed_signal.emit(mac)
        #从UI中删除
        for i, imu in enumerate(self.imus):
            if imu.mac == mac:
                self.sensorlayout.itemAt(i).widget().deleteLater()
                # self.sensorlayout.removeItem(self.sensorlayout.itemAt(i))
                self.imus.pop(i)
                return


    def device_removal(self, imu_info, all=False):
        if all == True:
            # print(self.imus)
            for i in reversed(range(self.sensorlayout.count())): 
                self.sensorlayout.itemAt(i).widget().setParent(None)
            for imu in self.imus:
                self.imu_device_removed_signal.emit(imu.mac)
            self.imus = []
            return

        for i, imu in enumerate(self.imus):
            if imu.mac == imu_info.name:
                self.sensorlayout.itemAt(i).widget().deleteLater()
                # self.sensorlayout.removeItem(self.sensorlayout.itemAt(i))
                self.imus.pop(i)
                break

    def steer_update(self, data):
        self.wheel.rotate(data['steeringwheel_deg'])
        self.wheel_degs.setText(f"方向盘转动角度: {round(data['steeringwheel_deg'], 1)}度")

    def acc_update(self, data):
        i = data['peal_percent'] * 100
        if i > 100:
            i = 100
        elif i < 0:
            i = 0
        self.acc.setValue(i)
        pass

    def brake_update(self, data):
        i = data['peal_percent'] * 100
        if i > 100:
            i = 100
        elif i < 0:
            i = 0
        self.brake.setValue(i)
        pass

    sensor_rate_changed_signal = Signal(object)


    def handle_rename(self, e):
        mac, alias = e
        self.sensor_alias[mac] = alias
        with open('./resources/imusensors.config', 'wb') as handle:
            pickle.dump(self.sensor_alias, handle)

    def handle_type_changed(self, e):
        mac, t = e
        self.sensor_type_changed_signal.emit((mac, t))
        all_used_types = []
        for imu in self.imus:
            if imu.typesel.currentText() != "无":
                all_used_types.append(imu.typesel.currentText())
        all_used_types = set(all_used_types)
        remain_types = [i for i in self.sensor_type if i not in all_used_types]
        # print("剩余类型：",remain_types)
        # print("已选类型：",all_used_types)
        for imu in self.imus:
            for tt in all_used_types:
                #移除所有重复的
                if tt != imu.typesel.currentText():
                    imu.typesel.removeItem(imu.typesel.findText(tt))
            for tt in remain_types:
                if tt != "无" and imu.typesel.findText(tt) == -1:
                    imu.typesel.addItem(tt)

    current_displaying_imu = None

    def handle_imu_select(self, e):
        for i in self.imus:
            if i.mac == e:
                if not i.is_selected:
                    i.selected()
                    self.current_displaying_imu = e
                else:
                    i.deselected()
                    self.preview_dial.move([0,0])
                    self.current_displaying_imu = None
            else:
                i.deselected()
        pass

    ble = False
    ble_state_signal = Signal(bool)

    def ble_state(self, _):
        if self.ble == False:
            self.ble = True
            self.blestate.setText("停止蓝牙扫描")
            self.ble_state_signal.emit(True)
        elif self.ble == True:
            self.ble = False
            self.blestate.setText("开启蓝牙扫描")
            self.ble_state_signal.emit(False)
            # self.device_removal(_, all=True)
        self.start_process_signal.emit(self.ble)
            

    def imupreviewbox(self):
        self.imu_preview_box = QGroupBox("传感器预览")
        l = QVBoxLayout()
        self.preview_dial = Dial()
        l.addWidget(self.preview_dial)
        self.imu_preview_box.setLayout(l)
        return self.imu_preview_box

    def imusettingbox(self):
        box = QGroupBox("传感器设置")
        l = QVBoxLayout()
        self.enableIMU = SettingWidget("启用")
        self.startscanIMU = QPushButton("开始扫描")
        self.stopscanIMU = QPushButton("关闭扫描")
        self.resetIMU = QPushButton("重置链接")
        refreshrate = QComboBox()

        self.blestate = QPushButton("开启蓝牙扫描")
        self.blestate.clicked.connect(self.ble_state)


        self.enableIMU.stateChanged.connect(self.enable_IMU)
        self.enableIMU.toggle.setChecked(False)

        l.addWidget(self.enableIMU)
        l.addWidget(self.blestate)
        # l.addWidget(self.startscanIMU)
        # l.addWidget(self.stopscanIMU)
        # l.addWidget(self.resetIMU)
        # l.addWidget(refreshrateset)
        l.addStretch()
        
        box.setLayout(l)
        return box

    def pedalbox(self):
        self.pedal_box = QGroupBox("踏板")
        l = QVBoxLayout()
        self.brake = QProgressBar()
        self.brake.setTextVisible(True)
        self.brake.setRange(0,100)
        # self.brake.setValue(50)
        self.brake.setStyleSheet("""
        QProgressBar::chunk {
            background-color: #ea6440;
        }
        """)

        self.acc = QProgressBar()
        self.acc.setTextVisible(True)
        self.acc.setRange(0,100)
        # self.acc.setValue(50)
        self.acc.setStyleSheet("""
        QProgressBar::chunk {
            background-color: #00abb3;
        }
        """)

        self.brake0 = QPushButton("标定零点")
        self.brake1 = QPushButton("标定行程")
        self.acc0 = QPushButton("标定零点")
        self.acc1 = QPushButton("标定行程")
        self.brake0.setFixedWidth(120)
        self.brake1.setFixedWidth(120)
        self.acc0.setFixedWidth(120)
        self.acc1.setFixedWidth(120)

        brake_w = LineWidget("制动踏板", self.brake)
        acc_w = LineWidget("加速踏板", self.acc)

        ll = QHBoxLayout()
        ll.addWidget(self.brake0)
        ll.addWidget(self.brake1)
        lll = QHBoxLayout()
        lll.addWidget(self.acc0)
        lll.addWidget(self.acc1)
        l.addWidget(brake_w)
        l.addLayout(ll)
        l.addWidget(acc_w)
        l.addLayout(lll)

        l.setContentsMargins(40,20,40,20)
        self.pedal_box.setLayout(l)
        return self.pedal_box

    def wheelbox(self):
        self.wheel_box = QGroupBox("方向盘")
        l = QVBoxLayout()
        self.wheel = Dial()
        self.wheel_degs = QPushButton("方向盘转动角度: 未初始化")
        l.addWidget(self.wheel)
        ll = QHBoxLayout()
        ll.addStretch()
        ll.addWidget(self.wheel_degs)
        ll.addStretch()
        l.addLayout(ll)
        self.wheel_box.setLayout(l)
        return self.wheel_box

    def imu_displaybox(self):
        self.imu_box = QGroupBox("传感器列表")
        self.imu_box.setLayout(self.sensorlayout)
        # notice = QLabel("当前没有任何蓝牙传感器连接")
        # self.sensorlayout.addWidget(notice)
        return self.imu_box

class IMUSensorLayout(QtWidgets.QHBoxLayout):
    imus = []
    def __init__(self):
        super().__init__()
        self.setContentsMargins(10,10,10,10)


class IMUSensor(QGroupBox):
    reportRate = [
        "10", "20", "50", "100"
    ]

    selected_signal = Signal(object)
    type_changed_signal = Signal(object)
    rename_signal = Signal(object)
    rate_changed_signal = Signal(object)

    def __init__(self, info, alias):
        super().__init__(alias)
        self.setMaximumSize(200,200)
        self.setStyleSheet("padding: 50px;")
        self.alias = alias
        self.mac = info

        self.voltagebar = QProgressBar()
        self.voltagebar.setTextVisible(True)
        self.voltagebar.setRange(320, 410)
        # bar.setValue(390)
        self.voltagebar.setStyleSheet("""
        QProgressBar::chunk {
            background-color: #48cc5f;
        }
        """)
        self.voltagebar.setMaximumWidth(70)

        self.typesel = QComboBox()
        self.typesel.addItems(['无', '方向盘', '制动踏板', '加速踏板'])
        self.typesel.setCurrentIndex(0)
        self.typesel.currentTextChanged.connect(self.type_changed)

        self.rate = QComboBox()
        self.rate.addItems(self.reportRate)
        self.rate.setCurrentIndex(0)
        self.rate.setFixedWidth(60)
        self.rate.currentTextChanged.connect(lambda rate: self.rate_changed_signal.emit((self.mac, rate)))


        self.macd = LineWidget("MAC地址：", QLabel(), self.mac)
        self.typed = LineWidget("传感器类型", self.typesel)
        self.batd = LineWidget("参考电量", self.voltagebar)
        self.rated = LineWidget("回报率(Hz)", self.rate)

        rename = QPushButton("重命名", clicked=self.rename)

        delete = QPushButton("移除", clicked=lambda : self.delete_signal.emit(self.mac))


        l = QVBoxLayout()
        l.addWidget(self.typed)
        l.addWidget(self.macd)
        l.addWidget(self.batd)
        l.addWidget(self.rated)

        ll = QHBoxLayout()

        ll.addWidget(rename)
        ll.addStretch()
        ll.addWidget(delete)

        l.addLayout(ll)

        self.setLayout(l)
        self.setStyleSheet("""
        QGroupBox {
            border-radius: 10px;
            border-width: 3;
            background-color: #23272e;
            border-color: #3f4042;
        }
        """)


    def type_changed(self, text):
        self.type_changed_signal.emit((self.mac, text))

    delete_signal = Signal(object)

    # def delete(self, e):
    #     self.delete_signal.emit(self.mac)


    def rename(self):
        name, ok = QInputDialog.getText(self, "设置传感器名称", "传感器名称:")
        if ok:
            self.alias = name
            self.rename_signal.emit((self.mac, name))
            self.update_gui()

    def update_info(self, info):
        self.battery = info.voltage
        self.type = info.type
        self.update_gui()
        
    def update_gui(self):
        self.setTitle(self.alias)
        try:
            self.voltagebar.setValue(int(self.battery))
        except:
            pass
    
    is_selected = False
    def selected(self):
        self.setStyleSheet("""
        QGroupBox {
            border-radius: 10px;
            border-width: 3;
            background-color: #23272e;
            border-color: beige;
        }
        """)
        self.is_selected = True
        pass

    def deselected(self):
        self.setStyleSheet("""
        QGroupBox {
            border-radius: 10px;
            border-width: 3;
            border-color: #3f4042;
            background-color: #23272e;
        }
        """)
        self.is_selected = False
        pass

    def mousePressEvent(self, event) -> None:
        self.selected_signal.emit(self.mac)
        

class SensorView(QWidget):
    def __init__(self):
        super().__init__()
        
        self.imugui = IMUSensorGUI()

        imu_box = QGroupBox("运动传感器管理")
        gps_box = QGroupBox("GPS/北斗传感器管理(未完成)")
        phone_box = QGroupBox("手机连接管理")
        wheel_box = self.imugui.wheelbox()
        pedal_box = self.imugui.pedalbox()
        speed_box = QGroupBox("定位信息")
        imu_box.setLayout(self.imubox())
        gps_box.setLayout(self.gpsbox())
        phone_box.setLayout(self.phonebox())
        speed_box.setLayout(self.speedbox())

        l1 = QHBoxLayout()
        l2 = QHBoxLayout()
        l1.addWidget(wheel_box, 3.5)
        l1.addWidget(pedal_box, 3.5)
        l1.addWidget(speed_box, 3)
        l2.addWidget(imu_box, 7.6)
        l3 = QVBoxLayout()
        l3.addWidget(gps_box,5)
        l3.addWidget(phone_box,5)
        l2.addLayout(l3, 2.4)

        layout = QVBoxLayout()
        layout.addLayout(l1, 3)
        layout.addLayout(l2, 7)
        self.setLayout(layout)
        self.imugui.enable_IMU(False)

    def speedbox_display(self, msg):
        self.latitude.setText(f"{round(msg['latitude'], 4)}度")
        self.longitude.setText(f"{round(msg['longitude'], 4)}度")
        self.altitude.setText(f"{round(msg['altitude'], 2)}米")
        self.accu.setText(f"{round(msg['accuracy'], 1)}米")
        if msg['heading'] is not None:
            self.direction.setText(f"{round(msg['heading'], 0)}度")
            self.car_direction.rotate(msg['heading'])
        if msg['speed'] is not None:
            self.speed.setText(f"{round(msg['speed'], 2)}m/s")

    def speedbox(self):

        l = QHBoxLayout()
        ll = QVBoxLayout()

        self.longitude = LineWidget("经度：", QLabel(), "无数据")
        self.latitude = LineWidget("纬度：", QLabel(), "无数据")
        self.speed = LineWidget("地速：", QLabel(), "无数据")
        self.altitude = LineWidget("海拔：", QLabel(), "无数据")
        self.direction = LineWidget("指向：", QLabel(), "无数据")
        self.accu = LineWidget("精确度：", QLabel(), "无数据")
        self.car_direction = Dial()
        ll.addStretch()
        for i in (self.longitude, self.latitude, self.speed, self.altitude, self.direction, self.accu):
            ll.addWidget(i)
        ll.addStretch()
        # l.addStretch()
        l.addLayout(ll)
        l.addWidget(self.car_direction)
        l.setContentsMargins(90,30,30,30)
        return l


    def phonebox(self):
        l = QVBoxLayout()
        self.enableWSS = SettingWidget("启用")
        phonesite = LineWidget("手机网址：", QLabel(), "https://jhkjux.com/active")
        self.phoneserver = LineWidget("服务器：", QLabel(), "尚未连接")
        notice = QLabel("<strong>您可以使用智能手机获取定位信息，替代GPS传感器。</strong>")
        notice.setWordWrap(True)
        # l.addWidget(self.phone_toggle)
        l.addStretch()
        l.addWidget(notice)
        l.addWidget(phonesite)
        l.addWidget(self.phoneserver)
        l.addWidget(self.enableWSS)
        l.addStretch()
        return l

    def imubox(self):
        imusettingsbox = self.imugui.imusettingbox()
        imupreviewbox = self.imugui.imupreviewbox()
        imu_displaybox = self.imugui.imu_displaybox()
        l = QHBoxLayout()
        ll = QVBoxLayout()
        ll.addWidget(imusettingsbox)
        ll.addWidget(imupreviewbox)
        l.addLayout(ll,2)
        l.addWidget(imu_displaybox, 8)
        return l

    def gpsbox(self):
        l = QVBoxLayout()
        com = QComboBox()
        self.com_select = LineWidget("COM端口", com)
        enable = SettingWidget("启用")
        cali = QPushButton("惯导调零")
        cali.setFixedWidth(100)

        l.addWidget(self.com_select)
        l.addWidget(enable)
        l.addWidget(cali)
        l.addStretch()
        
        return l


    def showEvent(self, event):
        print("切换到传感器设置界面")
        # if self.listener_running:
        #     self.pipe_end.send("CloseCSharp")
        # self.pipe_end.send("RestartCSharp")
        pass
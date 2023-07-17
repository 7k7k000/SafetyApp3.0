from src.src.sensors.SensorHub import SensorHub
from src.src.sensors.SensorModel import DataModel
from src.src.utils.Async import QAsyncWorkerThread
from src.ui.views.SensorView import SensorView
from PySide6.QtWidgets import (
    QWidget,
    QPushButton,
    QApplication,
    QVBoxLayout,
    QMainWindow,
)
import sys
import qdarktheme

def print_debug(e):
    print(e)

def sensor_management(hub:SensorHub, view:SensorView, model: DataModel):
    hub.sigConnection.connect(view.imugui.device_status_update)
    hub.sigBattery.connect(view.imugui.device_battery_update)
    hub.sigMotion.connect(view.imugui.device_data_update)
    hub.sigMotion.connect(model.on_data_receive)
    hub.sigGeolocation.connect(view.speedbox_display)
    view.imugui.ble_state_signal.connect(hub.scan)
    view.imugui.imu_device_removed_signal.connect(hub.stop_device)
    view.imugui.sensor_rate_changed_signal.connect(hub.change_sensor_rate)
    view.enableWSS.stateChanged.connect(hub.serve_wss)
    
    view.imugui.sensor_type_changed_signal.connect(model.on_sensor_type_change)
    view.imugui.wheel_degs.clicked.connect(model.wheel.calibrate)
    model.wheel.sigDataOutput.connect(view.imugui.steer_update)
    model.brake.sigDataOutput.connect(view.imugui.brake_update)
    model.acc.sigDataOutput.connect(view.imugui.acc_update)
    
    view.imugui.brake0.clicked.connect(lambda: model.brake.calibrate(0))
    view.imugui.brake1.clicked.connect(lambda: model.brake.calibrate(1))
    view.imugui.acc0.clicked.connect(lambda: model.acc.calibrate(0))
    view.imugui.acc1.clicked.connect(lambda: model.acc.calibrate(1))
    
    
app = QApplication(sys.argv)
qdarktheme.setup_theme()
hub = SensorHub()
thread = QAsyncWorkerThread()
hub.moveToThread(thread)
thread.start()
view = SensorView()
model = DataModel()
sensor_management(hub, view, model)

view.show()
# hub.sigSensorMsg.connect(print_debug)
# window = QMainWindow()
# window.setCentralWidget(QWidget())
# window.centralWidget().setLayout(QVBoxLayout())

# window.centralWidget().layout().addWidget(QPushButton(text="Start", clicked=hub.start_scanning))  # type: ignore
# window.centralWidget().layout().addWidget(QPushButton(text="Stop", clicked=hub.stop_scanning))  # type: ignore
# window.centralWidget().layout().addWidget(QPushButton(text="Test", clicked=hub.test))  # type: ignore
# # hub.start_scanning()
# window.show()
app.exec()
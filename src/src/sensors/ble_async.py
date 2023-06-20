import asyncio
import time
import numpy as np
from bleak import BleakScanner, BleakClient
from PySide6.QtCore import QObject, Signal, Slot, QThread
from qasync import QEventLoop

class QAsyncWorkerThread(QThread):
    def run(self):
        loop = QEventLoop(self)
        asyncio.set_event_loop(loop)
        loop.run_forever()
        
class Sensor(QObject):
    notify_uuid = "0000ffe4-0000-1000-8000-00805f9a34fb"
    write_uuid = "0000ffe9-0000-1000-8000-00805f9a34fb"
    voltage_req = b'\xFF\xAA\x27\x64\x00'
    
    def __init__(self, device) -> None:
        super().__init__()
        self.device = device
        
    @Slot()
    def start(self):
        asyncio.create_task(self.connect_sensor())
    
    @Slot()
    def battery(self):
        print("battery1")
        asyncio.create_task(self._get_battery_percent())
    
    @Slot()
    def stop(self):
        print(f"传感器{self.device.address}断开连接")
        self.stop_event.set()
        
    async def connect_sensor(self):
        self.stop_event = asyncio.Event()
        async with BleakClient(address_or_ble_device=self.device,
                                disconnected_callback=self._disconnected_callback) as self.client:
                await self.client.start_notify(self.notify_uuid, self._notification_handler)
                await self.stop_event.wait()
        
    def _notification_handler(self, sender, data: bytearray):
        # print(self.device.address, time.time())
        if data[1] == 0x61:
            self._decode_data(data)
        else:
            if data[2] == 0x64:
                self._decode_battery(data)
                
            
    async def _get_battery_percent(self):
        try:
            self.client.address
        except:
            return
        await self.client.write_gatt_char(
            self.write_uuid, 
            self.voltage_req)
        print("battery2")
        
    def _decode_data(self, data:bytearray):
        header_bit = data[0]
        assert header_bit == 0x55
        flag_bit = data[1] # 0x51 or 0x71
        assert flag_bit == 0x61 or flag_bit == 0x71
        # 2 byteずつ取り出して、後ろのbyteが上位ビットになるようにsigned shortに変換
        decoded = [int.from_bytes(data[i:i+2], byteorder='little', signed=True) for i in range(2, len(data), 2)]
        intstr = [int(i) for i in data]
        # print(bytearray(intstr))
        # signed shortの桁数なので、-32768~32767の範囲になる
        # なので一旦正規化してから各単位に合わせる
        ax    = decoded[0] / 32768.0 * 16 * 9.8
        ay    = decoded[1] / 32768.0 * 16 * 9.8
        az    = decoded[2] / 32768.0 * 16 * 9.8
        wx    = decoded[3] / 32768.0 * 2000
        wy    = decoded[4] / 32768.0 * 2000
        wz    = decoded[5] / 32768.0 * 2000
        roll  = decoded[6] / 32768.0 * 180
        pitch = decoded[7] / 32768.0 * 180
        yaw   = decoded[8] / 32768.0 * 180
        # print(f"ax: {ax:.3f}, ay: {ay:.3f}, az: {az:.3f}, wx: {wx:.3f}, wy: {wy:.3f}, wz: {wz:.3f}, roll: {roll:.3f}, pitch: {pitch:.3f}, yaw: {yaw:.3f}")
        # self.current_data = (roll * np.pi / 180, pitch * np.pi / 180, yaw * np.pi / 180) # roll, pitch, yawをラジアンに変換して格納

    def _decode_battery(self, data):
        data = [str(hex(i)).split('x')[1] for i in data[4:6]]
        voltage = int(data[1]+data[0], base=16)
        print(voltage)
        pass
    def _disconnected_callback(self, client):
        self.stop()
        
class SensorHub(QObject):
    def __init__(self) -> None:
        super().__init__()
        self.sensors = {}
        print(asyncio.get_event_loop())
        print("SensorHub已启动")
    
    @Slot()
    def start_scanning(self):
        print("开始扫描蓝牙设备...")
        asyncio.create_task(self.scan_devices())
        
    @Slot()
    def test(self):
        for i in self.sensors.values():
            i.battery()
        
    @Slot()
    def stop_scanning(self):
        print("蓝牙设备扫描结束!")
        try:
            self.stop_event.set()
        except:
            pass
    
    def _on_ble_device_detection(self, device, _):
        try:
            if device.name.startswith("WT901BLE") and device.address not in self.sensors:
                print("gotcha!", device)
                self.sensors[device.address] = Sensor(device)
                self.sensors[device.address].start()
                
        except:
            pass
        
    async def scan_devices(self):
        self.stop_event = asyncio.Event()
        async with BleakScanner(self._on_ble_device_detection) as self.scanner:
            # Important! Wait for an event to trigger stop, otherwise scanner
            # will stop immediately.
            await self.stop_event.wait()
    

    
if __name__ == "__main__":
    from PySide6.QtWidgets import QWidget, QPushButton, QApplication, QVBoxLayout, QMainWindow
    from PySide6.QtCore import QObject, Signal, Slot
    import sys
    app = QApplication(sys.argv)
    hub = SensorHub()
    thread = QAsyncWorkerThread()
    hub.moveToThread(thread)
    thread.start()
    window = QMainWindow()
    window.setCentralWidget(QWidget())
    window.centralWidget().setLayout(QVBoxLayout())

    window.centralWidget().layout().addWidget(QPushButton(text='Start', clicked=hub.start_scanning)) # type: ignore
    window.centralWidget().layout().addWidget(QPushButton(text='Stop', clicked=hub.stop_scanning)) # type: ignore
    window.centralWidget().layout().addWidget(QPushButton(text='Test', clicked=hub.test)) # type: ignore
    # hub.start_scanning()
    window.show()
    app.exec()
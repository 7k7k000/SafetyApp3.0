from src.src.sensors.BLESensor import BLESensor


from PySide6.QtCore import QObject, Signal, Slot
from bleak import BleakScanner


import asyncio


class SensorHub(QObject):
    sigSensorMsg = Signal(str)

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
            if (
                device.name.startswith("WT901BLE")
                and device.address not in self.sensors
            ):
                print("gotcha!", device)
                self.sensors[device.address] = BLESensor(device)
                self.sensors[device.address].sigDebugMsg.connect(self._on_sensor_msg)
                self.sensors[device.address].start()

        except:
            pass

    def _on_sensor_msg(self, e):
        self.sigSensorMsg.emit(e)

    async def scan_devices(self):
        self.stop_event = asyncio.Event()
        async with BleakScanner(self._on_ble_device_detection) as self.scanner:
            # Important! Wait for an event to trigger stop, otherwise scanner
            # will stop immediately.
            await self.stop_event.wait()

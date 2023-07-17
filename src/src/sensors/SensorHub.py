from src.src.sensors.BLESensor import BLESensor


from PySide6.QtCore import QObject, Signal, Slot
from bleak import BleakScanner
import ssl
import json
import websockets
import socket
import asyncio

class SensorHub(QObject):
    sigSensorMsg = Signal(str)
    sigBattery = Signal(dict)
    sigMotion = Signal(dict)
    sigGeolocation = Signal(dict)
    sigConnection = Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self.sensors = {}
        print(asyncio.get_event_loop())
        print("SensorHub已启动")
        
    @Slot()
    def scan(self, res):
        if res:
            self.start_scanning()
        else:
            self.stop_scanning()

    @Slot()
    def start_scanning(self):
        asyncio.create_task(self.scan_devices())
        
    @Slot()
    def change_sensor_rate(self, res):
        mac, rate = res
        if mac in self.sensors:
            self.sensors[mac].freq(rate)

    @Slot()
    def stop_scanning(self):
        try:
            self.ble_scan_stop_event.set()
        except:
            pass
        
    @Slot()
    def stop_device(self, id):
        print("Shutting down",id)
        self.sensors[id].stop()
        
    @Slot()
    def serve_wss(self, e):
        if e:
            asyncio.create_task(self.wss_server())
        else:
            self.wss_stop_event.set()

    def _on_ble_device_detection(self, device, _):
        try:
            if device.name.startswith("WT901BLE"):
                if device.address in self.sensors and self.sensors[device.address].active is True:
                    return
                print("gotcha!", device)
                self.sensors[device.address] = BLESensor(device)
                self.sensors[device.address].sigDebugMsg.connect(self._on_sensor_msg)
                self.sensors[device.address].start()
                self.sensors[device.address].sigConnection.connect(lambda x: self.sigConnection.emit(x))
                self.sensors[device.address].sigBattery.connect(lambda x: self.sigBattery.emit(x))
                self.sensors[device.address].sigMotion.connect(lambda x: self.sigMotion.emit(x))

        except:
            pass

    def _on_sensor_msg(self, e):
        self.sigSensorMsg.emit(e)

    async def scan_devices(self):
        self.ble_scan_stop_event = asyncio.Event()
        async with BleakScanner(self._on_ble_device_detection) as self.scanner:
            # Important! Wait for an event to trigger stop, otherwise scanner
            # will stop immediately.
            print("开始扫描设备")
            await self.ble_scan_stop_event.wait()
            print("设备扫描已正确结束")
            
    async def wss_server(self):
        self.wss_stop_event = asyncio.Event()
        self.ip = self.local_ip()
        print(self.ip)
        self._ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self._ssl_context.load_cert_chain(
        './resources/ssl/snakeoil.pem', keyfile='./resources/ssl/snakeoil.key')
        async with websockets.serve(self._on_wss_connection, self.ip, 8765, ssl=self._ssl_context):
            print(f"WebSocket Secure Serving @ {self.ip}:8765")
            await self.wss_stop_event.wait()
            print("WebSocket Secure Serving Terminated")
            
    async def _on_wss_connection(self, ws):
        while not self.wss_stop_event.is_set():
            msg = await ws.recv()
            try:
                msg = json.loads(msg)
                #Ignore Heartbeat
                if msg['type'] == 'Geolocation':
                    self.sigGeolocation.emit(msg)
            except:
                pass
        print(ws)
            
    def local_ip(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            # doesn't even have to be reachable
            s.connect(('10.254.254.254', 1))
            ip = s.getsockname()[0]
        except Exception:
            ip = '127.0.0.1'
        finally:
            s.close()
        return ip
